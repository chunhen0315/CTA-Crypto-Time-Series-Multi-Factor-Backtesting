import ast
import html
import json
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import read_json_html as report_lib
from config import (
    CORRELATION_ALPHA_IDS,
    CORRELATION_END_DATE,
    CORRELATION_MIN_OBSERVATIONS,
    CORRELATION_OUTPUT_FOLDER,
    CORRELATION_OUTPUT_HTML,
    CORRELATION_PARAM_SOURCE,
    CORRELATION_RESAMPLE,
    CORRELATION_START_DATE,
    CORRELATION_THRESHOLD,
    CORRELATION_USE_ABSOLUTE,
    JSON_FILE,
    OUTPUT_FOLDER_BACKTEST,
    PARAM_WINDOW,
    PARAM_THRESHOLD_1,
    PARAM_THRESHOLD_2,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

OUTPUT_DIR = Path(CORRELATION_OUTPUT_FOLDER)
TPE_DIR = Path(OUTPUT_FOLDER_BACKTEST) / "tpe_permutation"


def parse_logic_and_side(entry_exit_logic):
    entry_exit = str(entry_exit_logic or "").lower()
    if "_long" in entry_exit:
        return entry_exit.replace("_long", ""), "long"
    if "_short" in entry_exit:
        return entry_exit.replace("_short", ""), "short"
    return entry_exit, "both"


def normalize_alpha_id(value):
    text = str(value).strip()
    if text.isdigit():
        return f"TURTLE003_{int(text):05d}"
    return text


def requested_alpha_ids(all_alphas):
    if CORRELATION_ALPHA_IDS is None:
        return [alpha["alpha_id"] for alpha in all_alphas if alpha.get("alpha_id")]

    if isinstance(CORRELATION_ALPHA_IDS, str):
        value = CORRELATION_ALPHA_IDS.strip()
        if not value or value.lower() in {"none", "all"}:
            return [alpha["alpha_id"] for alpha in all_alphas if alpha.get("alpha_id")]
        raw_ids = [value]
    else:
        raw_ids = CORRELATION_ALPHA_IDS

    return list(
        dict.fromkeys(
            normalize_alpha_id(value)
            for value in raw_ids
            if value is not None and str(value).strip()
        )
    )


def get_thresholds(alpha):
    values = [
        alpha.get("long_entry_threshold", alpha.get("threshold_1", 0.0)),
        alpha.get("long_exit_threshold", 0.0),
        alpha.get("short_entry_threshold", alpha.get("threshold_2", 0.0)),
        alpha.get("short_exit_threshold", 0.0),
    ]
    values = [float(value) for value in values]
    return max(values), min(values)


def apply_json_param_overrides(window, threshold_1, threshold_2):
    if PARAM_WINDOW is not None:
        window = int(PARAM_WINDOW)
    if PARAM_THRESHOLD_1 is not None:
        threshold_1 = float(PARAM_THRESHOLD_1)
    if PARAM_THRESHOLD_2 is not None:
        threshold_2 = float(PARAM_THRESHOLD_2)
    return window, threshold_1, threshold_2


def json_param_source_label():
    overrides = []
    if PARAM_WINDOW is not None:
        overrides.append("window")
    if PARAM_THRESHOLD_1 is not None:
        overrides.append("threshold_1")
    if PARAM_THRESHOLD_2 is not None:
        overrides.append("threshold_2")
    if overrides:
        return "alpha JSON + config override (" + ", ".join(overrides) + ")"
    return "alpha JSON"


def load_best_tpe_params(alpha_id):
    path = TPE_DIR / alpha_id / f"{alpha_id}_tpe_filtered.csv"
    if not path.exists():
        raise FileNotFoundError(f"Filtered TPE result not found: {path}")

    filtered_df = pd.read_csv(path, skipinitialspace=True)
    filtered_df.columns = filtered_df.columns.str.strip()
    filtered_df = filtered_df.dropna(subset=["VT_SR"])
    if filtered_df.empty:
        raise ValueError("no filtered row with a VT_SR value")

    return filtered_df.loc[filtered_df["VT_SR"].astype(float).idxmax()]


def strategy_params(alpha):
    alpha_id = alpha["alpha_id"]
    if CORRELATION_PARAM_SOURCE == "tpe_best_vt":
        row = load_best_tpe_params(alpha_id)
        return {
            "model": str(row["model"]).strip(),
            "logic": str(row["logic"]).strip(),
            "side": str(row["side"]).strip().lower(),
            "window": int(float(row["window"])),
            "threshold_1": float(row["threshold_1"]),
            "threshold_2": float(row["threshold_2"]),
            "param_source": "best filtered validation SR",
        }

    if CORRELATION_PARAM_SOURCE != "json":
        raise ValueError(
            f"Unsupported CORRELATION_PARAM_SOURCE: {CORRELATION_PARAM_SOURCE}"
        )

    logic, side = parse_logic_and_side(alpha.get("entry_exit_logic"))
    threshold_1, threshold_2 = get_thresholds(alpha)
    window = int(alpha.get("window", alpha.get("rolling_window_1", 0)))
    window, threshold_1, threshold_2 = apply_json_param_overrides(
        window,
        threshold_1,
        threshold_2,
    )
    return {
        "model": alpha.get("model"),
        "logic": logic,
        "side": side,
        "window": window,
        "threshold_1": threshold_1,
        "threshold_2": threshold_2,
        "param_source": json_param_source_label(),
    }


def load_factor_frame(alpha):
    datasource_raw = alpha.get("datasource_structure", "{}")
    try:
        datasources = json.loads(datasource_raw.replace("'", '"'))
    except Exception:
        datasources = ast.literal_eval(datasource_raw)

    merged_df = None
    for key, value in datasources.items():
        source_df = report_lib.load_csv(
            value.get("topic"),
            value.get("column"),
            key,
        )
        if source_df is None:
            continue
        source_df = source_df.dropna(subset=[key])
        merged_df = (
            source_df
            if merged_df is None
            else pd.merge(merged_df, source_df, on="time", how="inner")
        )

    if merged_df is None:
        raise ValueError("no datasource could be loaded")

    merged_df = merged_df.drop(
        columns=[column for column in merged_df.columns if "start_time" in column]
    )
    merged_df = merged_df.set_index("time")
    formula = alpha.get("manual_alpha_formula") or alpha.get("alpha_formula", "")
    merged_df, rewritten_formula = report_lib.apply_diff_from_formula(
        merged_df,
        formula,
    )
    for column in merged_df.columns:
        merged_df[column] = pd.to_numeric(merged_df[column], errors="coerce")

    signal = report_lib.compute_alpha_signal(merged_df, rewritten_formula)
    if signal is None:
        raise ValueError("alpha signal computation failed")

    merged_df["factor"] = signal.replace([np.inf, -np.inf], np.nan)
    factor_min = merged_df["factor"].min()
    decimal_places = 20 if pd.notna(factor_min) and abs(factor_min) < 1.0 else 10
    merged_df["factor"] = merged_df["factor"].round(decimal_places)
    if alpha.get("data_preprocessing") == "diff":
        merged_df["factor"] = merged_df["factor"].diff()
    return merged_df.dropna()


def build_alpha_pnl(alpha, price_cache):
    alpha_id = alpha["alpha_id"]
    params = strategy_params(alpha)
    timeframe = alpha.get("timeframe", "1h")
    resolution = report_lib.TIMEFRAME_MAP.get(timeframe)
    annualizer = report_lib.METRIC_ANNUALIZER_MAP.get(timeframe, 365 * 24)
    if resolution is None:
        raise ValueError(f"unsupported timeframe: {timeframe}")

    trade_asset = alpha.get("trade_asset", "BTC")
    price_delay = int(alpha.get("shift_backtest_candle_minute", 0))
    price_key = (trade_asset, timeframe, price_delay)
    if price_key not in price_cache:
        price_cache[price_key] = report_lib.load_price_df(
            trade_asset,
            timeframe,
            price_delay,
        )
    price_df = price_cache[price_key]
    if price_df is None:
        raise ValueError("price data unavailable")

    factor_df = load_factor_frame(alpha)
    signal_df = report_lib.run_backtest(
        factor_df,
        params["window"],
        params["threshold_1"],
        params["threshold_2"],
        params["logic"],
        params["side"],
        params["model"],
    )
    date_range = pd.date_range(
        CORRELATION_START_DATE,
        CORRELATION_END_DATE,
        freq=resolution,
    )
    result_df = signal_df.loc[
        CORRELATION_START_DATE:CORRELATION_END_DATE
    ].copy()
    result_df = result_df.reindex(date_range).ffill()
    result_df = result_df.join(price_df[["close"]], how="inner")
    metrics, result_df = report_lib.calculate_metrics(
        result_df,
        params["window"],
        annualizer,
        float(alpha.get("fees", 0.035)) / 100,
    )

    quick_metrics = alpha_quick_view_metrics(result_df, annualizer, metrics)
    row = {
        "alpha_id": alpha_id,
        **params,
        "timeframe": timeframe,
        "annualizer": annualizer,
        **metrics,
        **quick_metrics,
    }
    return (
        result_df["pnl"].rename(alpha_id),
        result_df["pos"].rename(alpha_id),
        result_df["trades"].rename(alpha_id),
        row,
    )


def alpha_quick_view_metrics(result_df, annualizer, metrics):
    """Return the latest per-alpha values used by the quick-view table."""
    if result_df.empty:
        return {
            "latest_data_date": "",
            "current_month_pnl": np.nan,
            "rolling_sharpe_180": np.nan,
            "rolling_sharpe_365": np.nan,
            "sortino_365": np.nan,
            "current_dd": np.nan,
            "mdd_abs": np.nan,
            "current_dd_duration": np.nan,
            "ldd": np.nan,
        }

    latest_date = pd.Timestamp(result_df.index.max())
    month_mask = (
        result_df.index.year == latest_date.year
    ) & (result_df.index.month == latest_date.month)
    current_month_pnl = result_df.loc[month_mask, "pnl"].sum(min_count=1)

    bars_per_day = max(1, int(round(annualizer / 365)))
    sortino_window = max(2, 365 * bars_per_day)
    pnl_window = result_df["pnl"].dropna().tail(sortino_window)
    downside = np.minimum(pnl_window.to_numpy(dtype=float), 0.0)
    downside_deviation = np.sqrt(np.mean(downside ** 2)) if len(pnl_window) else np.nan
    sortino_365 = (
        pnl_window.mean() / downside_deviation * np.sqrt(annualizer)
        if pd.notna(downside_deviation) and downside_deviation > 0
        else np.nan
    )

    dd = pd.to_numeric(result_df["dd"], errors="coerce").dropna()
    current_dd = abs(float(dd.iloc[-1])) if not dd.empty else np.nan
    current_dd_duration = 0
    for value in reversed(dd.to_numpy()):
        if value < 0:
            current_dd_duration += 1
        else:
            break

    return {
        "latest_data_date": latest_date.strftime("%Y-%m-%d"),
        "current_month_pnl": current_month_pnl,
        "rolling_sharpe_180": result_df["rolling_sharpe_180"].iloc[-1],
        "rolling_sharpe_365": result_df["rolling_sharpe_365"].iloc[-1],
        "sortino_365": sortino_365,
        "current_dd": current_dd,
        "mdd_abs": abs(float(metrics["MDD"])) if pd.notna(metrics.get("MDD")) else np.nan,
        "current_dd_duration": current_dd_duration,
        "ldd": metrics.get("LDD", np.nan),
    }


def correlation_input(pnl_df):
    if not CORRELATION_RESAMPLE:
        return pnl_df
    return pnl_df.resample(CORRELATION_RESAMPLE).sum(min_count=1)


def pair_observations(pnl_df, left, right):
    return int(pnl_df[[left, right]].dropna().shape[0])


def filter_correlated_alphas(pnl_df, component_df):
    corr_input_df = correlation_input(pnl_df)
    corr_df = corr_input_df.corr(min_periods=CORRELATION_MIN_OBSERVATIONS)
    ranking = component_df.sort_values(
        ["SR", "alpha_id"],
        ascending=[False, True],
        na_position="last",
    )

    kept = []
    decision_rows = []
    for _, row in ranking.iterrows():
        alpha_id = row["alpha_id"]
        if not kept:
            kept.append(alpha_id)
            decision_rows.append(
                {
                    "alpha_id": alpha_id,
                    "decision": "KEEP",
                    "blocked_by": "",
                    "correlation": np.nan,
                    "observations": np.nan,
                    "reason": "highest-ranked candidate",
                }
            )
            continue

        comparisons = []
        for kept_id in kept:
            correlation = corr_df.at[alpha_id, kept_id]
            comparison_value = (
                abs(correlation) if CORRELATION_USE_ABSOLUTE else correlation
            )
            comparisons.append((kept_id, correlation, comparison_value))

        valid = [item for item in comparisons if pd.notna(item[2])]
        if not valid:
            decision_rows.append(
                {
                    "alpha_id": alpha_id,
                    "decision": "REMOVE",
                    "blocked_by": "",
                    "correlation": np.nan,
                    "observations": 0,
                    "reason": "insufficient overlapping PnL observations",
                }
            )
            continue

        blocker, correlation, comparison_value = max(valid, key=lambda item: item[2])
        observations = pair_observations(corr_input_df, alpha_id, blocker)
        if comparison_value >= CORRELATION_THRESHOLD:
            decision_rows.append(
                {
                    "alpha_id": alpha_id,
                    "decision": "REMOVE",
                    "blocked_by": blocker,
                    "correlation": correlation,
                    "observations": observations,
                    "reason": (
                        f"correlation >= {CORRELATION_THRESHOLD} with stronger alpha"
                    ),
                }
            )
        else:
            kept.append(alpha_id)
            decision_rows.append(
                {
                    "alpha_id": alpha_id,
                    "decision": "KEEP",
                    "blocked_by": blocker,
                    "correlation": correlation,
                    "observations": observations,
                    "reason": "below threshold versus every retained alpha",
                }
            )

    decisions_df = pd.DataFrame(decision_rows)
    decisions_df = decisions_df.merge(
        component_df[
            [
                "alpha_id",
                "SR",
                "CR",
                "MDD",
                "AR",
                "TR",
                "window",
                "threshold_1",
                "threshold_2",
                "model",
                "logic",
                "side",
            ]
        ],
        on="alpha_id",
        how="left",
    )
    return kept, corr_df, decisions_df


def build_selected_alpha_definitions(selected_ids, alpha_map, component_df):
    """Materialize selected alpha records with the parameters used in this run."""
    component_lookup = component_df.set_index("alpha_id")
    exported_at_utc = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    selected_records = []

    for selected_order, alpha_id in enumerate(selected_ids, start=1):
        if alpha_id not in alpha_map:
            raise KeyError(f"Selected alpha definition not found: {alpha_id}")
        if alpha_id not in component_lookup.index:
            raise KeyError(f"Selected alpha parameters not found: {alpha_id}")

        row = component_lookup.loc[alpha_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        model = str(row["model"]).strip()
        logic = str(row["logic"]).strip()
        side = str(row["side"]).strip().lower()
        window = int(float(row["window"]))
        threshold_1 = float(row["threshold_1"])
        threshold_2 = float(row["threshold_2"])
        entry_exit_logic = logic if side == "both" else f"{logic}_{side}"

        record = deepcopy(alpha_map[alpha_id])
        record.update(
            {
                "model": model,
                "entry_exit_logic": entry_exit_logic,
                "side": side,
                "window": window,
                "rolling_window_1": window,
                "rolling_window_2": window,
                "long_entry_threshold": threshold_1,
                "long_exit_threshold": threshold_2,
                "short_entry_threshold": threshold_2,
                "short_exit_threshold": threshold_1,
            }
        )
        record["correlation_selection"] = {
            "selected": True,
            "selected_order": selected_order,
            "parameter_source": str(row.get("param_source", CORRELATION_PARAM_SOURCE)),
            "correlation_threshold": float(CORRELATION_THRESHOLD),
            "use_absolute_correlation": bool(CORRELATION_USE_ABSOLUTE),
            "minimum_observations": int(CORRELATION_MIN_OBSERVATIONS),
            "resample": CORRELATION_RESAMPLE,
            "start_date": str(CORRELATION_START_DATE),
            "end_date": str(CORRELATION_END_DATE),
            "exported_at_utc": exported_at_utc,
        }
        selected_records.append(record)

    return selected_records


def save_selected_alpha_json(selected_ids, alpha_map, component_df, output_path=None):
    selected_records = build_selected_alpha_definitions(
        selected_ids,
        alpha_map,
        component_df,
    )
    output_path = Path(output_path) if output_path is not None else OUTPUT_DIR / "alpha.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(selected_records, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def pnl_metrics(pnl, annualizer, trades=None):
    pnl = pnl.dropna()
    metric_keys = ["SR", "CR", "MDD", "AR", "TR", "VAR", "num_trades"]
    if pnl.empty:
        return {key: np.nan for key in metric_keys}
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()
    std = pnl.std()
    sr = pnl.mean() / std * np.sqrt(annualizer) if std != 0 else np.nan
    mdd = drawdown.min()
    ar = pnl.mean() * annualizer
    num_trades = np.nan
    if trades is not None:
        num_trades = round(float(trades.reindex(pnl.index).fillna(0).sum()), 4)
    return {
        "SR": round(sr, 4) if pd.notna(sr) else np.nan,
        "CR": round(ar / abs(mdd), 4) if pd.notna(mdd) and mdd != 0 else np.nan,
        "MDD": round(mdd, 4),
        "AR": round(ar, 4),
        "TR": round(cumulative.iloc[-1], 4),
        "VAR": round(pnl.quantile(0.05) * 100, 4),
        "num_trades": num_trades,
    }


def build_portfolio(pnl_df, pos_df, trades_df, selected_ids, annualizer):
    selected_pnl = pnl_df[selected_ids]
    active_count = selected_pnl.notna().sum(axis=1)
    active_count = active_count.replace(0, np.nan)
    weighted = selected_pnl.div(active_count, axis=0)
    weighted_pos = pos_df[selected_ids].div(active_count, axis=0)
    weighted_trades = trades_df[selected_ids].div(active_count, axis=0)
    portfolio_pnl = weighted.sum(axis=1, min_count=1).where(active_count.notna())

    portfolio_df = pd.DataFrame({"pnl": portfolio_pnl})
    portfolio_df["pos"] = weighted_pos.sum(axis=1, min_count=1).where(active_count.notna())
    portfolio_df["trades"] = weighted_trades.sum(axis=1, min_count=1).where(active_count.notna())
    portfolio_df["cumu"] = portfolio_df["pnl"].fillna(0).cumsum()
    portfolio_df["dd"] = portfolio_df["cumu"] - portfolio_df["cumu"].cummax()
    bars_per_day = max(1, int(round(annualizer / 365)))
    for days in (180, 365):
        window = max(2, days * bars_per_day)
        portfolio_df[f"rolling_sharpe_{days}"] = (
            portfolio_df["pnl"].rolling(window).mean()
            / portfolio_df["pnl"].rolling(window).std()
            * np.sqrt(annualizer)
        )

    equity_df = weighted.fillna(0).cumsum()
    contribution = weighted.sum(axis=0, min_count=1)
    metrics = pnl_metrics(portfolio_df["pnl"], annualizer, portfolio_df["trades"])
    return portfolio_df, equity_df, contribution, metrics


def yearly_metrics(portfolio_df, annualizer):
    rows = []
    for year in sorted(portfolio_df.index.year.unique()):
        year_df = portfolio_df.loc[portfolio_df.index.year == year]
        metrics = pnl_metrics(
            year_df["pnl"],
            annualizer,
            year_df["trades"],
        )
        rows.append({"Year": int(year), **metrics})
    return pd.DataFrame(rows)


def correlation_heatmap(corr_df, title, include_plotlyjs=False):
    if corr_df.empty:
        return ""
    text = np.where(
        pd.isna(corr_df.values),
        "",
        np.round(corr_df.values, 3).astype(str),
    )
    fig = go.Figure(
        go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            reversescale=True,
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>correlation=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=max(540, 38 * len(corr_df) + 180),
        template="plotly_white",
        xaxis_tickangle=45,
    )
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def portfolio_chart(portfolio_df, equity_df, contribution, metrics):
    fig = make_subplots(
        rows=5,
        cols=1,
        vertical_spacing=0.08,
        subplot_titles=(
            "Equal-Weight Portfolio Cumulative PnL",
            "Portfolio Drawdown",
            "Selected Alpha Equity Curves",
            "Total PnL Contribution",
            "Rolling Sharpe",
        ),
        row_heights=[0.30, 0.15, 0.22, 0.15, 0.18],
    )
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["cumu"],
            name="Portfolio",
            line=dict(color="#1565c0"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["dd"],
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#c62828"),
        ),
        row=2,
        col=1,
    )
    for alpha_id in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df[alpha_id],
                name=alpha_id,
                opacity=0.65,
            ),
            row=3,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=contribution.index,
            y=contribution.values,
            name="Contribution",
            marker_color="#2e7d32",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["rolling_sharpe_180"],
            name="Rolling 180d SR",
            line=dict(color="#6a1b9a"),
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["rolling_sharpe_365"],
            name="Rolling 365d SR",
            line=dict(color="#00897b"),
        ),
        row=5,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555", row=5, col=1)
    fig.update_layout(
        height=1550,
        template="plotly_white",
        title=(
            f"Correlation-Filtered Portfolio | SR={metrics['SR']} "
            f"MDD={metrics['MDD']} TR={metrics['TR']}"
        ),
        title_x=0.5,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def position_trade_chart(portfolio_df):
    if portfolio_df.empty or "pos" not in portfolio_df.columns:
        return ""

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Portfolio Position", "Portfolio Trades"),
        row_heights=[0.62, 0.38],
    )
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["pos"],
            name="Position",
            line=dict(color="#1565c0"),
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=portfolio_df.index,
            y=portfolio_df["trades"],
            name="Trades",
            marker_color="#ef6c00",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Trades", row=2, col=1)
    fig.update_layout(
        height=650,
        template="plotly_white",
        title="Portfolio Position and Trade Time Series",
        title_x=0.5,
        bargap=0,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def portfolio_monthly_pnl_heatmap(portfolio_df):
    """Render whole-portfolio monthly PnL with red losses and green gains."""
    if portfolio_df.empty or "pnl" not in portfolio_df.columns:
        return ""

    monthly_pnl = portfolio_df["pnl"].resample("ME").sum(min_count=1)
    if monthly_pnl.empty:
        return ""

    values = monthly_pnl.to_numpy(dtype=float).reshape(1, -1)
    finite_values = np.abs(values[np.isfinite(values)])
    color_limit = float(finite_values.max()) if finite_values.size else 1.0
    if color_limit == 0:
        color_limit = 1.0

    text = np.empty(values.shape, dtype=object)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text[row_index, column_index] = (
                "" if not np.isfinite(value) else f"{value:.4f}"
            )

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=monthly_pnl.index.strftime("%Y-%m"),
            y=["Portfolio"],
            zmin=-color_limit,
            zmax=color_limit,
            zmid=0,
            colorscale=[
                [0.00, "#8b0000"],
                [0.25, "#ef9a9a"],
                [0.50, "#ffffff"],
                [0.75, "#a5d6a7"],
                [1.00, "#006400"],
            ],
            colorbar=dict(title="Monthly PnL"),
            text=text,
            texttemplate="%{text}",
            hovertemplate=(
                "Portfolio<br>Month=%{x}<br>Monthly PnL=%{z:.4f}<extra></extra>"
            ),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Whole-Portfolio Monthly PnL Heatmap",
        height=360,
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="",
        xaxis_tickangle=45,
        margin=dict(l=130, r=40, t=80, b=100),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def render_table(df, classes="metrics-table"):
    if df.empty:
        return "<p>No rows.</p>"
    return df.to_html(
        index=False,
        classes=classes,
        border=0,
        na_rep="",
        float_format=lambda value: f"{value:.4f}",
    )


def _quick_value(value, digits=4):
    if pd.isna(value):
        return ""
    if np.isposinf(value):
        return "∞"
    if np.isneginf(value):
        return "-∞"
    return f"{float(value):.{digits}f}"


def _quick_sort_value(value):
    if pd.isna(value):
        return ""
    return str(value)


def _allocation_rows_html(items):
    rows = []
    for label, percentage in items:
        safe_label = html.escape(str(label))
        rows.append(
            '<div class="allocation-row">'
            f'<div class="allocation-label">{safe_label}</div>'
            '<div class="allocation-track">'
            f'<div class="allocation-fill" style="width:{percentage:.4f}%"></div>'
            '</div>'
            f'<div class="allocation-value">{percentage:.2f}%</div>'
            '</div>'
        )
    return "".join(rows) if rows else '<div class="quick-view-note">No allocation data.</div>'


def _entry_logic_label(logic, side):
    logic = str(logic or "").strip().lower()
    side = str(side or "").strip().lower()
    if logic in {"trend", "trend_reverse"}:
        return "Trend / Momentum"
    if logic in {"mr", "mr_reverse", "mean_reversion"}:
        return "Mean Reversion"
    if logic == "fast":
        return "Fast Long" if side == "long" else "Fast"
    if logic == "fast_reverse":
        return "Fast Reverse"
    return logic.replace("_", " ").title() or "Unknown"


def _primary_factor_label(alpha):
    datasource_raw = alpha.get("datasource_structure", "{}")
    try:
        datasources = json.loads(datasource_raw.replace("'", '"'))
    except Exception:
        try:
            datasources = ast.literal_eval(datasource_raw)
        except Exception:
            datasources = {}
    return next(iter(datasources), "Unknown")


def build_portfolio_allocation_html(selected_ids, alpha_map, component_df):
    """Summarize equal-weight selected-alpha exposure by asset, logic, and factor."""
    if not selected_ids:
        return '<div class="quick-view-note">No selected alphas.</div>'

    component_lookup = component_df.set_index("alpha_id")
    asset_counts = {}
    logic_counts = {}
    factor_counts = {}
    for alpha_id in selected_ids:
        alpha = alpha_map.get(alpha_id, {})
        asset = str(alpha.get("trade_asset") or "Unknown")
        asset_counts[asset] = asset_counts.get(asset, 0) + 1

        if alpha_id in component_lookup.index:
            component = component_lookup.loc[alpha_id]
            if isinstance(component, pd.DataFrame):
                component = component.iloc[0]
            logic_label = _entry_logic_label(
                component.get("logic"),
                component.get("side"),
            )
        else:
            logic, side = parse_logic_and_side(alpha.get("entry_exit_logic"))
            logic_label = _entry_logic_label(logic, side)
        logic_counts[logic_label] = logic_counts.get(logic_label, 0) + 1

        factor = _primary_factor_label(alpha)
        factor_counts[factor] = factor_counts.get(factor, 0) + 1

    total = float(len(selected_ids))

    def percentages(counts):
        return sorted(
            ((label, count / total * 100.0) for label, count in counts.items()),
            key=lambda item: (-item[1], item[0]),
        )

    asset_html = _allocation_rows_html(percentages(asset_counts))
    logic_html = _allocation_rows_html(percentages(logic_counts))
    factor_html = _allocation_rows_html(percentages(factor_counts))
    return f"""
    <div class="allocation-top-grid">
      <section>
        <h3>Allocation by Asset</h3>
        {asset_html}
      </section>
      <section>
        <h3>Entry Logic</h3>
        {logic_html}
      </section>
    </div>
    <section class="factor-section">
      <h3>Factor Mix %</h3>
      <div class="factor-grid">{factor_html}</div>
    </section>
    """


def build_alpha_quick_view_table(component_df):
    """Build the sortable, color-coded per-alpha summary table."""
    if component_df.empty:
        return "<p>No alpha metrics available.</p>", "Current-month PnL"

    latest_dates = pd.to_datetime(
        component_df["latest_data_date"], errors="coerce"
    ).dropna()
    if latest_dates.empty:
        month_label = "Current-month PnL"
        latest_label = "latest data unavailable"
    else:
        latest_date = latest_dates.max()
        month_label = f"{latest_date.strftime('%b')} PnL"
        latest_label = f"latest data: {latest_date.strftime('%Y-%m-%d')}"

    headers = [
        "Alpha",
        month_label,
        "180d SR",
        "365d SR",
        "365d Sortino Ratio",
        "Current DD ▲ / MDD",
        "Current DD Duration / LDD",
    ]
    rows = []
    for _, row in component_df.iterrows():
        sr_180 = row.get("rolling_sharpe_180", np.nan)
        sr_365 = row.get("rolling_sharpe_365", np.nan)
        sortino_365 = row.get("sortino_365", np.nan)
        current_dd = row.get("current_dd", np.nan)
        mdd_abs = row.get("mdd_abs", np.nan)
        current_duration = row.get("current_dd_duration", np.nan)
        ldd = row.get("ldd", np.nan)

        def metric_cell(value, display=None, warning=False):
            class_name = " class=\"quick-warning\"" if warning else ""
            text = _quick_value(value) if display is None else display
            return (
                f'<td data-sort="{html.escape(_quick_sort_value(value))}"'
                f"{class_name}>{html.escape(text)}</td>"
            )

        dd_warning = (
            pd.notna(current_dd)
            and pd.notna(mdd_abs)
            and current_dd > mdd_abs
        )
        duration_warning = (
            pd.notna(current_duration)
            and pd.notna(ldd)
            and current_duration > ldd
        )
        dd_display = f"{_quick_value(current_dd)} / {_quick_value(mdd_abs)}"
        duration_display = (
            f"{_quick_value(current_duration, 0)} / {_quick_value(ldd, 0)}"
        )
        alpha_id = str(row.get("alpha_id", ""))
        rows.append(
            "<tr>"
            f'<td data-sort="{html.escape(alpha_id)}">{html.escape(alpha_id)}</td>'
            f'<td data-sort="{html.escape(_quick_sort_value(row.get("current_month_pnl", np.nan)))}">'
            f'{html.escape(_quick_value(row.get("current_month_pnl", np.nan)))}</td>'
            + metric_cell(sr_180, warning=pd.notna(sr_180) and sr_180 < 0)
            + metric_cell(sr_365, warning=pd.notna(sr_365) and sr_365 < 0)
            + metric_cell(sortino_365, warning=pd.notna(sortino_365) and sortino_365 < 0)
            + metric_cell(current_dd, display=dd_display, warning=dd_warning)
            + metric_cell(current_duration, display=duration_display, warning=duration_warning)
            + "</tr>"
        )

    header_html = "".join(
        f'<th data-sort-index="{index}">{html.escape(header)} <span class="sort-indicator"></span></th>'
        for index, header in enumerate(headers)
    )
    table_html = f"""
    <div class="quick-view-note">Click any column title to sort ({html.escape(latest_label)}).</div>
    <div class="table-scroll">
      <table class="metrics-table sortable-table" id="alpha-quick-view">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """
    return table_html, month_label


def build_html(
    all_corr_df,
    selected_corr_df,
    decisions_df,
    component_df,
    skipped_df,
    allocation_html,
    portfolio_df,
    equity_df,
    contribution,
    portfolio_metrics,
    yearly_df,
    requested_count,
):
    metric_keys = ["SR", "CR", "MDD", "AR", "TR", "VAR", "num_trades"]
    metrics_df = pd.DataFrame(
        {
            "Metric": metric_keys,
            "Correlation-Filtered Portfolio": [
                portfolio_metrics.get(key) for key in metric_keys
            ],
        }
    )
    selected_count = int((decisions_df["decision"] == "KEEP").sum())
    removed_count = int((decisions_df["decision"] == "REMOVE").sum())
    threshold_mode = "absolute correlation" if CORRELATION_USE_ABSOLUTE else "positive correlation"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_heatmap = correlation_heatmap(
        all_corr_df,
        "All Candidate PnL Correlations",
        include_plotlyjs=False,
    )
    selected_heatmap = correlation_heatmap(
        selected_corr_df,
        "Selected Portfolio PnL Correlations",
    )
    portfolio_html = portfolio_chart(
        portfolio_df,
        equity_df,
        contribution,
        portfolio_metrics,
    )
    position_html = position_trade_chart(portfolio_df)
    monthly_pnl_html = portfolio_monthly_pnl_heatmap(portfolio_df)

    decision_display = decisions_df[
        [
            "alpha_id",
            "decision",
            "blocked_by",
            "correlation",
            "observations",
            "SR",
            "window",
            "threshold_1",
            "threshold_2",
            "model",
            "logic",
            "side",
            "reason",
        ]
    ]
    component_display = component_df[
        [
            "alpha_id",
            "SR",
            "CR",
            "MDD",
            "AR",
            "TR",
            "window",
            "threshold_1",
            "threshold_2",
            "model",
            "logic",
            "side",
            "param_source",
        ]
    ].sort_values("SR", ascending=False)
    quick_view_html, quick_view_month_label = build_alpha_quick_view_table(component_df)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Correlation Portfolio Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f4f6f8; color: #202124; }}
    .dashboard {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(420px, 0.8fr); gap: 18px; height: 100vh; padding: 18px; box-sizing: border-box; }}
    .plots, .tables {{ overflow-y: auto; }}
    .header, .panel {{ background: white; border: 1px solid #dfe5ec; padding: 16px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 10px; font-size: 25px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; color: #333; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{ border: 1px solid #b8c5d3; padding: 5px 8px; background: #f8fafc; font-size: 12px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 7px; text-align: center; word-break: break-word; }}
    th {{ background: #4CAF50; color: white; position: sticky; top: 0; }}
    .wide-table {{ min-width: 1150px; }}
    .table-scroll {{ overflow-x: auto; }}
    .note {{ font-size: 13px; line-height: 1.5; margin-bottom: 12px; }}
    .quick-view-note {{ font-size: 12px; color: #5f6368; margin-bottom: 8px; }}
    .sortable-table th {{ cursor: pointer; user-select: none; white-space: nowrap; }}
    .sortable-table th:hover {{ background: #388e3c; }}
    .sort-indicator {{ font-size: 10px; margin-left: 3px; }}
    .quick-warning {{ color: #c62828; font-weight: 700; }}
    .allocation-top-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px; }}
    .allocation-top-grid h3, .factor-section h3 {{ margin: 0 0 12px; font-size: 14px; color: #1f4f7a; }}
    .factor-section {{ border-top: 1px solid #dfe5ec; margin-top: 18px; padding-top: 16px; }}
    .factor-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: 28px; }}
    .allocation-row {{ display: grid; grid-template-columns: minmax(145px, 1.1fr) minmax(100px, 2fr) 64px; gap: 8px; align-items: center; min-height: 28px; font-size: 12px; }}
    .allocation-label {{ overflow-wrap: anywhere; text-align: left; }}
    .allocation-track {{ height: 8px; background: #e5edf3; border-radius: 5px; overflow: hidden; }}
    .allocation-fill {{ height: 100%; background: #2e7d32; border-radius: 5px; }}
    .allocation-value {{ text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }}
    @media (max-width: 1000px) {{
      .dashboard {{ display: block; height: auto; }}
      .plots, .tables {{ overflow: visible; }}
      .allocation-top-grid, .factor-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<div class="dashboard">
  <div class="plots">
    <div class="header">
      <h1>Correlation Portfolio Report</h1>
      <div class="note">
        Candidates are ranked by full-period SR. A lower-ranked alpha is removed when its
        {html.escape(threshold_mode)} is at least {CORRELATION_THRESHOLD:.2f} with a retained alpha.
        Portfolio weights are equal across active selected alphas.
      </div>
      <div class="meta">
        <div class="pill">requested: {requested_count}</div>
        <div class="pill">tested: {len(component_df)}</div>
        <div class="pill">selected: {selected_count}</div>
        <div class="pill">removed: {removed_count}</div>
        <div class="pill">skipped: {len(skipped_df)}</div>
        <div class="pill">period: {CORRELATION_START_DATE} to {CORRELATION_END_DATE}</div>
        <div class="pill">correlation resample: {html.escape(str(CORRELATION_RESAMPLE or 'native'))}</div>
        <div class="pill">generated: {generated_at}</div>
      </div>
    </div>
    <div class="panel">
      <h2>Portfolio Allocation Mix</h2>
      {allocation_html}
    </div>
    <div class="panel">{portfolio_html}</div>
    <div class="panel">{position_html}</div>
    <div class="panel">{monthly_pnl_html}</div>
    <div class="panel">{all_heatmap}</div>
    <div class="panel">{selected_heatmap}</div>
  </div>
  <div class="tables">
    <div class="panel">
      <h2>Portfolio Metrics</h2>
      {render_table(metrics_df)}
    </div>
    <div class="panel">
      <h2>Yearly Portfolio Metrics</h2>
      {render_table(yearly_df)}
    </div>
    <div class="panel">
      <h2>Alpha Quick View — {html.escape(quick_view_month_label)}</h2>
      {quick_view_html}
    </div>
    <div class="panel">
      <h2>Correlation Keep / Remove Decisions</h2>
      <div class="table-scroll">{render_table(decision_display, "metrics-table wide-table")}</div>
    </div>
    <div class="panel">
      <h2>Candidate Alpha Metrics and Parameters</h2>
      <div class="table-scroll">{render_table(component_display, "metrics-table wide-table")}</div>
    </div>
    <div class="panel">
      <h2>Skipped Alphas</h2>
      {render_table(skipped_df)}
    </div>
  </div>
</div>
<script>
  document.querySelectorAll("#alpha-quick-view th").forEach(function (header) {{
    header.addEventListener("click", function () {{
      const table = header.closest("table");
      const index = Number(header.dataset.sortIndex);
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);
      const ascending = header.dataset.direction !== "asc";

      table.querySelectorAll("th").forEach(function (other) {{
        other.dataset.direction = "";
        const indicator = other.querySelector(".sort-indicator");
        if (indicator) indicator.textContent = "";
      }});
      header.dataset.direction = ascending ? "asc" : "desc";
      const indicator = header.querySelector(".sort-indicator");
      if (indicator) indicator.textContent = ascending ? "▲" : "▼";

      rows.sort(function (left, right) {{
        const leftValue = left.cells[index].dataset.sort || "";
        const rightValue = right.cells[index].dataset.sort || "";
        const leftNumber = Number(leftValue);
        const rightNumber = Number(rightValue);
        let comparison;
        if (leftValue !== "" && rightValue !== "" && !Number.isNaN(leftNumber) && !Number.isNaN(rightNumber)) {{
          comparison = leftNumber - rightNumber;
        }} else {{
          comparison = leftValue.localeCompare(rightValue);
        }}
        return ascending ? comparison : -comparison;
      }});
      rows.forEach(function (row) {{ tbody.appendChild(row); }});
    }});
  }});
</script>
</body>
</html>"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_FILE, "r", encoding="utf-8") as handle:
        all_alphas = json.load(handle)
    alpha_map = {alpha["alpha_id"]: alpha for alpha in all_alphas}
    requested_ids = requested_alpha_ids(all_alphas)

    pnl_series = []
    pos_series = []
    trades_series = []
    component_rows = []
    skipped_rows = []
    price_cache = {}
    for alpha_id in requested_ids:
        alpha = alpha_map.get(alpha_id)
        if alpha is None:
            skipped_rows.append(
                {"alpha_id": alpha_id, "reason": "alpha definition not found"}
            )
            continue
        print(f"Processing {alpha_id} ...")
        try:
            pnl, pos, trades, component = build_alpha_pnl(alpha, price_cache)
            pnl_series.append(pnl)
            pos_series.append(pos)
            trades_series.append(trades)
            component_rows.append(component)
            print(
                f"  SR={component['SR']} window={component['window']}"
            )
        except Exception as exc:
            skipped_rows.append({"alpha_id": alpha_id, "reason": str(exc)})
            print(f"  SKIPPED: {exc}")

    if not pnl_series:
        raise ValueError("No alpha PnL series were generated")

    pnl_df = pd.concat(pnl_series, axis=1).sort_index()
    pos_df = pd.concat(pos_series, axis=1).sort_index()
    trades_df = pd.concat(trades_series, axis=1).sort_index()
    component_df = pd.DataFrame(component_rows)
    skipped_df = pd.DataFrame(skipped_rows, columns=["alpha_id", "reason"])
    selected_ids, corr_df, decisions_df = filter_correlated_alphas(
        pnl_df,
        component_df,
    )
    if not selected_ids:
        raise ValueError("Correlation filter did not retain any alpha")

    annualizer = int(component_df["annualizer"].mode().iloc[0])
    portfolio_df, equity_df, contribution, portfolio_metrics = build_portfolio(
        pnl_df,
        pos_df,
        trades_df,
        selected_ids,
        annualizer,
    )
    yearly_df = yearly_metrics(portfolio_df, annualizer)
    selected_corr_df = corr_df.loc[selected_ids, selected_ids]

    pnl_df.to_csv(OUTPUT_DIR / "correlation_alpha_pnl.csv")
    pos_df.to_csv(OUTPUT_DIR / "correlation_alpha_pos.csv")
    corr_df.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    selected_corr_df.to_csv(OUTPUT_DIR / "correlation_selected_matrix.csv")
    decisions_df.to_csv(OUTPUT_DIR / "correlation_selection.csv", index=False)
    component_df.to_csv(OUTPUT_DIR / "correlation_components.csv", index=False)
    skipped_df.to_csv(OUTPUT_DIR / "correlation_skipped.csv", index=False)
    portfolio_df.to_csv(OUTPUT_DIR / "correlation_portfolio_timeseries.csv")
    selected_alpha_path = save_selected_alpha_json(
        selected_ids,
        alpha_map,
        component_df,
    )
    allocation_html = build_portfolio_allocation_html(
        selected_ids,
        alpha_map,
        component_df,
    )

    report = build_html(
        corr_df,
        selected_corr_df,
        decisions_df,
        component_df,
        skipped_df,
        allocation_html,
        portfolio_df,
        equity_df,
        contribution,
        portfolio_metrics,
        yearly_df,
        len(requested_ids),
    )
    output_path = OUTPUT_DIR / CORRELATION_OUTPUT_HTML
    output_path.write_text(report, encoding="utf-8")

    print(f"Selected alphas ({len(selected_ids)}): {', '.join(selected_ids)}")
    print(f"Portfolio SR: {portfolio_metrics['SR']}")
    print(f"Portfolio MDD: {portfolio_metrics['MDD']}")
    print(f"Selected alpha JSON saved: {selected_alpha_path}")
    print(f"HTML report saved: {output_path}")


if __name__ == "__main__":
    main()

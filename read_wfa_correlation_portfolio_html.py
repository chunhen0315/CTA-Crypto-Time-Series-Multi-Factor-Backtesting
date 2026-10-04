import argparse
import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
import read_wfa_stich_alpha_portfolio_html as wf_lib

from config import (
WF_CORR_THRESHOLD as DEFAULT_THRESHOLD,
WF_CORR_MIN_SR_GATE as DEFAULT_MIN_ALPHA_SR
)
DEFAULT_MIN_OBSERVATIONS = 100
DEFAULT_OUTPUT_NAME = "wf_corr_portfolio.html"
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a correlation-filtered equal-weight portfolio report from WFA round-best summaries."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=wf_lib.DEFAULT_INPUT_DIR,
        help=f"Folder to scan for walk-forward summary CSVs. Default: {wf_lib.DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for HTML/CSV outputs. Default: same as --input-dir.",
    )
    parser.add_argument(
        "--date-column",
        choices=wf_lib.DATE_COLUMNS,
        default="trade_start",
        help="Round date used for latest-window filtering. Default: trade_start.",
    )
    parser.add_argument(
        "--window",
        choices=["latest_round", "all", "calendar_year", "rolling_12m"],
        default="all",
        help="latest_round keeps only the latest row per alpha; all keeps every CSV row.",
    )
    parser.add_argument(
        "--target-alpha-id",
        nargs="*",
        default=None,
        help="Optional alpha ids to include. Default: all discovered CSVs.",
    )
    parser.add_argument(
        "--exclude-single-day-trade",
        action="store_true",
        help="Drop rows where trade_start and trade_end are the same date.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Maximum allowed correlation against retained alphas. Default: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=DEFAULT_MIN_OBSERVATIONS,
        help=f"Minimum overlapping PnL observations for correlation. Default: {DEFAULT_MIN_OBSERVATIONS}",
    )
    parser.add_argument(
        "--resample",
        default=None,
        help="Optional pandas resample rule before correlation, e.g. 1D. Default: native frequency.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Use raw positive correlation only. Default uses absolute correlation.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output HTML filename. Default: {DEFAULT_OUTPUT_NAME}",
    )
    return parser.parse_args()


def bool_text(value):
    return "absolute correlation" if value else "positive correlation"


def correlation_input(pnl_df, resample_rule):
    if not resample_rule:
        return pnl_df
    return pnl_df.resample(resample_rule).sum(min_count=1)


def pair_observations(pnl_df, left, right):
    return int(pnl_df[[left, right]].dropna().shape[0])


def compute_enabled_round_rate_by_alpha(df):
    rows = []
    if df.empty:
        return pd.DataFrame(
            columns=[
                "alpha_id",
                "ENABLED_ROUND_COUNT",
                "TOTAL_ROUND_COUNT",
                "ENABLED_ROUND_RATE",
                "ENABLED_ROUND_RATE_FILTER",
            ]
        )

    threshold = float(config.WF_pass_rate_threshold)
    enabled_mask = pd.Series(
        [wf_lib.row_trade_gate(row)[0] for _, row in df.iterrows()],
        index=df.index,
    )

    for alpha_id, alpha_df in df.assign(_trade_enabled=enabled_mask).groupby("alpha_id", sort=True):
        round_count = int(len(alpha_df))
        enabled_count = int(alpha_df["_trade_enabled"].sum())
        enabled_rate = enabled_count / round_count if round_count else np.nan
        rows.append(
            {
                "alpha_id": str(alpha_id),
                "ENABLED_ROUND_COUNT": enabled_count,
                "TOTAL_ROUND_COUNT": round_count,
                "ENABLED_ROUND_RATE": round(enabled_rate, 4) if pd.notna(enabled_rate) else np.nan,
                "ENABLED_ROUND_RATE_FILTER": bool(pd.notna(enabled_rate) and enabled_rate > threshold),
            }
        )

    return pd.DataFrame(rows)


def filter_enabled_round_rate_rows(df, pass_rate_df):
    df = df.copy()
    pass_rate_cols = [
        "alpha_id",
        "ENABLED_ROUND_COUNT",
        "TOTAL_ROUND_COUNT",
        "ENABLED_ROUND_RATE",
        "ENABLED_ROUND_RATE_FILTER",
    ]
    df = df.drop(columns=[col for col in pass_rate_cols[1:] if col in df.columns])
    df = df.merge(pass_rate_df[pass_rate_cols], on="alpha_id", how="left")
    df["ENABLED_ROUND_RATE_FILTER"] = df["ENABLED_ROUND_RATE_FILTER"].map(wf_lib.to_bool)
    mask = df["ENABLED_ROUND_RATE_FILTER"]
    return df[mask].copy(), int(mask.sum()), int((~mask).sum())


def build_alpha_stitched_backtests(recent_df):
    alpha_map = wf_lib.load_alpha_map()
    base_cache = {}
    price_cache = {}
    pnl_series = []
    component_rows = []
    errors = []
    annualizers = []

    for _, row in recent_df.sort_values(["alpha_id", "round_date"]).iterrows():
        alpha_id = str(row.get("alpha_id"))
        if alpha_id not in alpha_map:
            errors.append({"alpha_id": alpha_id, "round": row.get("round"), "error": "alpha_id not found in JSON_FILE"})
            continue

        cid = wf_lib.component_id(row)
        trade_enabled, skip_reason = wf_lib.row_trade_gate(row)
        try:
            if alpha_id not in base_cache:
                base_cache[alpha_id] = wf_lib.prepare_alpha_base(alpha_map[alpha_id], price_cache)
            base = base_cache[alpha_id]
            annualizers.append(base["annualizer"])
            if trade_enabled:
                metrics, evaluated_df, eval_start, eval_end = wf_lib.run_component_backtest(row, base)
                pnl_series.append(evaluated_df["pnl"].rename(cid))
            else:
                skipped_pnl, eval_start, eval_end = wf_lib.blank_trade_series(row, base["resolution"], cid)
                pnl_series.append(skipped_pnl)
                try:
                    metrics, _, _, _ = wf_lib.run_component_backtest(row, base)
                except Exception as metric_exc:
                    metrics = {}
                    errors.append(
                        {
                            "alpha_id": alpha_id,
                            "round": row.get("round"),
                            "error": f"disabled round metrics failed: {metric_exc}",
                        }
                    )
        except Exception as exc:
            errors.append({"alpha_id": alpha_id, "round": row.get("round"), "error": str(exc)})
            continue

        component_row = {
            "component_id": cid,
            "alpha_id": alpha_id,
            "round": row.get("round"),
            "eval_start": eval_start,
            "eval_end": eval_end,
            "timeframe": base["timeframe"],
            "model": row.get("model"),
            "logic": row.get("logic"),
            "side": row.get("side"),
            "window": row.get("window"),
            "threshold_1": row.get("threshold_1"),
            "threshold_2": row.get("threshold_2"),
            "trade_enabled": trade_enabled,
            "skip_reason": "" if trade_enabled else skip_reason,
            "SR": metrics.get("SR"),
            "CR": metrics.get("CR"),
            "MDD": metrics.get("MDD"),
            "AR": metrics.get("AR"),
            "TR": metrics.get("TR"),
            "TPI": metrics.get("TPI"),
            "num_trades": metrics.get("num_trades"),
        }
        for col in wf_lib.WFA_COMPONENT_COLUMNS:
            if col in row.index:
                component_row[col] = row.get(col)
        for col in [
            "ENABLED_ROUND_COUNT",
            "TOTAL_ROUND_COUNT",
            "ENABLED_ROUND_RATE",
            "ENABLED_ROUND_RATE_FILTER",
            "WFA_PASS_COUNT",
            "WFA_ROUND_COUNT",
            "WFA_PASS_RATE",
            "WFA_PASS_RATE_FILTER",
        ]:
            if col in row.index:
                component_row[col] = row.get(col)
        component_rows.append(component_row)

    component_df = pd.DataFrame(component_rows)
    error_df = pd.DataFrame(errors)
    annualizer = int(pd.Series(annualizers).mode().iloc[0]) if annualizers else 365 * 24
    if not pnl_series:
        return pd.DataFrame(), pd.DataFrame(), component_df, error_df, annualizer

    component_pnl_df = pd.concat(pnl_series, axis=1, sort=True).sort_index()
    alpha_pnl_parts = []
    component_alpha = component_df.set_index("component_id")["alpha_id"].to_dict()
    for alpha_id in sorted(component_df["alpha_id"].dropna().astype(str).unique()):
        alpha_cols = [
            col for col in component_pnl_df.columns
            if str(component_alpha.get(col)) == alpha_id
        ]
        if not alpha_cols:
            continue
        alpha_component_pnl = component_pnl_df[alpha_cols]
        active_count = alpha_component_pnl.notna().sum(axis=1).replace(0, np.nan)
        alpha_pnl = alpha_component_pnl.sum(axis=1, min_count=1).div(active_count)
        alpha_pnl_parts.append(alpha_pnl.rename(alpha_id))

    alpha_pnl_df = pd.concat(alpha_pnl_parts, axis=1, sort=True).sort_index() if alpha_pnl_parts else pd.DataFrame()
    return alpha_pnl_df, component_pnl_df, component_df, error_df, annualizer


def alpha_metrics(alpha_pnl_df, component_df, annualizer):
    rows = []
    if alpha_pnl_df.empty:
        return pd.DataFrame()

    for alpha_id in alpha_pnl_df.columns:
        metrics = wf_lib.calculate_pnl_metrics(alpha_pnl_df[alpha_id], annualizer)
        alpha_components = component_df[component_df["alpha_id"].astype(str) == str(alpha_id)]
        enabled = alpha_components["trade_enabled"].map(wf_lib.to_bool) if "trade_enabled" in alpha_components else pd.Series(dtype=bool)
        num_trades = pd.to_numeric(alpha_components.get("num_trades", pd.Series(dtype=float)), errors="coerce").sum()
        metrics["num_trades"] = num_trades
        metrics["TPI"] = round(num_trades / metrics["bars"] * 100, 4) if metrics.get("bars", 0) else np.nan
        rows.append(
            {
                "alpha_id": alpha_id,
                "rounds": int(len(alpha_components)),
                "enabled_rounds": int(enabled.sum()) if not enabled.empty else 0,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["SR", "TR", "alpha_id"], ascending=[False, False, True], na_position="last")


def filter_correlated_alphas(alpha_pnl_df, metrics_df, threshold, min_observations, resample_rule, use_absolute):
    corr_input_df = correlation_input(alpha_pnl_df, resample_rule)
    corr_df = corr_input_df.corr(min_periods=min_observations)
    ranking = metrics_df.sort_values(["SR", "TR", "alpha_id"], ascending=[False, False, True], na_position="last")

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
                    "comparison_correlation": np.nan,
                    "observations": np.nan,
                    "reason": "highest-ranked candidate",
                }
            )
            continue

        comparisons = []
        for kept_id in kept:
            correlation = corr_df.at[alpha_id, kept_id] if alpha_id in corr_df.index and kept_id in corr_df.columns else np.nan
            comparison_value = abs(correlation) if use_absolute else correlation
            comparisons.append((kept_id, correlation, comparison_value))

        valid = [item for item in comparisons if pd.notna(item[2])]
        if not valid:
            decision_rows.append(
                {
                    "alpha_id": alpha_id,
                    "decision": "REMOVE",
                    "blocked_by": "",
                    "correlation": np.nan,
                    "comparison_correlation": np.nan,
                    "observations": 0,
                    "reason": "insufficient overlapping PnL observations",
                }
            )
            continue

        blocker, correlation, comparison_value = max(valid, key=lambda item: item[2])
        observations = pair_observations(corr_input_df, alpha_id, blocker)
        if comparison_value >= threshold:
            decision_rows.append(
                {
                    "alpha_id": alpha_id,
                    "decision": "REMOVE",
                    "blocked_by": blocker,
                    "correlation": correlation,
                    "comparison_correlation": comparison_value,
                    "observations": observations,
                    "reason": f"correlation >= {threshold} with stronger alpha",
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
                    "comparison_correlation": comparison_value,
                    "observations": observations,
                    "reason": "below threshold versus every retained alpha",
                }
            )

    decisions_df = pd.DataFrame(decision_rows)
    decisions_df = decisions_df.merge(metrics_df, on="alpha_id", how="left")
    return kept, corr_df, decisions_df


def filter_alpha_sr(alpha_pnl_df, component_df, metrics_df, min_sr):
    sr = pd.to_numeric(metrics_df["SR"], errors="coerce")
    keep_ids = metrics_df.loc[sr >= min_sr, "alpha_id"].astype(str).tolist()
    removed_df = metrics_df.loc[~metrics_df["alpha_id"].astype(str).isin(keep_ids)].copy()
    filtered_metrics_df = metrics_df[metrics_df["alpha_id"].astype(str).isin(keep_ids)].copy()
    filtered_alpha_pnl_df = alpha_pnl_df[[col for col in alpha_pnl_df.columns if str(col) in keep_ids]].copy()
    filtered_component_df = component_df[component_df["alpha_id"].astype(str).isin(keep_ids)].copy()
    return filtered_alpha_pnl_df, filtered_component_df, filtered_metrics_df, removed_df


def add_equity_rolling_sharpe(portfolio_df, annualizer):
    if portfolio_df.empty or "cumu" not in portfolio_df.columns:
        return portfolio_df

    df = portfolio_df.copy()
    equity_pnl = df["cumu"].diff()
    if not equity_pnl.empty:
        equity_pnl.iloc[0] = df["cumu"].iloc[0]

    bars_per_day = max(1, int(round(annualizer / 365)))
    windows = {
        "rolling_sharpe_180": max(2, 180 * bars_per_day),
        "rolling_sharpe_365": max(2, annualizer),
    }
    for col, window in windows.items():
        roll = equity_pnl.rolling(window, min_periods=window)
        roll_mean = roll.mean()
        roll_std = roll.std()
        roll_count = equity_pnl.rolling(window, min_periods=1).count()
        df[col] = np.where(roll_std > 0, roll_mean / roll_std * np.sqrt(annualizer), 0.0)
        df.loc[roll_count < window, col] = np.nan
    return df


def build_equal_weight_portfolio(alpha_pnl_df, selected_ids, annualizer):
    selected_pnl = alpha_pnl_df[selected_ids].copy()
    active_count = selected_pnl.notna().sum(axis=1).replace(0, np.nan)
    weighted = selected_pnl.div(active_count, axis=0)
    portfolio_pnl = weighted.sum(axis=1, min_count=1).where(active_count.notna())
    portfolio_df = wf_lib.calculate_portfolio_timeseries(portfolio_pnl, annualizer)
    portfolio_df = add_equity_rolling_sharpe(portfolio_df, annualizer)
    equity_df = weighted.fillna(0.0).cumsum()
    contribution = weighted.sum(axis=0, min_count=1).sort_values(ascending=False)
    metrics_pnl = portfolio_df["pnl_metric"] if "pnl_metric" in portfolio_df.columns else portfolio_df["pnl"]
    metrics = wf_lib.calculate_pnl_metrics(metrics_pnl, annualizer)
    metrics["annualizer"] = annualizer
    metrics["selected_alphas"] = len(selected_ids)
    return portfolio_df, equity_df, contribution, metrics


def correlation_heatmap(corr_df, title, include_plotlyjs=False):
    if wf_lib.go is None or corr_df.empty:
        return ""
    text = np.where(pd.isna(corr_df.values), "", np.round(corr_df.values, 3).astype(str))
    fig = wf_lib.go.Figure(
        wf_lib.go.Heatmap(
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
        height=max(480, 36 * len(corr_df) + 160),
        template="plotly_white",
        xaxis_tickangle=45,
    )
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def portfolio_chart(portfolio_df, equity_df, contribution, metrics):
    if wf_lib.go is None or wf_lib.make_subplots is None or portfolio_df.empty:
        return ""
    fig = wf_lib.make_subplots(
        rows=5,
        cols=1,
        vertical_spacing=0.08,
        subplot_titles=(
            "Equal-Weight Portfolio Equity",
            "Portfolio Drawdown",
            "Selected Alpha Equity Curves",
            "Total PnL Contribution",
            "Rolling Sharpe",
        ),
        row_heights=[0.30, 0.15, 0.22, 0.15, 0.18],
    )
    fig.add_trace(wf_lib.go.Scatter(x=portfolio_df.index, y=portfolio_df["cumu"], name="Portfolio", line=dict(color="#1565c0")), row=1, col=1)
    fig.add_trace(wf_lib.go.Scatter(x=portfolio_df.index, y=portfolio_df["dd"], name="Drawdown", fill="tozeroy", line=dict(color="#c62828")), row=2, col=1)
    for alpha_id in equity_df.columns:
        fig.add_trace(wf_lib.go.Scatter(x=equity_df.index, y=equity_df[alpha_id], name=alpha_id, opacity=0.65), row=3, col=1)
    fig.add_trace(wf_lib.go.Bar(x=contribution.index, y=contribution.values, name="Contribution", marker_color="#2e7d32"), row=4, col=1)
    fig.add_trace(wf_lib.go.Scatter(x=portfolio_df.index, y=portfolio_df["rolling_sharpe_180"], name="Rolling 180d SR", line=dict(color="#6a1b9a")), row=5, col=1)
    fig.add_trace(wf_lib.go.Scatter(x=portfolio_df.index, y=portfolio_df["rolling_sharpe_365"], name="Rolling 365d SR", line=dict(color="#00897b")), row=5, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#555", row=5, col=1)
    fig.update_layout(
        height=1500,
        template="plotly_white",
        title=f"WFA Correlation Portfolio | SR={metrics.get('SR')} MDD={metrics.get('MDD')} TR={metrics.get('TR')}",
        title_x=0.5,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def render_table(df, classes="metrics-table wide-table"):
    if df.empty:
        return "<p>No rows.</p>"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[col]):
            display[col] = display[col].dt.strftime("%Y-%m-%d")
    return display.to_html(
        index=False,
        classes=classes,
        border=0,
        na_rep="",
        escape=False,
        float_format=lambda value: f"{value:.4f}",
    )


def build_html(
    args,
    recent_df,
    source_files_df,
    alpha_metrics_df,
    decisions_df,
    component_df,
    error_df,
    all_corr_df,
    selected_corr_df,
    portfolio_df,
    equity_df,
    contribution,
    portfolio_metrics,
    yearly_df,
    selected_ids,
):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_keys = ["SR", "CR", "MDD", "AR", "TR", "VAR", "TPI", "bars", "nonzero_bars", "selected_alphas", "annualizer"]
    metrics_df = pd.DataFrame(
        {
            "Metric": metric_keys,
            "Correlation Portfolio": [portfolio_metrics.get(key) for key in metric_keys],
        }
    )

    selected_metrics = alpha_metrics_df[alpha_metrics_df["alpha_id"].isin(selected_ids)].copy()
    removed_metrics = alpha_metrics_df[~alpha_metrics_df["alpha_id"].isin(selected_ids)].copy()
    decision_cols = [
        "alpha_id",
        "decision",
        "blocked_by",
        "correlation",
        "comparison_correlation",
        "observations",
        "SR",
        "CR",
        "MDD",
        "TR",
        "TPI",
        "rounds",
        "enabled_rounds",
        "reason",
    ]
    available_decision_cols = [col for col in decision_cols if col in decisions_df.columns]

    portfolio_html = portfolio_chart(portfolio_df, equity_df, contribution, portfolio_metrics)
    all_heatmap = correlation_heatmap(all_corr_df, "All Candidate WFA PnL Correlations", include_plotlyjs=False)
    selected_heatmap = correlation_heatmap(selected_corr_df, "Selected Portfolio WFA PnL Correlations")

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>WFA Correlation Portfolio</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f4f6f8; color: #202124; }}
    .dashboard {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(420px, 0.8fr); gap: 18px; height: 100vh; padding: 18px; box-sizing: border-box; }}
    .plots, .tables {{ overflow-y: auto; }}
    .header, .panel {{ background: #fff; border: 1px solid #dfe3e8; border-radius: 8px; padding: 16px; margin-bottom: 18px; box-shadow: 0 1px 2px rgba(60,64,67,.12); }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; }}
    .note {{ color: #5f6368; line-height: 1.45; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .pill {{ background: #edf2f7; border: 1px solid #d7dee8; border-radius: 999px; padding: 5px 10px; font-size: 12px; }}
    .table-scroll {{ overflow-x: auto; }}
    table.metrics-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    table.metrics-table th, table.metrics-table td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 9px; text-align: right; white-space: nowrap; }}
    table.metrics-table th:first-child, table.metrics-table td:first-child {{ text-align: left; }}
    table.metrics-table th {{ background: #f8fafc; position: sticky; top: 0; z-index: 1; }}
    .wide-table {{ min-width: 980px; }}
  </style>
</head>
<body>
  <div class="dashboard">
    <div class="plots">
      <div class="header">
        <h1>WFA Correlation Portfolio</h1>
        <div class="note">
          Candidates are stitched from walk-forward trade windows, ranked by individual stitched SR,
          filtered by {html.escape(bool_text(not args.positive_only))} &lt; {args.threshold:.2f},
          then combined with equal weights across active selected alphas.
        </div>
        <div class="meta">
          <div class="pill">candidate alphas: {alpha_metrics_df['alpha_id'].nunique() if not alpha_metrics_df.empty else 0}</div>
          <div class="pill">selected alphas: {len(selected_ids)}</div>
          <div class="pill">removed alphas: {int((decisions_df['decision'] == 'REMOVE').sum()) if not decisions_df.empty else 0}</div>
          <div class="pill">rows: {len(recent_df)}</div>
          <div class="pill">window: {html.escape(str(args.window))}</div>
          <div class="pill">enabled-round threshold: {config.WF_pass_rate_threshold}</div>
          <div class="pill">alpha SR threshold: {DEFAULT_MIN_ALPHA_SR}</div>
          <div class="pill">correlation resample: {html.escape(str(args.resample or 'native'))}</div>
          <div class="pill">generated: {generated_at}</div>
        </div>
      </div>
      <div class="panel">{portfolio_html}</div>
      <div class="panel">{all_heatmap}</div>
      <div class="panel">{selected_heatmap}</div>
    </div>
    <div class="tables">
      <div class="panel">
        <h2>Portfolio Metrics</h2>
        {render_table(metrics_df, "metrics-table")}
      </div>
      <div class="panel">
        <h2>Yearly Portfolio Metrics</h2>
        {render_table(yearly_df, "metrics-table")}
      </div>
      <div class="panel">
        <h2>Selected Alpha Metrics</h2>
        <div class="table-scroll">{render_table(selected_metrics)}</div>
      </div>
      <div class="panel">
        <h2>Removed Alpha Metrics</h2>
        <div class="table-scroll">{render_table(removed_metrics)}</div>
      </div>
      <div class="panel">
        <h2>Correlation Keep / Remove Decisions</h2>
        <div class="table-scroll">{render_table(decisions_df[available_decision_cols] if available_decision_cols else decisions_df)}</div>
      </div>
      <div class="panel">
        <h2>Individual Alpha Metrics</h2>
        <div class="table-scroll">{render_table(alpha_metrics_df)}</div>
      </div>
      <div class="panel">
        <h2>WFA Summary CSV Paths Read</h2>
        <div class="table-scroll">{render_table(source_files_df)}</div>
      </div>
    </div>
  </div>
</body>
</html>"""


def write_report(args):
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = wf_lib.discover_summary_files(input_dir)
    alpha_filter = wf_lib.target_alpha_ids(args.target_alpha_id)
    all_df = wf_lib.add_param_key(wf_lib.load_all_summaries(files, alpha_filter=alpha_filter))
    recent_df, _ = wf_lib.filter_recent_rounds(
        all_df,
        args.date_column,
        args.window,
        exclude_single_day_trade=args.exclude_single_day_trade,
    )
    pass_rate_df = compute_enabled_round_rate_by_alpha(recent_df)
    pre_wfa_filter_rows = len(recent_df)
    recent_df, wfa_pass_rows, wfa_removed_rows = filter_enabled_round_rate_rows(recent_df, pass_rate_df)
    if recent_df.empty:
        raise ValueError(f"no rows remain after enabled-round rate > {config.WF_pass_rate_threshold}")

    if "source_file" in recent_df.columns:
        source_files_df = (
            recent_df[["source_file"]]
            .dropna()
            .drop_duplicates()
            .sort_values("source_file")
            .reset_index(drop=True)
        )
    else:
        source_files_df = pd.DataFrame({"source_file": [str(path) for path in files]})

    alpha_pnl_df, component_pnl_df, component_df, error_df, annualizer = build_alpha_stitched_backtests(recent_df)
    if alpha_pnl_df.empty:
        raise ValueError("no alpha PnL series were built")

    metrics_df = alpha_metrics(alpha_pnl_df, component_df, annualizer)
    if metrics_df.empty:
        raise ValueError("no alpha metrics were built")

    pre_sr_alpha_count = int(metrics_df["alpha_id"].nunique())
    alpha_pnl_df, component_df, metrics_df, sr_removed_df = filter_alpha_sr(
        alpha_pnl_df,
        component_df,
        metrics_df,
        DEFAULT_MIN_ALPHA_SR,
    )
    if metrics_df.empty:
        raise ValueError(f"no alpha remains after stitched SR >= {DEFAULT_MIN_ALPHA_SR}")
    sr_removed_count = int(sr_removed_df["alpha_id"].nunique()) if not sr_removed_df.empty else 0
    kept_component_ids = set(component_df["component_id"].astype(str)) if "component_id" in component_df.columns else set()
    if kept_component_ids and not component_pnl_df.empty:
        component_pnl_df = component_pnl_df[[col for col in component_pnl_df.columns if str(col) in kept_component_ids]]

    selected_ids, corr_df, decisions_df = filter_correlated_alphas(
        alpha_pnl_df,
        metrics_df,
        args.threshold,
        args.min_observations,
        args.resample,
        use_absolute=not args.positive_only,
    )
    if not selected_ids:
        raise ValueError("correlation filter did not retain any alpha")

    portfolio_df, equity_df, contribution, portfolio_metrics = build_equal_weight_portfolio(
        alpha_pnl_df,
        selected_ids,
        annualizer,
    )
    selected_tpi = pd.to_numeric(metrics_df.loc[metrics_df["alpha_id"].isin(selected_ids), "TPI"], errors="coerce")
    portfolio_metrics["TPI"] = round(selected_tpi.mean(), 4) if selected_tpi.notna().any() else np.nan
    yearly_df = wf_lib.calculate_portfolio_yearly_metrics(portfolio_df, annualizer)
    selected_corr_df = corr_df.loc[selected_ids, selected_ids]

    selected_df = decisions_df[decisions_df["decision"] == "KEEP"].copy()
    removed_df = decisions_df[decisions_df["decision"] == "REMOVE"].copy()

    recent_df.to_csv(output_dir / "wf_corr_portfolio_input_rows.csv", index=False)
    source_files_df.to_csv(output_dir / "wf_corr_source_files.csv", index=False)
    alpha_pnl_df.to_csv(output_dir / "wf_corr_alpha_pnl.csv")
    component_pnl_df.to_csv(output_dir / "wf_corr_component_pnl.csv")
    component_df.to_csv(output_dir / "wf_corr_portfolio_components.csv", index=False)
    error_df.to_csv(output_dir / "wf_corr_portfolio_errors.csv", index=False)
    metrics_df.to_csv(output_dir / "wf_corr_alpha_metrics.csv", index=False)
    sr_removed_df.to_csv(output_dir / "wf_corr_sr_removed.csv", index=False)
    decisions_df.to_csv(output_dir / "wf_corr_selection.csv", index=False)
    selected_df.to_csv(output_dir / "wf_corr_portfolio_selected.csv", index=False)
    removed_df.to_csv(output_dir / "wf_corr_portfolio_removed.csv", index=False)
    corr_df.to_csv(output_dir / "wf_corr_matrix_all.csv")
    selected_corr_df.to_csv(output_dir / "wf_corr_matrix_selected.csv")
    portfolio_df.to_csv(output_dir / "wf_corr_portfolio_timeseries.csv")

    report = build_html(
        args,
        recent_df,
        source_files_df,
        metrics_df,
        decisions_df,
        component_df,
        error_df,
        corr_df,
        selected_corr_df,
        portfolio_df,
        equity_df,
        contribution,
        portfolio_metrics,
        yearly_df,
        selected_ids,
    )
    html_path = output_dir / args.output_name
    html_path.write_text(report, encoding="utf-8")

    print(f"input csv files: {len(files)}")
    print(f"enabled-round threshold: {config.WF_pass_rate_threshold}")
    print(f"enabled-round filter rows: {wfa_pass_rows} kept, {wfa_removed_rows} removed from {pre_wfa_filter_rows}")
    print(f"alpha SR threshold: {DEFAULT_MIN_ALPHA_SR}")
    print(f"alpha SR filter: {metrics_df['alpha_id'].nunique()} kept, {sr_removed_count} removed from {pre_sr_alpha_count}")
    print(f"candidate alphas: {metrics_df['alpha_id'].nunique()}")
    print(f"selected alphas ({len(selected_ids)}): {', '.join(selected_ids)}")
    print(f"removed alphas: {len(removed_df)}")
    print(f"correlation mode: {bool_text(not args.positive_only)} < {args.threshold}")
    print(f"portfolio SR: {portfolio_metrics.get('SR')}")
    print(f"portfolio MDD: {portfolio_metrics.get('MDD')}")
    print(f"portfolio TPI: {portfolio_metrics.get('TPI')}")
    print(f"source files saved: {output_dir / 'wf_corr_source_files.csv'}")
    print(f"portfolio timeseries saved: {output_dir / 'wf_corr_portfolio_timeseries.csv'}")
    print(f"selection saved: {output_dir / 'wf_corr_selection.csv'}")
    print(f"html report saved: {html_path}")
    return html_path


def main():
    args = parse_args()
    write_report(args)


if __name__ == "__main__":
    main()

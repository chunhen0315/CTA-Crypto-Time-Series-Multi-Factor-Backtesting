import ast
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

from config import (
    ASSET_PRICE_SOURCE,
    BT_END_DATE,
    BT_START_DATE,
    DATA_FOLDER as CONFIG_DATA_FOLDER,
    FT_END_DATE,
    FT_START_DATE,
    JSON_FILE,
    OUTPUT_FOLDER_BACKTEST,
    PRICE_FOLDER,
    PARAM_THRESHOLD_1,
    PARAM_THRESHOLD_2,
    PARAM_WINDOW,
    TARGET_ALPHA_ID,
    TOP_N_RESULTS,
    TPE_DEFAULT_MODEL_OPTIONS,
    TPE_MODEL_SEARCH_MODE,
    TPE_FT_SR_TOLERANCE,
    TPE_LOGIC_OPTIONS,
    TPE_MAX_TPI,
    TPE_MIN_FT_SR,
    TPE_MIN_BOTH_SIDE_TPI,
    TPE_MIN_MDD,
    TPE_MIN_SINGLE_SIDE_TPI,
    TPE_MIN_SR,
    TPE_N_TRIALS,
    TPE_PLATEAU_MIN_NEIGHBORS,
    TPE_PLATEAU_THRESHOLD_DELTA,
    TPE_PLATEAU_WINDOW_DELTA,   
    TPE_SEED,
    TPE_SIDE_OPTIONS,
    TPE_WINDOW_MAX,
    TPE_WINDOW_MIN,
    TPE_WINDOW_STEP,
    VT_END_DATE,
    VT_START_DATE,
)
from lib import entry_exit_logic_lib, models_lib, selection_scoring_lib, transformation_lib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"Fixed parameter .* is out of range for distribution", category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


DATA_FOLDER = Path(CONFIG_DATA_FOLDER)
OUTPUT_DIR = Path(OUTPUT_FOLDER_BACKTEST) / "tpe_permutation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = TPE_SEED
N_TRIALS = TPE_N_TRIALS
WINDOW_MIN = TPE_WINDOW_MIN
WINDOW_MAX = TPE_WINDOW_MAX
WINDOW_STEP = TPE_WINDOW_STEP

MIN_SR = TPE_MIN_SR
MIN_FT_SR = TPE_MIN_FT_SR
FT_SR_TOLERANCE = TPE_FT_SR_TOLERANCE
MIN_MDD = TPE_MIN_MDD
MIN_SINGLE_SIDE_TPI = TPE_MIN_SINGLE_SIDE_TPI
MIN_BOTH_SIDE_TPI = TPE_MIN_BOTH_SIDE_TPI
MAX_TPI = TPE_MAX_TPI
PLATEAU_WINDOW_DELTA = TPE_PLATEAU_WINDOW_DELTA
PLATEAU_THRESHOLD_DELTA = TPE_PLATEAU_THRESHOLD_DELTA
PLATEAU_MIN_NEIGHBORS = TPE_PLATEAU_MIN_NEIGHBORS

TIMEFRAME_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1D",
}

METRIC_ANNUALIZER_MAP = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 60 / 5,
    "15m": 365 * 24 * 60 / 15,
    "30m": 365 * 24 * 60 / 30,
    "1h": 365 * 24,
    "2h": 365 * 12,
    "4h": 365 * 6,
    "1d": 365,
}

DEFAULT_MODEL_OPTIONS = TPE_DEFAULT_MODEL_OPTIONS
MODEL_SEARCH_MODE = str(TPE_MODEL_SEARCH_MODE or "categorical").strip().lower()
LOGIC_OPTIONS = TPE_LOGIC_OPTIONS
SIDE_OPTIONS = TPE_SIDE_OPTIONS


def build_side_options(base_side):
    if SIDE_OPTIONS is None:
        return [base_side]
    options = [str(side).strip() for side in SIDE_OPTIONS if str(side).strip()]
    return options or [base_side]


def build_logic_options(base_logic):
    if LOGIC_OPTIONS is None:
        return [base_logic]
    ordered = [base_logic] + [str(item).strip() for item in LOGIC_OPTIONS if str(item).strip()]
    return [logic for idx, logic in enumerate(ordered) if logic and logic not in ordered[:idx]]


def topic_to_filename(topic: str) -> str:
    topic = topic.replace("|", "_").replace("/", "_").replace("-", "_")
    topic = topic.replace("?", "_").replace("&", "_").replace("=", "_")
    topic = "".join(char for char in topic if char.isalnum() or char == "_")
    return topic + ".csv"


def parse_time_column(values):
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        parsed = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.dt.tz_convert(None)


def load_price_df(trade_asset: str, timeframe: str, price_delay: int):
    exchange = ASSET_PRICE_SOURCE.get(trade_asset)
    if not exchange:
        raise ValueError(f"No price source for asset: {trade_asset}")

    symbol = f"{trade_asset}USDT"
    filename = f"{exchange}_1m_{symbol}.csv"
    filepath = os.path.join(PRICE_FOLDER, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Price file not found: {filepath}")

    df = pd.read_csv(filepath)
    df["time"] = parse_time_column(df["start_time"])
    df = df.set_index("time")[["close"]].astype(float)
    if price_delay != 0:
        df["close"] = df["close"].shift(price_delay)

    resolution = TIMEFRAME_MAP.get(timeframe)
    if resolution is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    return df.resample(resolution).last()


def load_csv(topic: str, column_name: str, datasource_key: str):
    file_path = DATA_FOLDER / topic_to_filename(topic)
    if not file_path.exists():
        print(f"[DATA] CSV not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df["time"] = parse_time_column(df["start_time"])

    skip_cols = {"start_time", "datetime", "time"}
    data_cols = [col for col in df.columns if col not in skip_cols]

    if column_name in data_cols:
        return df[["time", column_name]].rename(columns={column_name: datasource_key})

    if "o" in data_cols:
        def get_key(value):
            if pd.isnull(value):
                return None
            try:
                return ast.literal_eval(value).get(column_name, None)
            except Exception:
                return None

        df[datasource_key] = df["o"].apply(get_key)
        return df[["time", datasource_key]]

    if len(data_cols) == 1:
        actual_col = data_cols[0]
        print(f"[INFO] Column '{column_name}' not found, using '{actual_col}' for {datasource_key}")
        return df[["time", actual_col]].rename(columns={actual_col: datasource_key})

    partial = [col for col in data_cols if column_name in col or col in column_name]
    if partial:
        actual_col = partial[0]
        print(f"[INFO] Column '{column_name}' not found, using partial match '{actual_col}' for {datasource_key}")
        return df[["time", actual_col]].rename(columns={actual_col: datasource_key})

    print(f"[WARN] Column '{column_name}' not found in {file_path.name}. Available: {data_cols}")
    return None


def apply_diff_from_formula(df: pd.DataFrame, alpha_formula: str):
    return transformation_lib.apply_diff_from_formula(df, alpha_formula)


def compute_alpha_signal(df: pd.DataFrame, formula: str):
    return transformation_lib.compute_alpha_signal(df, formula)


def parse_datasource_structure(raw):
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw.replace("'", '"'))
    except Exception:
        return ast.literal_eval(raw)


def parse_logic_and_side(entry_exit_logic: str):
    entry_exit = str(entry_exit_logic or "").lower()
    if "_long" in entry_exit:
        return entry_exit.replace("_long", ""), "long"
    if "_short" in entry_exit:
        return entry_exit.replace("_short", ""), "short"
    return entry_exit, "both"


def get_thresholds(alpha: dict):
    values = [
        alpha.get("long_entry_threshold", alpha.get("threshold_1", 0.0)),
        alpha.get("long_exit_threshold", 0.0),
        alpha.get("short_entry_threshold", alpha.get("threshold_2", 0.0)),
        alpha.get("short_exit_threshold", 0.0),
    ]
    values = [float(value) for value in values if value is not None]
    return max(values), min(values)


def get_strategy_config(alpha: dict):
    timeframe = alpha.get("timeframe", "1h")
    logic, side = parse_logic_and_side(alpha.get("entry_exit_logic"))
    t1, t2 = get_thresholds(alpha)
    window = int(PARAM_WINDOW) if PARAM_WINDOW is not None else int(alpha.get("rolling_window_1", alpha.get("window", 0)))
    t1 = float(PARAM_THRESHOLD_1) if PARAM_THRESHOLD_1 is not None else t1
    t2 = float(PARAM_THRESHOLD_2) if PARAM_THRESHOLD_2 is not None else t2
    return {
        "timeframe": timeframe,
        "resolution": TIMEFRAME_MAP.get(timeframe),
        "annualizer": METRIC_ANNUALIZER_MAP.get(timeframe, 365 * 24),
        "window": window,
        "model": alpha.get("model"),
        "fees": float(alpha.get("fees", 0.035)) / 100,
        "logic": logic,
        "side": side,
        "t1": t1,
        "t2": t2,
    }


def build_factor_dataframe(alpha: dict):
    ds = parse_datasource_structure(alpha.get("datasource_structure", "{}"))
    merged_df = None

    for key, val in ds.items():
        df_src = load_csv(val.get("topic"), val.get("column"), key)
        if df_src is None:
            continue
        df_src = df_src.dropna(subset=[key])
        merged_df = df_src if merged_df is None else pd.merge(merged_df, df_src, on="time", how="inner")

    if merged_df is None:
        raise ValueError(f"No data loaded for {alpha.get('alpha_id')}")

    merged_df = merged_df.set_index("time").sort_index()
    alpha_formula = alpha.get("manual_alpha_formula") or alpha.get("alpha_formula", "")
    merged_df, rewritten_formula = apply_diff_from_formula(merged_df, alpha_formula)

    for col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    factor = compute_alpha_signal(merged_df, rewritten_formula)
    if factor is None:
        raise ValueError(f"Signal computation failed for {alpha.get('alpha_id')}")

    merged_df["factor"] = pd.Series(factor, index=merged_df.index).replace([np.inf, -np.inf], np.nan)
    if alpha.get("data_preprocessing") == "diff":
        merged_df["factor"] = merged_df["factor"].diff()

    return merged_df.dropna(subset=["factor"]), alpha_formula


def calculate_metrics(df, window, annualizer, fees):
    df = df.copy()
    df["chg"] = df["close"].pct_change()
    df["pos_t-1"] = df["pos"].shift(1).fillna(0)
    df["trades"] = (df["pos_t-1"] - df["pos"]).abs()
    df["pnl"] = df["pos_t-1"] * df["chg"] - df["trades"] * fees
    df["pnl"] = df["pnl"].fillna(0)
    df["cumu"] = df["pnl"].cumsum()
    df["dd"] = df["cumu"] - df["cumu"].cummax()

    pnl_std = df["pnl"].std()
    sr = round(df["pnl"].mean() / pnl_std * np.sqrt(annualizer), 4) if pnl_std and not np.isnan(pnl_std) else np.nan
    mdd = round(df["dd"].min(), 4)
    ar = round(df["pnl"].mean() * annualizer, 4)
    cr = round(ar / abs(mdd), 4) if mdd != 0 else np.nan
    tr = round(df["cumu"].iloc[-1], 4) if not df.empty else np.nan
    num_trades = int(df["trades"].sum())
    tpi = round(num_trades / (len(df) - window) * 100, 4) if len(df) > window else np.nan
    var = round(df["pnl"].quantile(0.05) * 100, 4)

    return {
        "SR": sr,
        "CR": cr,
        "MDD": mdd,
        "AR": ar,
        "TR": tr,
        "VAR": var,
        "num_trades": num_trades,
        "TPI": tpi,
    }, df


def run_backtest(base_df, window, t1, t2, logic, side, model):
    bt = models_lib.choose_model(base_df.copy(), window, model)
    bt = bt.dropna().copy()
    return entry_exit_logic_lib.signal_logic_2(bt, t1, t2, logic, side)


def warmup_period_df(base_df, start_date, end_date, window):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    history_df = base_df.loc[:end_ts].copy()
    if history_df.empty:
        return history_df

    warmup_bars = max(int(window) * 2, 0)
    start_pos = history_df.index.searchsorted(start_ts, side="left")
    calc_start_pos = max(0, start_pos - warmup_bars)
    return history_df.iloc[calc_start_pos:].copy()


def evaluate_period(base_df, price_df, start_date, end_date, resolution, window, t1, t2, logic, side, model, annualizer, fees):
    calc_df = warmup_period_df(base_df, start_date, end_date, window)
    bt = run_backtest(calc_df, window, t1, t2, logic, side, model)
    full_range = pd.date_range(start_date, end_date, freq=resolution)
    bt = bt.loc[start_date:end_date].copy()
    bt = bt.reindex(full_range).ffill()
    bt.index.name = "time"
    bt = bt.join(price_df[["close"]], how="inner").dropna(subset=["close"])
    return calculate_metrics(bt, window, annualizer, fees)


def evaluate_equity_curve(base_df, price_df, start_date, end_date, resolution, window, t1, t2, logic, side, model, annualizer, fees):
    calc_df = warmup_period_df(base_df, start_date, end_date, window)
    curve_df = run_backtest(calc_df, window, t1, t2, logic, side, model)
    full_range = pd.date_range(start_date, end_date, freq=resolution)
    curve_df = curve_df.loc[start_date:end_date].copy()
    curve_df = curve_df.reindex(full_range).ffill()
    curve_df.index.name = "time"
    curve_df = curve_df.join(price_df[["close"]], how="inner").dropna(subset=["close"])
    return calculate_metrics(curve_df, window, annualizer, fees)


def plot_top_params_equity_curves(top_df, factor_df, price_df, strategy, alpha_id, output_path):
    if top_df.empty:
        return None

    fig = go.Figure()
    params_lines = []
    plotted = 0
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        try:
            metrics, curve_df = evaluate_equity_curve(
                factor_df,
                price_df,
                BT_START_DATE,
                VT_END_DATE,
                strategy["resolution"],
                int(row["window"]),
                float(row["threshold_1"]),
                float(row["threshold_2"]),
                str(row["logic"]),
                str(row["side"]),
                str(row["model"]),
                strategy["annualizer"],
                strategy["fees"],
            )
        except Exception as exc:
            print(f"[PLOT] skip rank {rank}: {exc}")
            continue

        if curve_df.empty or "cumu" not in curve_df.columns:
            continue

        trial = row.get("trial", "")
        params_label = (
            f"#{rank} trial={trial} {row.get('model', '')} {row.get('logic', '')} {row.get('side', '')} "
            f"w={int(row['window'])} t1={float(row['threshold_1']):.2f} t2={float(row['threshold_2']):.2f}"
        )
        fig.add_trace(
            go.Scatter(
                x=curve_df.index,
                y=curve_df["cumu"],
                mode="lines+text",
                name=f"#{rank}",
                text=[""] * (len(curve_df) - 1) + [f"#{rank}"],
                textposition="middle right",
                line=dict(width=1.6),
                hovertemplate=(
                    f"{params_label}<br>"
                    "time=%{x}<br>"
                    "cumu=%{y:.4f}<extra></extra>"
                ),
            )
        )
        params_lines.append(params_label)
        plotted += 1

    if plotted == 0:
        return None

    for boundary, label in [(FT_START_DATE, "FT start"), (VT_START_DATE, "VT start")]:
        boundary_text = pd.Timestamp(boundary).strftime("%Y-%m-%d")
        fig.add_shape(
            type="line",
            x0=boundary_text,
            x1=boundary_text,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="black", width=1, dash="dash"),
        )
        fig.add_annotation(
            x=boundary_text,
            y=1,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            yanchor="bottom",
            xanchor="left",
            font=dict(size=11, color="black"),
        )

    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text="<br>".join(params_lines),
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="top",
        font=dict(size=10, color="#111"),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="rgba(0,0,0,0.25)",
        borderwidth=1,
        borderpad=6,
    )

    fig.update_layout(
        title=f"{alpha_id} Top {min(len(top_df), TOP_N_RESULTS)} Unique TPE Params Equity Curves",
        xaxis_title="Time",
        yaxis_title="Cumulative PnL",
        width=1800,
        height=1000,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=70, r=40, t=80, b=70),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.10)")
    try:
        fig.write_image(output_path, scale=2)
    except ValueError as exc:
        print(f"[PLOT] Plotly image export requires kaleido. Install with: pip install kaleido. Error: {exc}")
        return None
    return output_path


def unique_param_rows(df):
    param_cols = ["model", "logic", "side", "window", "threshold_1", "threshold_2"]
    if df.empty:
        return df.copy()
    return df.drop_duplicates(subset=param_cols, keep="first").reset_index(drop=True)


def threshold_bounds_for_model(model: str):
    model = model or ""
    if model == "percentilerank" or "meannorm" in model or "minmax" in model:
        return 0.2, 1.00, -1.00, -0.2, 0.1
    if model.endswith("_rsi") or model in {"rsi", "ersi"}:
        return 10.0, 100.0, 10.0, 100.0, 5.0
    return 0.60, 4.00, -4.00, -0.60, 0.2


def round_to_step(value, step):
    return round(round(value / step) * step, 4)


def suggest_thresholds(trial, model):
    t1_min, t1_max, t2_min, t2_max, step = threshold_bounds_for_model(model)
    t1 = trial.suggest_float("threshold_1", t1_min, t1_max, step=step)
    t2 = trial.suggest_float("threshold_2", t2_min, t2_max, step=step)
    t1 = round_to_step(t1, step)
    t2 = round_to_step(t2, step)
    if t1 <= t2:
        raise optuna.TrialPruned("threshold_1 must be greater than threshold_2")
    return t1, t2


def build_model_options(base_model):
    configured_options = DEFAULT_MODEL_OPTIONS or []
    ordered = [base_model] + [str(item).strip() for item in configured_options if str(item).strip()]
    return [model for idx, model in enumerate(ordered) if model and model not in ordered[:idx]]


def objective_score(bt_metrics, side="both"):
    sr = selection_scoring_lib.safe_float(bt_metrics.get("SR"))
    if np.isnan(sr):
        return selection_scoring_lib.NEG_INF
    return round(sr, 6)


def tpi_threshold_for_side(side):
    side = str(side or "").lower()
    if side in {"long", "short"}:
        return MIN_SINGLE_SIDE_TPI
    return MIN_BOTH_SIDE_TPI


def passes_result_filters(row, include_vt=False):
    # include_vt is kept for backward compatibility, but VT is intentionally audit-only.
    side = row.get("side")
    return selection_scoring_lib.period_pass(row, "BT", side, min_sr=MIN_SR, max_mdd=MIN_MDD) and validation_period_pass(row, "BT", "FT", side)


def validation_sr_bounds(row, primary_prefix, fallback_min=MIN_FT_SR, tolerance=FT_SR_TOLERANCE):
    primary_sr = selection_scoring_lib.safe_float(row.get(f"{primary_prefix}_SR"))
    if np.isnan(primary_sr):
        return fallback_min, np.nan
    return primary_sr * (1.0 - tolerance), primary_sr * (1.0 + tolerance)


def validation_period_pass(row, primary_prefix, validation_prefix, side):
    min_sr, max_sr = validation_sr_bounds(row, primary_prefix)
    validation_sr = selection_scoring_lib.safe_float(row.get(f"{validation_prefix}_SR"))
    if np.isnan(validation_sr):
        return False
    if not np.isnan(max_sr) and validation_sr > max_sr:
        return False
    return selection_scoring_lib.period_pass(row, validation_prefix, side, min_sr=min_sr, max_mdd=MIN_MDD)


def add_plateau_scores(trials_df):
    trials_df = trials_df.copy()
    metric_cols = [
        "plateau_count",
        "plateau_pass_count",
        "plateau_pass_rate",
        "plateau_mean_sr",
        "plateau_median_sr",
        "plateau_q25_sr",
        "plateau_worst_mdd",
        "plateau_score",
    ]
    for col in metric_cols:
        trials_df[col] = np.nan

    valid_df = trials_df[trials_df["BT_SR"].notna()].copy()
    for idx, row in valid_df.iterrows():
        same_family = valid_df[
            (valid_df["model"] == row["model"])
            & (valid_df["logic"] == row["logic"])
            & (valid_df["side"] == row["side"])
            & ((valid_df["window"] - row["window"]).abs() <= PLATEAU_WINDOW_DELTA)
            & ((valid_df["threshold_1"] - row["threshold_1"]).abs() <= PLATEAU_THRESHOLD_DELTA)
            & ((valid_df["threshold_2"] - row["threshold_2"]).abs() <= PLATEAU_THRESHOLD_DELTA)
        ]
        if same_family.empty:
            continue

        pass_mask = same_family.apply(lambda item: selection_scoring_lib.period_pass(item, "BT", item.get("side"), min_sr=MIN_SR, max_mdd=MIN_MDD), axis=1)
        pass_count = int(pass_mask.sum())
        pass_rate = pass_count / len(same_family)
        q25_sr = same_family["BT_SR"].quantile(0.25)
        mean_sr = same_family["BT_SR"].mean()
        median_sr = same_family["BT_SR"].median()
        worst_mdd = same_family["BT_MDD"].min()
        neighbor_penalty = 0.0 if len(same_family) >= PLATEAU_MIN_NEIGHBORS else 0.25

        trials_df.loc[idx, "plateau_count"] = len(same_family)
        trials_df.loc[idx, "plateau_pass_count"] = pass_count
        trials_df.loc[idx, "plateau_pass_rate"] = round(pass_rate, 4)
        trials_df.loc[idx, "plateau_mean_sr"] = round(mean_sr, 4)
        trials_df.loc[idx, "plateau_median_sr"] = round(median_sr, 4)
        trials_df.loc[idx, "plateau_q25_sr"] = round(q25_sr, 4)
        trials_df.loc[idx, "plateau_worst_mdd"] = round(worst_mdd, 4)
        trials_df.loc[idx, "plateau_score"] = round(q25_sr + 0.25 * pass_rate + 0.05 * np.log1p(pass_count) - neighbor_penalty, 4)

    return trials_df


def run_tpe_permutation(alpha):
    alpha_id = alpha.get("alpha_id")
    print(f"\nProcessing {alpha_id} with TPE sampler seed={SEED} on BT only ...")

    factor_df, alpha_formula = build_factor_dataframe(alpha)
    strategy = get_strategy_config(alpha)
    side_options = build_side_options(strategy["side"])
    if strategy["resolution"] is None:
        raise ValueError(f"Unsupported timeframe: {strategy['timeframe']}")

    price_df = load_price_df(
        trade_asset=alpha.get("trade_asset", "BTC"),
        timeframe=strategy["timeframe"],
        price_delay=int(alpha.get("shift_backtest_candle_minute", 0)),
    )

    model_options = build_model_options(strategy["model"])
    logic_options = build_logic_options(strategy["logic"])
    trial_rows = []

    def objective(trial, fixed_model=None):
        model = fixed_model or trial.suggest_categorical("model", model_options)
        logic = trial.suggest_categorical("logic", logic_options)
        side = trial.suggest_categorical("side", side_options)
        window = trial.suggest_int("window", WINDOW_MIN, WINDOW_MAX, step=WINDOW_STEP)
        t1, t2 = suggest_thresholds(trial, model)

        try:
            bt_metrics, _ = evaluate_period(
                factor_df,
                price_df,
                BT_START_DATE,
                BT_END_DATE,
                strategy["resolution"],
                window,
                t1,
                t2,
                logic,
                side,
                model,
                strategy["annualizer"],
                strategy["fees"],
            )
        except Exception as exc:
            trial_rows.append(
                {
                    "alpha_id": alpha_id,
                    "trial": trial.number,
                    "model": model,
                    "logic": logic,
                    "side": side,
                    "window": window,
                    "threshold_1": t1,
                    "threshold_2": t2,
                    "error": str(exc),
                    "objective": -1e9,
                }
            )
            raise optuna.TrialPruned(str(exc))

        score = objective_score(bt_metrics, side=side)
        trial_rows.append(
            {
                "alpha_id": alpha_id,
                "trial": trial.number,
                "model": model,
                "logic": logic,
                "side": side,
                "window": window,
                "threshold_1": t1,
                "threshold_2": t2,
                "objective": score,
                "BT_SR": bt_metrics["SR"],
                "BT_CR": bt_metrics["CR"],
                "BT_MDD": bt_metrics["MDD"],
                "BT_AR": bt_metrics["AR"],
                "BT_TR": bt_metrics["TR"],
                "BT_TPI": bt_metrics["TPI"],
                "BT_num_trades": bt_metrics["num_trades"],
            }
        )
        return score

    def run_study(fixed_model=None, study_seed=SEED, desc=None):
        sampler = optuna.samplers.TPESampler(seed=study_seed)
        study = optuna.create_study(sampler=sampler, direction="maximize")
        enqueue_model = fixed_model or strategy["model"]
        should_enqueue = fixed_model is None or fixed_model == strategy["model"]
        if should_enqueue and enqueue_model in model_options:
            study.enqueue_trial(
                {
                    **({} if fixed_model else {"model": enqueue_model}),
                    "logic": strategy["logic"],
                    "side": strategy["side"],
                    "window": strategy["window"],
                    "threshold_1": strategy["t1"],
                    "threshold_2": strategy["t2"],
                }
            )

        with tqdm(total=N_TRIALS, desc=desc or f"{alpha_id} TPE", unit="trial") as pbar:
            def callback(_study, _trial):
                pbar.update(1)

            study.optimize(lambda trial: objective(trial, fixed_model=fixed_model), n_trials=N_TRIALS, callbacks=[callback])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if MODEL_SEARCH_MODE == "per_model":
        for idx, model in enumerate(model_options):
            run_study(fixed_model=model, study_seed=SEED + idx, desc=f"{alpha_id} TPE {model}")
    elif MODEL_SEARCH_MODE == "categorical":
        run_study()
    else:
        raise ValueError(f"unsupported TPE_MODEL_SEARCH_MODE: {MODEL_SEARCH_MODE}")

    trials_df = pd.DataFrame(trial_rows)
    if trials_df.empty:
        raise ValueError(f"No completed trials for {alpha_id}")
    trials_df["trial"] = range(len(trials_df))

    metric_cols = [
        "BT_SR",
        "BT_CR",
        "BT_MDD",
        "BT_AR",
        "BT_TR",
        "BT_TPI",
        "BT_num_trades",
        "FT_SR",
        "FT_CR",
        "FT_MDD",
        "FT_AR",
        "FT_TR",
        "FT_TPI",
        "FT_num_trades",
    ]
    for col in metric_cols:
        if col not in trials_df.columns:
            trials_df[col] = np.nan

    trials_df = add_plateau_scores(trials_df)
    trials_df = trials_df.sort_values(["objective", "BT_SR", "plateau_score"], ascending=False).reset_index(drop=True)

    for col in [
        "FT_SR",
        "FT_CR",
        "FT_MDD",
        "FT_AR",
        "FT_TR",
        "FT_TPI",
        "FT_num_trades",
        "FT_error",
        "VT_SR",
        "VT_CR",
        "VT_MDD",
        "VT_AR",
        "VT_TR",
        "VT_TPI",
        "VT_num_trades",
        "VT_error",
    ]:
        if col not in trials_df.columns:
            trials_df[col] = "" if col.endswith("_error") else np.nan

    for idx, candidate in tqdm(trials_df.iterrows(), total=len(trials_df), desc=f"{alpha_id} FT/VT", unit="row"):
        try:
            ft_metrics, _ = evaluate_period(
                factor_df,
                price_df,
                FT_START_DATE,
                FT_END_DATE,
                strategy["resolution"],
                int(candidate["window"]),
                float(candidate["threshold_1"]),
                float(candidate["threshold_2"]),
                str(candidate["logic"]),
                str(candidate["side"]),
                str(candidate["model"]),
                strategy["annualizer"],
                strategy["fees"],
            )
        except Exception as exc:
            trials_df.loc[idx, "FT_error"] = str(exc)
        else:
            trials_df.loc[idx, "FT_SR"] = ft_metrics["SR"]
            trials_df.loc[idx, "FT_CR"] = ft_metrics["CR"]
            trials_df.loc[idx, "FT_MDD"] = ft_metrics["MDD"]
            trials_df.loc[idx, "FT_AR"] = ft_metrics["AR"]
            trials_df.loc[idx, "FT_TR"] = ft_metrics["TR"]
            trials_df.loc[idx, "FT_TPI"] = ft_metrics["TPI"]
            trials_df.loc[idx, "FT_num_trades"] = ft_metrics["num_trades"]

        try:
            vt_metrics, _ = evaluate_period(
                factor_df,
                price_df,
                VT_START_DATE,
                VT_END_DATE,
                strategy["resolution"],
                int(candidate["window"]),
                float(candidate["threshold_1"]),
                float(candidate["threshold_2"]),
                str(candidate["logic"]),
                str(candidate["side"]),
                str(candidate["model"]),
                strategy["annualizer"],
                strategy["fees"],
            )
        except Exception as exc:
            trials_df.loc[idx, "VT_error"] = str(exc)
            continue

        trials_df.loc[idx, "VT_SR"] = vt_metrics["SR"]
        trials_df.loc[idx, "VT_CR"] = vt_metrics["CR"]
        trials_df.loc[idx, "VT_MDD"] = vt_metrics["MDD"]
        trials_df.loc[idx, "VT_AR"] = vt_metrics["AR"]
        trials_df.loc[idx, "VT_TR"] = vt_metrics["TR"]
        trials_df.loc[idx, "VT_TPI"] = vt_metrics["TPI"]
        trials_df.loc[idx, "VT_num_trades"] = vt_metrics["num_trades"]

    trials_df["BT_ROBUST_SCORE"] = trials_df.apply(
        lambda row: objective_score(
            {"SR": row.get("BT_SR"), "CR": row.get("BT_CR"), "MDD": row.get("BT_MDD"), "TPI": row.get("BT_TPI"), "num_trades": row.get("BT_num_trades")},
            side=row.get("side"),
        ),
        axis=1,
    )
    trials_df["BT_PASS"] = trials_df.apply(lambda row: selection_scoring_lib.period_pass(row, "BT", row.get("side"), min_sr=MIN_SR, max_mdd=MIN_MDD), axis=1)
    trials_df[["FT_MIN_SR", "FT_MAX_SR"]] = trials_df.apply(
        lambda row: pd.Series(validation_sr_bounds(row, "BT")),
        axis=1,
    )
    trials_df["FT_SR_WITHIN_BT_TOLERANCE"] = trials_df.apply(
        lambda row: not pd.isna(row.get("FT_SR"))
        and row.get("FT_SR") >= row.get("FT_MIN_SR")
        and (pd.isna(row.get("FT_MAX_SR")) or row.get("FT_SR") <= row.get("FT_MAX_SR")),
        axis=1,
    )
    trials_df["FT_PASS"] = trials_df.apply(lambda row: validation_period_pass(row, "BT", "FT", row.get("side")), axis=1)
    trials_df["VT_PASS"] = trials_df.apply(lambda row: selection_scoring_lib.period_pass(row, "VT", row.get("side"), min_sr=1.0, max_mdd=MIN_MDD), axis=1)
    trials_df["VT_AUDIT_ONLY"] = True
    trials_df["SR_DRIFT_BT_FT"] = trials_df.apply(lambda row: selection_scoring_lib.sr_drift(row, "BT", "FT"), axis=1)
    trials_df["MDD_DRIFT_BT_FT"] = trials_df.apply(lambda row: selection_scoring_lib.mdd_drift(row, "BT", "FT"), axis=1)
    trials_df["TPI_DRIFT_BT_FT"] = trials_df.apply(lambda row: selection_scoring_lib.tpi_drift(row, "BT", "FT"), axis=1)
    trials_df["SELECTION_SCORE"] = trials_df.apply(
        lambda row: selection_scoring_lib.selection_score(
            row,
            "BT",
            "FT",
            side=row.get("side"),
            min_primary_sr=MIN_SR,
            min_validation_sr=row.get("FT_MIN_SR", MIN_FT_SR),
            max_mdd=MIN_MDD,
        ),
        axis=1,
    )

    trials_df["FILTER_PASS"] = trials_df["BT_PASS"] & trials_df["FT_PASS"]
    trials_df["VT_SR_POSITIVE"] = pd.to_numeric(trials_df["VT_SR"], errors="coerce").gt(0)
    trials_df["FILTER_SELECT_PASS"] = trials_df["FILTER_PASS"]

    trials_df = trials_df.sort_values(
        ["FILTER_SELECT_PASS", "SELECTION_SCORE", "FT_SR", "BT_SR"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    filtered_df = trials_df[trials_df["FILTER_SELECT_PASS"]].copy()
    filtered_df = filtered_df.sort_values(
        ["SELECTION_SCORE"],
        ascending=[False],
        na_position="last",
    ).reset_index(drop=True)
    trials_df["PARAM_SELECTED"] = False
    filtered_df["PARAM_SELECTED"] = False
    if not filtered_df.empty:
        filtered_df.loc[filtered_df.head(1).index, "PARAM_SELECTED"] = True

    alpha_dir = OUTPUT_DIR / alpha_id
    alpha_dir.mkdir(parents=True, exist_ok=True)
    trials_path = alpha_dir / f"{alpha_id}_tpe_trials.csv"
    filtered_path = alpha_dir / f"{alpha_id}_tpe_filtered.csv"
    top_path = alpha_dir / f"{alpha_id}_tpe_top_{TOP_N_RESULTS}.csv"
    equity_plot_limit = min(10, TOP_N_RESULTS)
    equity_plot_path = alpha_dir / f"{alpha_id}_tpe_top_{equity_plot_limit}_equity_curves.png"
    trials_df.to_csv(trials_path, index=False)
    filtered_df.to_csv(filtered_path, index=False)
    unique_filtered_df = unique_param_rows(filtered_df)
    top_results_df = unique_filtered_df.head(TOP_N_RESULTS).copy()
    top_results_df.to_csv(top_path, index=False)
    top_plot_df = top_results_df.head(equity_plot_limit).copy()
    saved_equity_plot_path = plot_top_params_equity_curves(
        top_plot_df,
        factor_df,
        price_df,
        strategy,
        alpha_id,
        equity_plot_path,
    )

    print(f"formula: {alpha_formula}")
    print(f"base: model={strategy['model']} logic={strategy['logic']} window={strategy['window']} t1={strategy['t1']} t2={strategy['t2']}")
    if filtered_df.empty:
        print("best filtered: none passed BT/FT filters")
    else:
        best = filtered_df.iloc[0]
        print(
            "best filtered by highest selection score: "
            f"model={best['model']} logic={best['logic']} side={best['side']} "
            f"window={int(best['window'])} t1={best['threshold_1']} t2={best['threshold_2']} "
            f"BT_SR={best['BT_SR']} FT_SR={best['FT_SR']} VT_SR={best['VT_SR']} "
            f"SR_DRIFT_BT_FT={best['SR_DRIFT_BT_FT']} "
            f"BT_MDD={best['BT_MDD']} FT_MDD={best['FT_MDD']} VT_MDD={best['VT_MDD']} "
            f"BT_TPI={best['BT_TPI']} FT_TPI={best['FT_TPI']} VT_TPI={best['VT_TPI']} "
            f"SELECTION_SCORE={best['SELECTION_SCORE']} VT_AUDIT_ONLY={best['VT_AUDIT_ONLY']}"
        )

    vt_top_10 = top_results_df.head(10)
    if vt_top_10.empty:
        print("top 10 unique filtered results by selection score: none passed BT/FT filters")
    else:
        print(
            "top 10 unique filtered results sorted by highest selection score:\n"
            + vt_top_10[
                [
                    "model",
                    "logic",
                    "side",
                    "window",
                    "threshold_1",
                    "threshold_2",
                    "SELECTION_SCORE",
                    "BT_SR",
                    "FT_SR",
                    "VT_SR",
                    "BT_MDD",
                    "FT_MDD",
                    "VT_MDD",
                    "BT_TPI",
                    "FT_TPI",
                    "VT_TPI",
                ]
            ].to_string(index=False)
        )
    print(f"FT counted: {trials_df['FT_SR'].notna().sum()} / {len(trials_df)}")
    print(f"VT counted: {trials_df['VT_SR'].notna().sum()} / {len(trials_df)}")
    print(f"filtered results: {len(filtered_df)} / {len(trials_df)}")
    print(f"unique filtered params: {len(unique_filtered_df)} / {len(filtered_df)}")
    # print(f"saved trials: {trials_path}")
    # print(f"saved all passing filtered results: {filtered_path}")
    print(f"saved filtered top results: {top_path}")
    # if saved_equity_plot_path is not None:
    #     print(f"saved top params equity curves: {saved_equity_plot_path}")

    return filtered_df


def selected_alphas(alphas):
    if TARGET_ALPHA_ID is None:
        return alphas

    target_ids = TARGET_ALPHA_ID if isinstance(TARGET_ALPHA_ID, (list, tuple, set)) else [TARGET_ALPHA_ID]
    target_ids = [item for item in target_ids if item is not None]
    selected = [alpha for alpha in alphas if alpha.get("alpha_id") in set(target_ids)]
    if not selected:
        raise ValueError(f"Alpha ID not found: {TARGET_ALPHA_ID}")
    return selected


def main():
    with open(JSON_FILE, "r", encoding="utf-8") as file:
        alphas = json.load(file)

    all_filtered_results = []
    all_top_results = []
    for alpha in selected_alphas(alphas):
        result = run_tpe_permutation(alpha)
        all_filtered_results.append(result.copy())
        all_top_results.append(result.head(TOP_N_RESULTS).copy())

    if all_filtered_results:
        filtered_summary_df = pd.concat(all_filtered_results, ignore_index=True)
        filtered_summary_path = OUTPUT_DIR / "tpe_filtered_results_summary.csv"
        filtered_summary_df.to_csv(filtered_summary_path, index=False)
        # print(f"\nall passing filtered summary saved: {filtered_summary_path}")

        top_summary_df = pd.concat(all_top_results, ignore_index=True)
        top_summary_path = OUTPUT_DIR / "tpe_top_results_summary.csv"
        top_summary_df.to_csv(top_summary_path, index=False)
        # print(f"top results summary saved: {top_summary_path}")


if __name__ == "__main__":
    main()

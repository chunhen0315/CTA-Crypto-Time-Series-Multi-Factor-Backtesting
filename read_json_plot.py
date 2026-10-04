import json
import os
import re
import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import probplot

from lib import entry_exit_logic_lib, models_lib, transformation_lib
from config import (
    JSON_FILE,
    DATA_FOLDER as CONFIG_DATA_FOLDER,
    PRICE_FOLDER,
    TARGET_ALPHA_ID,
    ASSET_PRICE_SOURCE,
    START_DATE,
    END_DATE,
    PARAM_WINDOW,
    PARAM_THRESHOLD_1,
    PARAM_THRESHOLD_2,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

# =========================================================
# CONFIG
# =========================================================
DATA_FOLDER = Path(CONFIG_DATA_FOLDER)

TIMEFRAME_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1D"
}

METRIC_ANNUALIZER_MAP = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 60 / 5,
    "15m": 365 * 24 * 60 / 15,
    "30m": 365 * 24 * 60 / 30,
    "1h": 365 * 24,
    "2h": 365 * 12,
    "4h": 365 * 6,
    "1d": 365
}



# =========================================================
# HELPERS
# =========================================================
def apply_param_overrides(window, threshold_1, threshold_2):
    if PARAM_WINDOW is not None:
        window = int(PARAM_WINDOW)
    if PARAM_THRESHOLD_1 is not None:
        threshold_1 = float(PARAM_THRESHOLD_1)
    if PARAM_THRESHOLD_2 is not None:
        threshold_2 = float(PARAM_THRESHOLD_2)
    return window, threshold_1, threshold_2


def topic_to_filename(topic: str) -> str:
    topic = topic.replace("|", "_").replace("/", "_").replace("-", "_")
    topic = topic.replace("?", "_").replace("&", "_").replace("=", "_")
    topic = "".join(c for c in topic if c.isalnum() or c == "_")
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
    if not resolution:
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
    data_cols = [c for c in df.columns if c not in skip_cols]

    if column_name in data_cols:
        df = df[["time", column_name]].rename(columns={column_name: datasource_key})

    elif "o" in data_cols:
        def get_key(x):
            if pd.isnull(x):
                return None
            try:
                return ast.literal_eval(x).get(column_name, None)
            except Exception:
                return None

        df[datasource_key] = df["o"].apply(get_key)
        df = df[["time", datasource_key]]

    elif len(data_cols) == 1:
        actual_col = data_cols[0]
        print(f"[INFO] Column '{column_name}' not found, using '{actual_col}' for {datasource_key}")
        df = df[["time", actual_col]].rename(columns={actual_col: datasource_key})

    else:
        partial = [c for c in data_cols if column_name in c or c in column_name]
        if partial:
            actual_col = partial[0]
            print(f"[INFO] Column '{column_name}' not found, using partial match '{actual_col}' for {datasource_key}")
            df = df[["time", actual_col]].rename(columns={actual_col: datasource_key})
        else:
            print(f"[WARN] Column '{column_name}' not found in {file_path.name}. Available: {data_cols}")
            return None

    return df


def apply_diff_from_formula(df: pd.DataFrame, alpha_formula: str):
    return transformation_lib.apply_diff_from_formula(df, alpha_formula)


def compute_alpha_signal(df: pd.DataFrame, formula: str):
    return transformation_lib.compute_alpha_signal(df, formula)


def parse_formula_columns(alpha_formula: str):
    pattern = r"df\['([^']+)'\]"
    cols = re.findall(pattern, alpha_formula)
    cols = list(dict.fromkeys(cols))
    return cols


def calculate_pnl_and_position(df, window, model, logic, side, t1, t2, annualizer, fees):
    bt = models_lib.choose_model(df.copy(), window, model)
    bt = bt.dropna().copy()
    bt = entry_exit_logic_lib.signal_logic_2(bt, t1, t2, logic, side)

    bt["chg"] = bt["close"].pct_change()
    bt["pos_t-1"] = bt["pos"].shift(1).fillna(0)
    bt["trades"] = (bt["pos_t-1"] - bt["pos"]).abs()
    bt["pnl"] = bt["pos_t-1"] * bt["chg"] - bt["trades"] * fees
    bt["pnl"] = bt["pnl"].fillna(0)
    bt["cumu"] = bt["pnl"].cumsum()

    return bt


def plot_transformation_qq_axes(ax_transform, ax_qq, df_plot, max_points=5000):
    """Draw raw-factor transformation and signal Q-Q diagnostics left/right."""
    required_columns = ["factor", "signal"]
    if any(column not in df_plot.columns for column in required_columns):
        message = "Raw factor or transformed signal is unavailable"
        for axis in (ax_transform, ax_qq):
            axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
        ax_transform.set_title("Transformation: Raw Factor vs Model Signal")
        ax_qq.set_title("Q-Q Plot: Transformed Signal vs Normal")
        return

    diagnostic_df = df_plot[required_columns].copy()
    for column in required_columns:
        diagnostic_df[column] = pd.to_numeric(diagnostic_df[column], errors="coerce")
    diagnostic_df = diagnostic_df.replace([np.inf, -np.inf], np.nan).dropna()

    if diagnostic_df.empty:
        for axis in (ax_transform, ax_qq):
            axis.text(0.5, 0.5, "No finite observations", ha="center", va="center", transform=axis.transAxes)
        ax_transform.set_title("Transformation: Raw Factor vs Model Signal")
        ax_qq.set_title("Q-Q Plot: Transformed Signal vs Normal")
        return

    if len(diagnostic_df) > max_points:
        sample_positions = np.linspace(0, len(diagnostic_df) - 1, max_points, dtype=int)
        diagnostic_df = diagnostic_df.iloc[sample_positions]

    ax_transform.scatter(
        diagnostic_df["factor"],
        diagnostic_df["signal"],
        s=10,
        alpha=0.35,
        color="#1565c0",
        edgecolors="none",
    )
    ax_transform.set_title("Transformation: Raw Factor vs Model Signal")
    ax_transform.set_xlabel("Raw Factor")
    ax_transform.set_ylabel("Model Signal")
    ax_transform.grid(alpha=0.2)

    theoretical_quantiles, ordered_signal = probplot(
        diagnostic_df["signal"].to_numpy(),
        dist="norm",
        fit=False,
    )
    theoretical_quantiles = np.asarray(theoretical_quantiles, dtype=float)
    ordered_signal = np.asarray(ordered_signal, dtype=float)

    ax_qq.scatter(
        theoretical_quantiles,
        ordered_signal,
        s=10,
        alpha=0.45,
        color="#ef6c00",
        edgecolors="none",
        label="Signal quantiles",
    )
    if len(theoretical_quantiles) >= 2:
        slope, intercept = np.polyfit(theoretical_quantiles, ordered_signal, 1)
        reference_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
        ax_qq.plot(
            reference_x,
            intercept + slope * reference_x,
            color="#424242",
            linestyle="--",
            linewidth=1.5,
            label="Normal reference",
        )

    ax_qq.set_title("Q-Q Plot: Transformed Signal vs Normal")
    ax_qq.set_xlabel("Theoretical Normal Quantile")
    ax_qq.set_ylabel("Ordered Signal Quantile")
    ax_qq.grid(alpha=0.2)
    ax_qq.legend(loc="best")


def create_research_dashboard_axes():
    """Create left-side research charts and right-side diagnostics."""
    fig = plt.figure(figsize=(24, 20))
    grid = fig.add_gridspec(
        nrows=6,
        ncols=2,
        width_ratios=[1.65, 1.0],
        hspace=0.45,
        wspace=0.25,
    )

    left_axes = []
    for row in range(6):
        shared_axis = left_axes[0] if left_axes else None
        left_axes.append(fig.add_subplot(grid[row, 0], sharex=shared_axis))

    ax_transform = fig.add_subplot(grid[0:3, 1])
    ax_qq = fig.add_subplot(grid[3:6, 1])
    return fig, left_axes, ax_transform, ax_qq


# =========================================================
# MAIN
# =========================================================
def main():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Support both the original alpha-list JSON format and the newer
    # selection bundle format.  A bundle stores the runnable alpha payload
    # under each record's ``alpha_json_copy`` key.
    if isinstance(payload, list):
        alphas = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        alphas = [
            record["alpha_json_copy"]
            for record in payload["records"]
            if isinstance(record, dict)
            and isinstance(record.get("alpha_json_copy"), dict)
        ]
    else:
        raise ValueError(
            f"Unsupported alpha JSON format in {JSON_FILE}; expected a list "
            "or an object containing a 'records' list."
        )

    target_ids = TARGET_ALPHA_ID if isinstance(TARGET_ALPHA_ID, (list, tuple, set)) else [TARGET_ALPHA_ID]
    target_ids = [item for item in target_ids if item is not None]
    target_id = target_ids[0] if target_ids else None
    alpha = next((a for a in alphas if a.get("alpha_id") == target_id), None)
    if alpha is None:
        raise ValueError(f"Alpha ID not found: {TARGET_ALPHA_ID}")

    print(f"\nProcessing {target_id} ...")

    ds_raw = alpha.get("datasource_structure", "{}")
    try:
        ds = json.loads(ds_raw.replace("'", '"'))
    except Exception:
        ds = ast.literal_eval(ds_raw)

    merged_df = None
    for key, val in ds.items():
        topic = val.get("topic")
        col = val.get("column")
        df_src = load_csv(topic, col, key)
        if df_src is None:
            continue
        df_src = df_src.dropna(subset=[key])
        merged_df = df_src if merged_df is None else pd.merge(merged_df, df_src, on="time", how="inner")

    if merged_df is None:
        raise ValueError(f"No data loaded for {target_id}")

    merged_df = merged_df.set_index("time").sort_index()

    alpha_formula = alpha.get("manual_alpha_formula") or alpha.get("alpha_formula", "")
    merged_df, rewritten_formula = apply_diff_from_formula(merged_df, alpha_formula)

    for c in merged_df.columns:
        merged_df[c] = pd.to_numeric(merged_df[c], errors="coerce")

    signal = compute_alpha_signal(merged_df, rewritten_formula)
    if signal is None:
        raise ValueError(f"Signal computation failed for {target_id}")

    merged_df["factor"] = signal
    merged_df["factor"] = merged_df["factor"].replace([np.inf, -np.inf], np.nan)

    if alpha.get("data_preprocessing") == "diff":
        merged_df["factor"] = merged_df["factor"].diff()

    merged_df = merged_df.dropna(subset=["factor"])

    timeframe = alpha.get("timeframe", "1h")
    trade_asset = alpha.get("trade_asset", "BTC")
    price_delay = int(alpha.get("shift_backtest_candle_minute", 0))
    resolution = TIMEFRAME_MAP.get(timeframe)
    annualizer = METRIC_ANNUALIZER_MAP.get(timeframe, 365 * 24)
    fees = float(alpha.get("fees", 0.035)) / 100
    window = int(alpha.get("rolling_window_1", alpha.get("window", 0)))
    model = alpha.get("model")

    entry_exit = alpha.get("entry_exit_logic", "").lower()
    if "_long" in entry_exit:
        logic, side = entry_exit.replace("_long", ""), "long"
    elif "_short" in entry_exit:
        logic, side = entry_exit.replace("_short", ""), "short"
    else:
        logic, side = entry_exit, "both"

    long_entry_threshold = float(alpha.get("long_entry_threshold", alpha.get("threshold_1", 0.0)))
    long_exit_threshold = float(alpha.get("long_exit_threshold", 0.0))
    short_entry_threshold = float(alpha.get("short_entry_threshold", alpha.get("threshold_2", 0.0)))
    short_exit_threshold = float(alpha.get("short_exit_threshold", 0.0))

    t1 = max(long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold)
    t2 = min(long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold)
    window, t1, t2 = apply_param_overrides(window, t1, t2)

    price_df = load_price_df(trade_asset, timeframe, price_delay)

    full_df = merged_df.join(price_df[["close"]], how="inner")
    full_df = full_df.loc[START_DATE:END_DATE].copy()
    full_df = full_df.resample(resolution).last()
    full_df = full_df.dropna(subset=["close", "factor"])

    result_df = calculate_pnl_and_position(
        full_df, window, model, logic, side, t1, t2, annualizer, fees
    )

    formula_cols = parse_formula_columns(alpha_formula)
    factor_a = formula_cols[0] if len(formula_cols) >= 1 else None
    factor_b = formula_cols[1] if len(formula_cols) >= 2 else None

    print(f"alpha_id: {alpha.get('alpha_id')}")
    print(f"formula: {alpha_formula}")
    print(f"factor_a: {factor_a}")
    print(f"factor_b: {factor_b}")
    print(f"rows: {len(result_df)}")

    fig, axes, ax_transform, ax_qq = create_research_dashboard_axes()

    if factor_a and factor_a in result_df.columns:
        axes[0].plot(result_df.index, result_df[factor_a])
        axes[0].set_title(f"Factor A: {factor_a}")
        axes[0].set_ylabel("Value")
    else:
        axes[0].text(0.5, 0.5, "Factor A not found", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Factor A")

    if factor_b and factor_b in result_df.columns:
        axes[1].plot(result_df.index, result_df[factor_b])
        axes[1].set_title(f"Factor B: {factor_b}")
        axes[1].set_ylabel("Value")
    else:
        axes[1].text(0.5, 0.5, "Factor B not found", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Factor B")

    axes[2].plot(result_df.index, result_df["factor"])
    axes[2].set_title("Final Factor")
    axes[2].set_ylabel("Factor")

    axes[3].plot(result_df.index, result_df["close"])
    axes[3].set_title("Price")
    axes[3].set_ylabel("Close")

    axes[4].plot(result_df.index, result_df["cumu"])
    axes[4].set_title("Cumulative PnL")
    axes[4].set_ylabel("Cumu")

    axes[5].step(result_df.index, result_df["pos"], where="post")
    axes[5].set_title("Position Time Series")
    axes[5].set_ylabel("Position")
    axes[5].set_xlabel("Time")

    plot_transformation_qq_axes(ax_transform, ax_qq, result_df)

    plt.suptitle(f"{target_id} Research Plot", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.show()


if __name__ == "__main__":
    main()

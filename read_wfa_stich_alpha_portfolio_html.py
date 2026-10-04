import argparse
import ast
import html
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover - report still works without charts
    go = None
    make_subplots = None

from lib import entry_exit_logic_lib, transformation_lib
from config import (
    ASSET_PRICE_SOURCE,
    DATA_FOLDER as CONFIG_DATA_FOLDER,
    JSON_FILE,
    OUTPUT_FOLDER_BACKTEST,
    PRICE_FOLDER,
    TARGET_ALPHA_ID,
)


DEFAULT_INPUT_DIR = Path(OUTPUT_FOLDER_BACKTEST) / "tpe_walk_forward"
WFA_SELECTED_SUMMARY_RE = re.compile(
    r"^(?P<alpha_id>.+)_wfa_selected_summary(?:_\d+_\d+_\d+_\d+(?:_full_pass)?)?\.csv$"
)
DATA_FOLDER = Path(CONFIG_DATA_FOLDER)
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
DATE_COLUMNS = ["train_start", "test_start", "trade_start"]
PASS_COLUMNS = ["TRAIN_PASS", "TEST_PASS", "TRAIN_TEST_PASS", "TRADE_PASS", "FULL_PASS"]
PARAM_COLUMNS = ["model", "logic", "side", "window", "threshold_1", "threshold_2"]
WFA_COMPONENT_COLUMNS = [
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "trade_start",
    "trade_end",
    "trial",
    "objective",
    "TRAIN_SR",
    "TRAIN_CR",
    "TRAIN_MDD",
    "TRAIN_AR",
    "TRAIN_TR",
    "TRAIN_TPI",
    "TRAIN_num_trades",
    "TRAIN_PASS",
    "TEST_SR",
    "TEST_CR",
    "TEST_MDD",
    "TEST_AR",
    "TEST_TR",
    "TEST_TPI",
    "TEST_num_trades",
    "TEST_PASS",
    "TRAIN_TEST_PASS",
    "TRADE_SR",
    "TRADE_CR",
    "TRADE_MDD",
    "TRADE_AR",
    "TRADE_TR",
    "TRADE_TPI",
    "TRADE_num_trades",
    "TRADE_PASS",
    "TRADE_SR_POSITIVE",
    "FILTER_PASS",
    "FILTER_SELECT_PASS",
    "PARAM_SELECTED",
    "WFA_SELECTED_PASS",
    "SELECTION_MODE",
]
RECENT_COLUMNS = [
    "alpha_id",
    "round",
    "round_date",
    "train_start",
    "test_start",
    "trade_start",
    "trade_end",
    "model",
    "logic",
    "side",
    "window",
    "threshold_1",
    "threshold_2",
    "TRAIN_SR",
    "TEST_SR",
    "TRADE_SR",
    "FULL_SR",
    "TRAIN_MDD",
    "TEST_MDD",
    "TRADE_MDD",
    "FULL_MDD",
    "TRAIN_TPI",
    "TEST_TPI",
    "TRADE_TPI",
    "FULL_TPI",
    "TRADE_num_trades",
    "SELECTION_MODE",
    "TRAIN_TEST_PASS",
    "TRADE_PASS",
    "FULL_PASS",
]
INTERNAL_REPORT_COLUMNS = {"param_key", "round_date"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a portfolio HTML report from *_wf_round_best_summary.csv files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder to scan for walk-forward summary CSVs. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for HTML/CSV outputs. Default: same as --input-dir.",
    )
    parser.add_argument(
        "--date-column",
        choices=DATE_COLUMNS,
        default="trade_start",
        help="Round date used for latest-window filtering. Default: trade_start.",
    )
    parser.add_argument(
        "--window",
        choices=["latest_round", "all", "calendar_year", "rolling_12m"],
        default="all",
        help="latest_round keeps only the latest row per alpha; all keeps every CSV row; calendar_year keeps Jan 1 through latest round; rolling_12m keeps the last 12 months.",
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
        "--output-name",
        default="wf_round_best_portfolio.html",
        help="Output HTML filename.",
    )
    parser.add_argument(
        "--max-component-traces",
        type=int,
        default=30,
        help="Maximum component equity traces to draw in the detail chart.",
    )
    return parser.parse_args()


def target_alpha_ids(cli_values):
    if cli_values:
        return {str(item).strip() for item in cli_values if str(item).strip()}

    if TARGET_ALPHA_ID is None:
        return None

    if isinstance(TARGET_ALPHA_ID, str):
        value = TARGET_ALPHA_ID.strip()
        if not value or value.lower() in {"none", "all"}:
            return None
        return {value}

    values = {str(item).strip() for item in TARGET_ALPHA_ID if item is not None and str(item).strip()}
    return values or None


SUMMARY_FILE_PATTERNS = [
    "*_wfa_selected_summary*.csv",
    "wfa_selected_summary.csv",
    "*_wf_round_best_summary.csv",
]


def discover_summary_files(input_dir):
    if not input_dir.exists():
        raise FileNotFoundError(f"input folder not found: {input_dir}")

    wfa_files = sorted(
        {
            path
            for pattern in ["*_wfa_selected_summary*.csv", "wfa_selected_summary.csv"]
            for path in input_dir.rglob(pattern)
            if path.name == "wfa_selected_summary.csv" or WFA_SELECTED_SUMMARY_RE.match(path.name)
        }
    )
    if wfa_files:
        per_alpha_wfa_files = [path for path in wfa_files if path.parent != input_dir]
        if per_alpha_wfa_files:
            wfa_files = per_alpha_wfa_files

        chosen = {}
        for path in wfa_files:
            alpha_id = alpha_id_from_summary_path(path)
            is_named = WFA_SELECTED_SUMMARY_RE.match(path.name) is not None
            is_alpha_folder = path.parent != input_dir
            priority = (2 if is_named else 1 if is_alpha_folder else 0, path.stat().st_mtime)
            if alpha_id not in chosen or priority > chosen[alpha_id][0]:
                chosen[alpha_id] = (priority, path)
        return sorted(item[1] for item in chosen.values())

    files = sorted(input_dir.rglob("*_wf_round_best_summary.csv"))
    if not files:
        patterns = ", ".join(SUMMARY_FILE_PATTERNS)
        raise FileNotFoundError(f"no walk-forward summary CSV files ({patterns}) found under: {input_dir}")
    return files


def alpha_id_from_summary_path(path):
    name = path.name
    match = WFA_SELECTED_SUMMARY_RE.match(name)
    if match:
        return match.group("alpha_id")
    if name.endswith("_wf_round_best_summary.csv"):
        return name.replace("_wf_round_best_summary.csv", "")
    if name == "wfa_selected_summary.csv":
        return path.parent.name
    return path.stem


def read_summary_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    text_cols = df.select_dtypes(include=["object"]).columns
    df[text_cols] = df[text_cols].apply(lambda col: col.str.strip())
    if df.empty:
        return df

    if "alpha_id" not in df.columns:
        df["alpha_id"] = alpha_id_from_summary_path(path)

    for col in DATE_COLUMNS + ["train_end", "test_end", "trade_end", "full_start", "full_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in df.columns:
        if col.endswith("_PASS") or col in PASS_COLUMNS:
            df[col] = df[col].map(to_bool)

    df["source_file"] = str(path)
    return df


def load_all_summaries(files, alpha_filter=None):
    frames = []
    for path in files:
        df = read_summary_csv(path)
        if df.empty:
            continue
        if alpha_filter is not None:
            df = df[df["alpha_id"].astype(str).isin(alpha_filter)]
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("no non-empty walk-forward summary rows matched the input filters")

    return pd.concat(frames, ignore_index=True)


def to_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def recent_window_start(latest_date, window):
    latest_date = pd.Timestamp(latest_date).normalize()
    if window == "rolling_12m":
        return latest_date - pd.DateOffset(months=12) + pd.Timedelta(days=1)
    return pd.Timestamp(year=latest_date.year, month=1, day=1)


def add_param_key(df):
    df = df.copy()
    for col in PARAM_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    def fmt(value):
        if pd.isna(value):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    df["param_key"] = df[PARAM_COLUMNS].apply(
        lambda row: " | ".join(f"{col}={fmt(row[col])}" for col in PARAM_COLUMNS),
        axis=1,
    )
    return df


def drop_single_day_trade_rows(df):
    if not {"trade_start", "trade_end"}.issubset(df.columns):
        return df
    return df[df["trade_start"].dt.normalize() != df["trade_end"].dt.normalize()].copy()


def filter_recent_rounds(df, date_column, window, exclude_single_day_trade=False):
    if date_column not in df.columns:
        raise ValueError(f"missing date column: {date_column}")

    if exclude_single_day_trade:
        df = drop_single_day_trade_rows(df)

    if window == "latest_round":
        base_df = df.dropna(subset=[date_column]).copy()
        if base_df.empty:
            raise ValueError(f"no rows have usable {date_column} values")

        sort_cols = ["alpha_id", date_column]
        if "round" in base_df.columns:
            sort_cols.append("round")
        selected = base_df.sort_values(sort_cols).groupby("alpha_id", as_index=False).tail(1).copy()
        selected["round_date"] = selected[date_column]
        windows = []
        for alpha_id, alpha_df in base_df.groupby("alpha_id", sort=True):
            chosen = selected[selected["alpha_id"] == alpha_id].iloc[0]
            windows.append(
                {
                    "alpha_id": alpha_id,
                    "window_start": chosen[date_column],
                    "latest_round_date": chosen[date_column],
                    "rounds_in_window": 1,
                }
            )
        return selected.reset_index(drop=True), pd.DataFrame(windows)

    if window == "all":
        selected = df.dropna(subset=[date_column]).copy()
        selected["round_date"] = selected[date_column]
        windows = []
        for alpha_id, alpha_df in selected.groupby("alpha_id", sort=True):
            windows.append(
                {
                    "alpha_id": alpha_id,
                    "window_start": alpha_df[date_column].min(),
                    "latest_round_date": alpha_df[date_column].max(),
                    "rounds_in_window": len(alpha_df),
                }
            )
        if selected.empty:
            raise ValueError(f"no rows have usable {date_column} values")
        return selected, pd.DataFrame(windows)

    rows = []
    windows = []
    for alpha_id, alpha_df in df.dropna(subset=[date_column]).groupby("alpha_id", sort=True):
        latest_date = alpha_df[date_column].max()
        start_date = recent_window_start(latest_date, window)
        selected = alpha_df[(alpha_df[date_column] >= start_date) & (alpha_df[date_column] <= latest_date)].copy()
        selected["round_date"] = selected[date_column]
        rows.append(selected)
        windows.append(
            {
                "alpha_id": alpha_id,
                "window_start": start_date,
                "latest_round_date": latest_date,
                "rounds_in_window": len(selected),
            }
        )

    if not rows:
        raise ValueError(f"no rows have usable {date_column} values")

    return pd.concat(rows, ignore_index=True), pd.DataFrame(windows)


def latest_per_alpha(df):
    sort_cols = ["alpha_id", "round_date"]
    if "round" in df.columns:
        sort_cols.append("round")
    return df.sort_values(sort_cols).groupby("alpha_id", as_index=False).tail(1).reset_index(drop=True)


def build_alpha_summary(recent_df, trade_component_df):
    rows = []
    for alpha_id, alpha_df in recent_df.groupby("alpha_id", sort=True):
        alpha_df = alpha_df.copy()
        component_alpha_df = trade_component_df[trade_component_df["alpha_id"].astype(str) == str(alpha_id)] if not trade_component_df.empty else pd.DataFrame()
        if not component_alpha_df.empty and "trade_enabled" in component_alpha_df.columns:
            train_test_pass_count = int(component_alpha_df["trade_enabled"].map(to_bool).sum())
        else:
            train_test_pass_count = 0
        total_rounds = int(len(alpha_df))
        trade_sr = pd.to_numeric(alpha_df.get("TRADE_SR", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "alpha_id": alpha_id,
                "start_date": alpha_df["train_start"].min() if "train_start" in alpha_df.columns else pd.NaT,
                "end_date": alpha_df["trade_end"].max() if "trade_end" in alpha_df.columns else pd.NaT,
                "total_wfa_round": total_rounds,
                "train_test_round_pass": train_test_pass_count,
                "train_test_round_pass_rate": train_test_pass_count / total_rounds if total_rounds else np.nan,
                "max_trade_sr": trade_sr.max(),
                "min_trade_sr": trade_sr.min(),
                "median_trade_sr": trade_sr.median(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["train_test_round_pass_rate", "median_trade_sr"],
        ascending=[False, False],
    )


def build_unique_param_summary(recent_df):
    metric_cols = [col for col in ["TRAIN_SR", "TEST_SR", "TRADE_SR", "FULL_SR", "TRADE_MDD", "FULL_MDD"] if col in recent_df.columns]
    grouped = (
        recent_df.groupby(["alpha_id", "param_key"], dropna=False)
        .agg(
            first_round_date=("round_date", "min"),
            last_round_date=("round_date", "max"),
            rounds=("round_date", "count"),
            trade_pass_rate=("TRADE_PASS", "mean"),
            train_test_pass_rate=("TRAIN_TEST_PASS", "mean"),
            selection_modes=("SELECTION_MODE", lambda values: ", ".join(sorted({str(v) for v in values if pd.notna(v)}))),
            **{f"avg_{col}": (col, "mean") for col in metric_cols},
        )
        .reset_index()
    )
    return grouped.sort_values(["trade_pass_rate", "avg_TRADE_SR", "rounds"], ascending=[False, False, False])


def topic_to_filename(topic):
    topic = topic.replace("|", "_").replace("/", "_").replace("-", "_")
    topic = topic.replace("?", "_").replace("&", "_").replace("=", "_")
    topic = "".join(char for char in topic if char.isalnum() or char == "_")
    return topic + ".csv"


def parse_start_time_column(values):
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.notna().all():
        parsed = pd.to_datetime(numeric_values, unit="ms", utc=True)
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.dt.tz_localize(None)


def load_price_df(trade_asset, timeframe, price_delay):
    exchange = ASSET_PRICE_SOURCE.get(trade_asset)
    if not exchange:
        raise ValueError(f"No price source for asset: {trade_asset}")

    symbol = f"{trade_asset}USDT"
    filepath = os.path.join(PRICE_FOLDER, f"{exchange}_1m_{symbol}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"price file not found: {filepath}")

    df = pd.read_csv(filepath)
    df["time"] = parse_start_time_column(df["start_time"])
    df = df.set_index("time")[["close"]].astype(float)
    if price_delay != 0:
        df["close"] = df["close"].shift(price_delay)

    resolution = TIMEFRAME_MAP.get(timeframe)
    if not resolution:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return df.resample(resolution).last()


def load_csv(topic, column_name, datasource_key):
    file_path = DATA_FOLDER / topic_to_filename(topic)
    if not file_path.exists():
        raise FileNotFoundError(f"data CSV not found: {file_path}")

    df = pd.read_csv(file_path)
    df["time"] = parse_start_time_column(df["start_time"])
    skip_cols = {"start_time", "datetime", "time"}
    data_cols = [col for col in df.columns if col not in skip_cols]

    if column_name in data_cols:
        return df[["time", column_name]].rename(columns={column_name: datasource_key})

    if "o" in data_cols:
        def get_key(value):
            if pd.isnull(value):
                return None
            try:
                parsed = ast.literal_eval(value)
                return parsed.get(column_name, None)
            except Exception:
                return None

        df[datasource_key] = df["o"].apply(get_key)
        return df[["time", datasource_key]]

    if len(data_cols) == 1:
        actual_col = data_cols[0]
        return df[["time", actual_col]].rename(columns={actual_col: datasource_key})

    partial = [col for col in data_cols if column_name in col or col in column_name]
    if partial:
        actual_col = partial[0]
        return df[["time", actual_col]].rename(columns={actual_col: datasource_key})

    raise ValueError(f"column '{column_name}' not found in {file_path.name}")


def apply_diff_from_formula(df, alpha_formula):
    return transformation_lib.apply_diff_from_formula(df, alpha_formula)

def compute_alpha_signal(df, formula):
    return transformation_lib.compute_alpha_signal(df, formula)


def calculate_zscore(df, input_column, window):
    df["mean"] = df[input_column].rolling(window=window, min_periods=window).mean()
    df["std"] = df[input_column].rolling(window=window, min_periods=window).std()
    df["zscore"] = np.where(
        (df[input_column].rolling(window=window, min_periods=window).count() >= window) & (df["std"] != 0),
        (df[input_column] - df["mean"]) / df["std"],
        np.nan,
    )
    return df


def calculate_meannorm(df, input_column, window):
    df["mean"] = df[input_column].rolling(window=window, min_periods=window).mean()
    df["max"] = df[input_column].rolling(window=window, min_periods=window).max()
    df["min"] = df[input_column].rolling(window=window, min_periods=window).min()
    df["mean_norm"] = np.where(
        df[input_column].rolling(window=window, min_periods=window).count() >= window,
        (df[input_column] - df["mean"]) / (df["max"] - df["min"]),
        np.nan,
    )
    return df


def choose_model(df, window, model):
    if window < 0:
        raise ValueError("window size must be >= 0")

    if model == "zscore":
        df = calculate_zscore(df, "factor", window)
        df["signal"] = df["zscore"]
    elif model == "ezscorev1":
        df["ema"] = df["factor"].ewm(span=window, adjust=False, min_periods=window).mean()
        df["std"] = df["factor"].ewm(span=window, adjust=False, min_periods=window).std()
        df["signal"] = (df["factor"] - df["ema"]) / df["std"]
    elif model == "kurtosis_zscore":
        df["kurtosis"] = df["factor"].rolling(window=window, min_periods=window).kurt()
        df = calculate_zscore(df, "kurtosis", window)
        df["signal"] = df["zscore"]
    elif model == "percentilerank_meannorm":
        df["percentile_rank"] = df["factor"].rolling(window=window).rank(pct=True) * 2 - 1
        df = calculate_meannorm(df, "percentile_rank", window)
        df["signal"] = df["mean_norm"]
    elif model == "quantile_zscore":
        df["quantile"] = df["factor"].rolling(window=window).quantile(q=0.5)
        df = calculate_zscore(df, "quantile", window)
        df["signal"] = df["zscore"]
    elif model == "roc_zscore":
        df["roc"] = df["factor"].pct_change(periods=window) * 100
        df = calculate_zscore(df, "roc", window)
        df["signal"] = df["zscore"]
    elif model == "cciv2_zscore":
        df["sma"] = df["factor"].rolling(window, min_periods=window).mean()
        df["mad"] = (df["factor"] - df["sma"]).abs().rolling(window, min_periods=window).mean()
        df["cciv2"] = (df["factor"] - df["sma"]) / (0.015 * df["mad"].replace(0, np.nan))
        df = calculate_zscore(df, "cciv2", window)
        df["signal"] = df["zscore"]
    else:
        raise ValueError(f"unsupported local model for portfolio report: {model}")

    return df


def run_backtest(df, window, entry_threshold, exit_threshold, logic, side, model):
    df = choose_model(df.copy(), window, model)
    df = df.dropna().copy()
    df = entry_exit_logic_lib.signal_logic_2(df, entry_threshold, exit_threshold, logic, side)
    return df


def calculate_component_metrics(df, window, annualizer, fees):
    df = df.copy()
    df["chg"] = df["close"].pct_change()
    df["pos_t-1"] = df["pos"].shift(1).fillna(0)
    df["trades"] = (df["pos_t-1"] - df["pos"]).abs()
    df["pnl"] = df["pos_t-1"] * df["chg"] - df["trades"] * fees
    df["pnl"] = df["pnl"].fillna(0)
    df["cumu"] = df["pnl"].cumsum()
    df["dd"] = df["cumu"] - df["cumu"].cummax()
    std_pnl = df["pnl"].std()
    sr = round(df["pnl"].mean() / std_pnl * np.sqrt(annualizer), 4) if pd.notna(std_pnl) and std_pnl != 0 else np.nan
    mdd = round(df["dd"].min(), 4)
    ar = round(df["pnl"].mean() * annualizer, 4)
    cr = round(ar / abs(mdd), 4) if mdd != 0 else np.nan
    tr = round(df["cumu"].iloc[-1], 4)
    num_trades = int(df["trades"].sum())
    tpi = round(num_trades / (len(df) - window) * 100, 4) if len(df) > window else np.nan
    return {"SR": sr, "CR": cr, "MDD": mdd, "AR": ar, "TR": tr, "TPI": tpi, "num_trades": num_trades}, df


def load_alpha_map():
    with open(JSON_FILE, "r", encoding="utf-8") as file:
        alphas = json.load(file)
    return {alpha.get("alpha_id"): alpha for alpha in alphas if alpha.get("alpha_id")}


ALPHA_CONFIG_COLUMNS = [
    "alpha_id",
    "custom_id",
    "alpha_formula",
    "manual_alpha_formula",
    "data_preprocessing",
    "preprocessing_window",
    "data_asset",
    "trade_asset",
    "shift_backtest_candle_minute",
    "fees",
    "model",
    "entry_exit_logic",
    "timeframe",
]


def build_alpha_config_summary(alpha_ids):
    alpha_map = load_alpha_map()
    rows = []
    for alpha_id in sorted({str(item) for item in alpha_ids if pd.notna(item)}):
        alpha = alpha_map.get(alpha_id, {})
        row = {col: alpha.get(col, "") for col in ALPHA_CONFIG_COLUMNS}
        row["alpha_id"] = alpha_id
        rows.append(row)
    return pd.DataFrame(rows, columns=ALPHA_CONFIG_COLUMNS)


def build_alpha_config_vertical(alpha_ids):
    config_df = build_alpha_config_summary(alpha_ids)
    rows = []
    for _, alpha in config_df.iterrows():
        for field in ALPHA_CONFIG_COLUMNS:
            rows.append(
                {
                    "alpha_id": alpha["alpha_id"],
                    "field": field,
                    "value": alpha.get(field, ""),
                }
            )
    return pd.DataFrame(rows, columns=["alpha_id", "field", "value"])


def parse_datasource_structure(alpha):
    raw = alpha.get("datasource_structure", "{}")
    try:
        return json.loads(raw.replace("'", '"'))
    except Exception:
        import ast

        return ast.literal_eval(raw)


def prepare_alpha_base(alpha, price_cache):
    datasource_structure = parse_datasource_structure(alpha)
    merged_df = None

    for key, value in datasource_structure.items():
        topic = value.get("topic")
        column = value.get("column")
        src_df = load_csv(topic, column, key)
        src_df = src_df.dropna(subset=[key])
        merged_df = src_df if merged_df is None else pd.merge(merged_df, src_df, on="time", how="inner")

    if merged_df is None or merged_df.empty:
        raise ValueError("no datasource rows loaded")

    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if "start_time" in col])
    merged_df = merged_df.set_index("time")

    alpha_formula = alpha.get("manual_alpha_formula") or alpha.get("alpha_formula", "")
    merged_df, rewritten = apply_diff_from_formula(merged_df, alpha_formula)
    for col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    signal = compute_alpha_signal(merged_df, rewritten)

    merged_df["factor"] = signal
    merged_df["factor"] = merged_df["factor"].replace([np.inf, -np.inf], np.nan)
    merged_df["factor"] = (
        merged_df["factor"].round(10)
        if abs(merged_df["factor"].min()) < 1.0
        else merged_df["factor"].round(10)
    )
    if alpha.get("data_preprocessing") == "diff":
        merged_df["factor"] = merged_df["factor"].diff()
        merged_df = merged_df.dropna(subset=["factor"])

    factor_df = merged_df.dropna()
    if factor_df.empty:
        raise ValueError("factor dataframe is empty after preprocessing")

    timeframe = alpha.get("timeframe", "1h")
    resolution = TIMEFRAME_MAP.get(timeframe)
    if resolution is None:
        raise ValueError(f"unsupported timeframe: {timeframe}")

    annualizer = int(round(METRIC_ANNUALIZER_MAP.get(timeframe, 365 * 24)))
    fees = float(alpha.get("fees", 0.035)) / 100
    trade_asset = alpha.get("trade_asset", "BTC")
    price_delay = int(alpha.get("shift_backtest_candle_minute", 0))
    price_key = (trade_asset, timeframe, price_delay)
    if price_key not in price_cache:
        price_cache[price_key] = load_price_df(trade_asset, timeframe, price_delay)

    price_df = price_cache[price_key]
    if price_df is None or price_df.empty:
        raise ValueError("price dataframe is empty")

    return {
        "alpha_id": alpha.get("alpha_id"),
        "factor_df": factor_df,
        "price_df": price_df,
        "timeframe": timeframe,
        "resolution": resolution,
        "annualizer": annualizer,
        "fees": fees,
    }


def component_id(row):
    round_value = row.get("round", "")
    if pd.isna(round_value):
        round_text = ""
    else:
        try:
            round_number = float(round_value)
            round_text = str(int(round_number)) if round_number.is_integer() else str(round_value)
        except (TypeError, ValueError):
            round_text = str(round_value)
    return f"{row.get('alpha_id')}__r{round_text}__{pd.Timestamp(row.get('round_date')).strftime('%Y%m%d')}"


def metric_pass(value, threshold, mode):
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return False
    if mode == "gt":
        return value > threshold
    return value >= threshold


def row_trade_gate(row):
    checks = [
        ("TRAIN_SR", 1.5, "gte"),
        ("TRAIN_MDD", -0.5, "gt"),
        ("TRAIN_TPI", 3.0, "gte"),
        ("TRAIN_num_trades", 10, "gte"),
        ("TEST_SR", 1.0, "gte"),
        ("TEST_MDD", -0.5, "gt"),
        ("TEST_TPI", 3.0, "gte"),
        ("TEST_num_trades", 10, "gte"),
    ]
    failed = [metric for metric, threshold, mode in checks if not metric_pass(row.get(metric), threshold, mode)]
    return not failed, ", ".join(failed)


def blank_trade_series(row, resolution, name):
    eval_start = pd.Timestamp(row["trade_start"])
    eval_end = pd.Timestamp(row["trade_end"])
    full_range = pd.date_range(eval_start, eval_end, freq=resolution)
    if full_range.empty:
        raise ValueError("empty evaluation date range")
    return pd.Series(np.nan, index=full_range, name=name), eval_start, eval_end


def run_component_backtest(row, base):
    eval_start = pd.Timestamp(row["trade_start"])
    eval_end = pd.Timestamp(row["trade_end"])
    full_range = pd.date_range(eval_start, eval_end, freq=base["resolution"])
    if full_range.empty:
        raise ValueError("empty evaluation date range")

    signal_df = run_backtest(
        base["factor_df"].copy(),
        int(row["window"]),
        float(row["threshold_1"]),
        float(row["threshold_2"]),
        str(row["logic"]),
        str(row["side"]),
        str(row["model"]),
    )
    trade_df = signal_df.loc[eval_start:eval_end].copy()
    trade_df = trade_df.reindex(full_range).ffill()
    trade_df = trade_df.join(base["price_df"][["close"]], how="inner")
    if trade_df.empty:
        raise ValueError("empty trade dataframe after price join")

    metrics, evaluated_df = calculate_component_metrics(
        trade_df,
        int(row["window"]),
        base["annualizer"],
        base["fees"],
    )
    return metrics, evaluated_df, eval_start, eval_end


def build_portfolio_backtest(recent_df):
    alpha_map = load_alpha_map()
    base_cache = {}
    price_cache = {}
    pnl_series = []
    component_rows = []
    errors = []
    annualizers = []

    for _, row in recent_df.sort_values(["alpha_id", "round_date"]).iterrows():
        alpha_id = row.get("alpha_id")
        if alpha_id not in alpha_map:
            errors.append({"alpha_id": alpha_id, "round": row.get("round"), "error": "alpha_id not found in JSON_FILE"})
            continue

        cid = component_id(row)
        trade_enabled, skip_reason = row_trade_gate(row)
        try:
            if alpha_id not in base_cache:
                base_cache[alpha_id] = prepare_alpha_base(alpha_map[alpha_id], price_cache)
            base = base_cache[alpha_id]
            annualizers.append(base["annualizer"])
            if trade_enabled:
                metrics, evaluated_df, eval_start, eval_end = run_component_backtest(row, base)
                pnl_series.append(evaluated_df["pnl"].rename(cid))
            else:
                skipped_pnl, eval_start, eval_end = blank_trade_series(row, base["resolution"], cid)
                pnl_series.append(skipped_pnl)
                try:
                    metrics, _, _, _ = run_component_backtest(row, base)
                except Exception as metric_exc:
                    metrics = {}
                    errors.append({"alpha_id": alpha_id, "round": row.get("round"), "error": f"disabled round metrics failed: {metric_exc}"})
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
        for col in WFA_COMPONENT_COLUMNS:
            if col in row.index:
                component_row[col] = row.get(col)
        component_rows.append(component_row)

    if not pnl_series:
        empty = pd.DataFrame(columns=["pnl", "cumu", "dd", "rolling_sharpe_180", "rolling_sharpe_365"])
        return empty, pd.DataFrame(), pd.DataFrame(component_rows), pd.DataFrame(errors), {}

    pnl_wide = pd.concat(pnl_series, axis=1, sort=True).sort_index()
    active_count = pnl_wide.notna().sum(axis=1)
    weighted_pnl = pnl_wide.div(active_count.replace(0, np.nan), axis=0)
    portfolio_pnl = weighted_pnl.sum(axis=1, min_count=1)
    portfolio_pnl = portfolio_pnl.where(active_count > 0)
    component_equity_df = weighted_pnl.fillna(0.0).cumsum()
    annualizer = int(pd.Series(annualizers).mode().iloc[0]) if annualizers else 365 * 24
    portfolio_df = calculate_portfolio_timeseries(portfolio_pnl, annualizer)

    component_df = pd.DataFrame(component_rows)
    actual_contribution = weighted_pnl.sum(axis=0, min_count=1)
    component_df["portfolio_TR_contribution"] = component_df["component_id"].map(actual_contribution)
    metrics_pnl = portfolio_df["pnl_metric"] if "pnl_metric" in portfolio_df.columns else portfolio_df["pnl"]
    metrics = calculate_pnl_metrics(metrics_pnl, annualizer)
    metrics["skipped_components"] = int((~component_df.get("trade_enabled", pd.Series(dtype=bool))).sum()) if not component_df.empty else 0
    metrics["annualizer"] = annualizer
    return portfolio_df, component_equity_df, component_df, pd.DataFrame(errors), metrics


def calculate_portfolio_timeseries(pnl, annualizer):
    df = pd.DataFrame({"pnl": pnl}).sort_index()
    if not df.empty:
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is None and len(df.index) > 1:
            gaps = df.index.to_series().diff().dropna()
            inferred_freq = gaps.median() if not gaps.empty else None
        if inferred_freq is not None:
            full_index = pd.date_range(df.index.min(), df.index.max(), freq=inferred_freq)
            df = df.reindex(full_index)
            df.index.name = "time"

    df["pnl_metric"] = df["pnl"]
    df["pnl"] = df["pnl"].fillna(0.0)
    df["cumu"] = df["pnl"].cumsum()
    df["dd"] = df["cumu"] - df["cumu"].cummax()
    equity_pnl = df["cumu"].diff()
    if not equity_pnl.empty:
        equity_pnl.iloc[0] = df["cumu"].iloc[0]
    bars_per_day = max(1, int(round(annualizer / 365)))
    roll_180 = max(2, 180 * bars_per_day)
    roll_365 = max(2, annualizer)
    for col, window in [("rolling_sharpe_180", roll_180), ("rolling_sharpe_365", roll_365)]:
        roll = equity_pnl.rolling(window, min_periods=window)
        roll_mean = roll.mean()
        roll_std = roll.std()
        roll_count = equity_pnl.rolling(window, min_periods=1).count()
        df[col] = np.where(roll_std > 0, roll_mean / roll_std * np.sqrt(annualizer), 0.0)
        df.loc[roll_count < window, col] = np.nan
    return df


def calculate_pnl_metrics(pnl, annualizer):
    pnl = pnl.dropna()
    if pnl.empty:
        return {
            "SR": np.nan,
            "CR": np.nan,
            "MDD": np.nan,
            "AR": np.nan,
            "TR": np.nan,
            "VAR": np.nan,
            "bars": 0,
            "nonzero_bars": 0,
        }
    cumu = pnl.cumsum()
    dd = cumu - cumu.cummax()
    std_pnl = pnl.std()
    sr = round(pnl.mean() / std_pnl * np.sqrt(annualizer), 4) if pd.notna(std_pnl) and std_pnl != 0 else np.nan
    mdd = round(dd.min(), 4) if not dd.empty else np.nan
    ar = round(pnl.mean() * annualizer, 4)
    cr = round(ar / abs(mdd), 4) if pd.notna(mdd) and mdd != 0 else np.nan
    var = round(pnl.quantile(0.05) * 100, 4) if not pnl.empty else np.nan
    return {
        "SR": sr,
        "CR": cr,
        "MDD": mdd,
        "AR": ar,
        "TR": round(cumu.iloc[-1], 4) if not cumu.empty else np.nan,
        "VAR": var,
        "bars": int(len(pnl)),
        "nonzero_bars": int((pnl != 0).sum()),
    }


def calculate_portfolio_yearly_metrics(portfolio_df, annualizer):
    rows = []
    if portfolio_df.empty:
        return pd.DataFrame(columns=["Year", "SR", "CR", "MDD", "AR", "TR", "VAR", "bars", "nonzero_bars"])

    for year in sorted(portfolio_df.index.year.unique()):
        pnl_col = "pnl_metric" if "pnl_metric" in portfolio_df.columns else "pnl"
        metrics = calculate_pnl_metrics(portfolio_df.loc[portfolio_df.index.year == year, pnl_col], annualizer)
        rows.append({"Year": int(year), **metrics})
    return pd.DataFrame(rows)


def make_portfolio_dashboard(
    portfolio_df,
    component_equity_df,
    component_df,
    portfolio_metrics,
    max_component_traces,
    title_prefix="Portfolio",
):
    if go is None or make_subplots is None or portfolio_df.empty:
        return ""

    fig = make_subplots(
        rows=5,
        cols=1,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{title_prefix} Equity",
            "Drawdown",
            f"{title_prefix} Component Equity Curves",
            "Roll-180-Day SR",
            "Roll-365-Day SR",
        ),
        row_heights=[0.32, 0.16, 0.20, 0.16, 0.16],
    )
    fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df["cumu"], name=f"{title_prefix} Cumulative PnL", line=dict(color="#1565c0")), row=1, col=1)
    fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df["dd"], fill="tozeroy", name="Drawdown", line=dict(color="#c62828")), row=2, col=1)

    component_order = component_df.sort_values("portfolio_TR_contribution", ascending=False)["component_id"].head(max_component_traces).tolist()
    for cid in component_order:
        if cid not in component_equity_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=component_equity_df.index,
                y=component_equity_df[cid],
                mode="lines",
                name=str(cid),
                opacity=0.55,
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df["rolling_sharpe_180"], name="Roll-180d SR", line=dict(color="#6a1b9a")), row=4, col=1)
    fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df["rolling_sharpe_365"], name="Roll-365d SR", line=dict(color="#2e7d32")), row=5, col=1)

    for row in [4, 5]:
        for y0, color in [(2, "#2e7d32"), (0, "#111")]:
            fig.add_shape(type="line", x0=portfolio_df.index[0], x1=portfolio_df.index[-1], y0=y0, y1=y0, line=dict(color=color, dash="dash"), row=row, col=1)

    title = (
        f"{title_prefix} Dashboard | SR={portfolio_metrics.get('SR', np.nan)} "
        f"MDD={portfolio_metrics.get('MDD', np.nan)} TR={portfolio_metrics.get('TR', np.nan)}"
    )
    fig.update_layout(height=1550, showlegend=True, title_text=title, title_x=0.5, template="plotly_white")
    fig.update_yaxes(title_text="Cumulative PnL", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def make_component_metric_chart(component_df):
    if go is None or component_df.empty:
        return ""

    plot_df = component_df.sort_values("portfolio_TR_contribution", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=plot_df["component_id"], y=plot_df["portfolio_TR_contribution"], name="Portfolio TR Contribution"))
    fig.add_trace(go.Scatter(x=plot_df["component_id"], y=plot_df["SR"], mode="markers", yaxis="y2", name="Component SR"))
    fig.update_layout(
        title="Component Contribution and SR",
        height=520,
        template="plotly_white",
        xaxis=dict(title="Component", tickangle=45),
        yaxis=dict(title="Portfolio TR Contribution"),
        yaxis2=dict(title="Component SR", overlaying="y", side="right"),
        margin=dict(l=50, r=60, t=60, b=180),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def make_trade_sr_chart(recent_df):
    if go is None or recent_df.empty or "TRADE_SR" not in recent_df.columns:
        return ""

    fig = go.Figure()
    for alpha_id, alpha_df in recent_df.sort_values("round_date").groupby("alpha_id", sort=True):
        fig.add_trace(
            go.Scatter(
                x=alpha_df["round_date"],
                y=alpha_df["TRADE_SR"],
                mode="lines+markers",
                name=str(alpha_id),
                hovertemplate=(
                    "alpha=%{text}<br>round=%{x|%Y-%m-%d}<br>"
                    "TRADE_SR=%{y:.3f}<extra></extra>"
                ),
                text=[alpha_id] * len(alpha_df),
            )
        )

    fig.update_layout(
        title="Recent Trade SR by Round",
        height=520,
        margin=dict(l=50, r=30, t=60, b=50),
        xaxis_title="Round Date",
        yaxis_title="TRADE_SR",
        template="plotly_white",
        legend_title="Alpha",
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def make_latest_full_chart(latest_df):
    if go is None or latest_df.empty:
        return ""

    plot_df = latest_df.sort_values("FULL_SR", ascending=False) if "FULL_SR" in latest_df.columns else latest_df
    fig = go.Figure()
    if "FULL_SR" in plot_df.columns:
        fig.add_trace(go.Bar(x=plot_df["alpha_id"], y=plot_df["FULL_SR"], name="FULL_SR"))
    if "FULL_MDD" in plot_df.columns:
        fig.add_trace(go.Bar(x=plot_df["alpha_id"], y=plot_df["FULL_MDD"], name="FULL_MDD"))

    fig.update_layout(
        title="Latest Round CSV FULL Metrics",
        height=480,
        barmode="group",
        margin=dict(l=50, r=30, t=60, b=120),
        xaxis_title="Alpha",
        yaxis_title="Metric",
        template="plotly_white",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def make_test_trade_scatter(recent_df):
    if go is None or recent_df.empty or not {"TEST_SR", "TRADE_SR"}.issubset(recent_df.columns):
        return ""

    colors = recent_df["TRADE_PASS"].map({True: "#2e7d32", False: "#c62828"}).fillna("#777")
    hover = recent_df.apply(
        lambda row: (
            f"alpha={row.get('alpha_id')}<br>"
            f"round={pd.Timestamp(row.get('round_date')).date()}<br>"
            f"{html.escape(str(row.get('param_key', '')))}"
        ),
        axis=1,
    )
    fig = go.Figure(
        go.Scatter(
            x=recent_df["TEST_SR"],
            y=recent_df["TRADE_SR"],
            mode="markers",
            marker=dict(size=9, color=colors, opacity=0.78, line=dict(width=0.5, color="#333")),
            text=hover,
            hovertemplate="%{text}<br>TEST_SR=%{x:.3f}<br>TRADE_SR=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="TEST_SR vs TRADE_SR",
        height=480,
        margin=dict(l=50, r=30, t=60, b=50),
        xaxis_title="TEST_SR",
        yaxis_title="TRADE_SR",
        template="plotly_white",
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="#777")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def format_value(value, precision=4):
    if pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (float, np.floating)):
        return f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return value


def decorate_pass(value):
    passed = to_bool(value)
    label = "PASS" if passed else "FAIL"
    cls = "pass" if passed else "fail"
    return f'<span class="{cls}">{label}</span>'


def decorate_true_false(value):
    passed = to_bool(value)
    label = "True" if passed else "False"
    cls = "pass" if passed else "fail"
    return f'<span class="{cls}">{label}</span>'


def render_table(df, classes="metrics-table", max_rows=None):
    if df.empty:
        return "<p>No rows.</p>"

    display = df.copy()
    if max_rows is not None:
        display = display.head(max_rows)

    for col in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[col]):
            display[col] = display[col].dt.strftime("%Y-%m-%d")
        elif col == "trade_enabled":
            display[col] = display[col].apply(decorate_true_false)
        elif col in PASS_COLUMNS or col.endswith("_PASS"):
            display[col] = display[col].apply(decorate_pass)
        elif pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].apply(format_value)

    return display.to_html(index=False, classes=classes, escape=False)


HEALTH_DASHBOARD_COLUMNS = [
    ("total_wfa_round", "Total WFA Round"),
    ("train_test_round_pass", "Train/Test Round Pass"),
    ("train_test_round_pass_rate", "Train/Test Pass Rate"),
    ("start_date", "Start Date"),
    ("end_date", "End Date"),
    ("max_trade_sr", "Max Trade SR"),
    ("min_trade_sr", "Min Trade SR"),
    ("median_trade_sr", "Median Trade SR"),
]


def health_value_class(metric, value):
    if metric.endswith("_PASS"):
        return "good" if to_bool(value) else "bad"
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "muted"
    if "MDD" in metric:
        return "good" if numeric > -0.5 else "bad"
    if "pass_rate" in metric:
        if numeric >= 0.75:
            return "good"
        if numeric >= 0.5:
            return "warn"
        return "bad"
    if "sr" in metric.lower():
        return "good" if numeric > 0 else "bad"
    return "neutral"


def render_alpha_health_dashboard(alpha_summary_df):
    if alpha_summary_df.empty:
        return "<p>No rows.</p>"

    panels = []
    for _, row in alpha_summary_df.iterrows():
        cards = []
        for metric, label in HEALTH_DASHBOARD_COLUMNS:
            value = row.get(metric, np.nan)
            if metric.endswith("_PASS"):
                formatted = "PASS" if to_bool(value) else "FAIL"
            else:
                formatted = format_value(value)
            cls = health_value_class(metric, value)
            cards.append(
                f"""
                <div class="stat-card {cls}">
                  <div class="stat-label">{html.escape(label)}</div>
                  <div class="stat-value">{html.escape(str(formatted))}</div>
                </div>
                """
            )

        subtitle = "train/test pass rounds shown against all WFA rounds"
        panels.append(
            f"""
            <div class="health-alpha">
              <div class="health-title">{html.escape(str(row.get('alpha_id', '')))}</div>
              <div class="health-subtitle">{html.escape(subtitle)}</div>
              <div class="stat-grid">{''.join(cards)}</div>
            </div>
            """
        )

    return f'<div class="health-dashboard">{"".join(panels)}</div>'


def select_columns(df, columns):
    return df[[col for col in columns if col in df.columns]].copy()


def build_html_report(
    recent_df,
    latest_df,
    alpha_summary_df,
    trade_portfolio_df,
    trade_component_equity_df,
    trade_component_df,
    trade_portfolio_metrics,
    trade_portfolio_yearly_df,
    trade_error_df,
    input_dir,
    output_csv_name,
    date_column,
    window,
    max_component_traces,
):
    alpha_count = recent_df["alpha_id"].nunique()
    row_count = len(recent_df)
    latest_min = recent_df["round_date"].min()
    latest_max = recent_df["round_date"].max()

    charts_html = "\n".join(
        chart
        for chart in [
            make_portfolio_dashboard(
                trade_portfolio_df,
                trade_component_equity_df,
                trade_component_df,
                trade_portfolio_metrics,
                max_component_traces,
                title_prefix="Stitched WF Trade Portfolio",
            ),
            make_component_metric_chart(trade_component_df),
            make_test_trade_scatter(recent_df),
        ]
        if chart
    )
    if not charts_html:
        charts_html = '<div class="panel"><p>Plotly is not installed, so this report was generated with tables only.</p></div>'

    metric_keys = [
        "SR",
        "CR",
        "MDD",
        "AR",
        "TR",
        "VAR",
        "bars",
        "nonzero_bars",
        "skipped_components",
        "annualizer",
    ]
    trade_portfolio_metrics_html = render_table(
        pd.DataFrame(
            {
                "Metric": metric_keys,
                "Stitched WF Trade Portfolio": [trade_portfolio_metrics.get(key, "") for key in metric_keys],
            }
        ),
        classes="metrics-table",
    )
    alpha_config_html = render_table(
        build_alpha_config_vertical(recent_df["alpha_id"]),
        classes="metrics-table vertical-table",
    )
    trade_portfolio_yearly_html = render_table(trade_portfolio_yearly_df, classes="metrics-table")
    alpha_summary_html = render_alpha_health_dashboard(alpha_summary_df)
    trade_component_html = render_table(
        select_columns(
            trade_component_df.sort_values(["eval_start", "alpha_id", "round"]) if not trade_component_df.empty else trade_component_df,
            [
                "component_id",
                "alpha_id",
                "trade_enabled",
                "round",
                "eval_start",
                "eval_end",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "trade_start",
                "trade_end",
                "timeframe",
                "model",
                "logic",
                "side",
                "window",
                "threshold_1",
                "threshold_2",
                "trial",
                "objective",
                "skip_reason",
                "TRAIN_SR",
                "TRAIN_CR",
                "TRAIN_MDD",
                "TRAIN_AR",
                "TRAIN_TR",
                "TRAIN_TPI",
                "TRAIN_num_trades",
                "TRAIN_PASS",
                "TEST_SR",
                "TEST_CR",
                "TEST_MDD",
                "TEST_AR",
                "TEST_TR",
                "TEST_TPI",
                "TEST_num_trades",
                "TEST_PASS",
                "TRAIN_TEST_PASS",
                "TRADE_SR",
                "TRADE_CR",
                "TRADE_MDD",
                "TRADE_AR",
                "TRADE_TR",
                "TRADE_TPI",
                "TRADE_num_trades",
                "TRADE_PASS",
                "TRADE_SR_POSITIVE",
                "FILTER_PASS",
                "FILTER_SELECT_PASS",
                "PARAM_SELECTED",
                "WFA_SELECTED_PASS",
                "SELECTION_MODE",
                "SR",
                "CR",
                "MDD",
                "AR",
                "TR",
                "portfolio_TR_contribution",
                "TPI",
                "num_trades",
            ],
        ),
        classes="metrics-table wide-table",
    )
    trade_errors_html = render_table(trade_error_df, classes="metrics-table") if not trade_error_df.empty else "<p>No stitched trade component errors.</p>"
    source_files_df = (
        recent_df.groupby("source_file", dropna=False)
        .agg(alphas=("alpha_id", "nunique"), rows=("alpha_id", "size"), first_trade_start=("trade_start", "min"), last_trade_end=("trade_end", "max"))
        .reset_index()
        .sort_values("source_file")
    )
    source_files_html = render_table(source_files_df, classes="metrics-table wide-table")

    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    period_label = f"{format_value(latest_min)} to {format_value(latest_max)}"
    method_note = (
        "Each selected round is checked against the TRAIN and TEST pass requirements before trading. "
        "Passing rounds are recomputed only for their trade_start to trade_end period; failing rounds are included as zero-PnL periods in the final equity curve and rolling SR."
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Walk-Forward Round Best Portfolio</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f4f6f8; color: #202124; }}
    .dashboard {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(420px, 0.75fr); gap: 18px; height: 100vh; padding: 18px; box-sizing: border-box; }}
    .plots {{ overflow-y: auto; }}
    .tables {{ overflow-y: auto; }}
    .header {{ background: white; border: 1px solid #dfe5ec; padding: 18px 20px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
    h2 {{ margin: 0 0 12px 0; font-size: 18px; color: #243447; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    .pill {{ background: #eef3f7; border: 1px solid #d7e1ea; padding: 6px 10px; font-size: 13px; }}
    .panel {{ background: white; border: 1px solid #dfe5ec; padding: 16px; margin-bottom: 18px; overflow-x: auto; }}
    .metrics-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .metrics-table th, .metrics-table td {{ border: 1px solid #d9e2ea; padding: 7px 8px; text-align: center; vertical-align: top; }}
    .metrics-table th {{ background: #315b7d; color: white; position: sticky; top: 0; z-index: 1; }}
    .metrics-table tr:nth-child(even) {{ background: #f8fafc; }}
    .wide-table td {{ min-width: 72px; }}
    .wide-table td:last-child {{ text-align: left; min-width: 420px; }}
    .vertical-table td {{ text-align: left; }}
    .vertical-table td:nth-child(2) {{ min-width: 170px; font-weight: 700; color: #243447; }}
    .vertical-table td:nth-child(3) {{ min-width: 360px; max-width: 680px; white-space: normal; word-break: break-word; }}
    .pass {{ color: #2e7d32; font-weight: 700; }}
    .fail {{ color: #c62828; font-weight: 700; }}
    .health-dashboard {{ display: grid; gap: 14px; }}
    .health-alpha {{ background: #111217; border: 1px solid #2b2f3a; padding: 14px; color: #d8d9da; }}
    .health-title {{ font-size: 14px; font-weight: 700; color: #f2f4f8; margin-bottom: 4px; }}
    .health-subtitle {{ color: #9aa4b2; font-size: 12px; margin-bottom: 12px; }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 10px; }}
    .stat-card {{ background: #1f2430; border: 1px solid #343b49; padding: 11px 12px; min-height: 70px; }}
    .stat-card.good {{ border-left: 4px solid #73bf69; }}
    .stat-card.warn {{ border-left: 4px solid #f2cc0c; }}
    .stat-card.bad {{ border-left: 4px solid #f2495c; }}
    .stat-card.muted {{ border-left: 4px solid #6b7280; }}
    .stat-card.neutral {{ border-left: 4px solid #5794f2; }}
    .stat-label {{ color: #a7b1c2; font-size: 11px; line-height: 1.25; min-height: 28px; }}
    .stat-value {{ color: #f2f4f8; font-size: 21px; font-weight: 700; line-height: 1.2; margin-top: 6px; }}
    .stat-card.good .stat-value {{ color: #73bf69; }}
    .stat-card.warn .stat-value {{ color: #f2cc0c; }}
    .stat-card.bad .stat-value {{ color: #f2495c; }}
    .note {{ color: #5f6b7a; font-size: 13px; line-height: 1.45; }}
    a {{ color: #1d5f8f; }}
  </style>
</head>
<body>
  <div class="dashboard">
    <div class="plots">
      <div class="header">
        <h1>Walk-Forward Round Best Portfolio</h1>
        <div class="note">
          Built from generated <code>*_wf_round_best_summary.csv</code> files. {method_note}
        </div>
        <div class="meta">
          <div class="pill">alphas: {alpha_count}</div>
          <div class="pill">selected rows: {row_count}</div>
          <div class="pill">round date: {html.escape(date_column)}</div>
          <div class="pill">window: {html.escape(window)}</div>
          <div class="pill">selected rounds: {period_label}</div>
          <div class="pill">portfolio: stitched trade windows</div>
          <div class="pill">input: {html.escape(str(input_dir))}</div>
          <div class="pill">selected csv: {html.escape(output_csv_name)}</div>
          <div class="pill">generated: {generated_at}</div>
        </div>
      </div>
      <div class="panel">
        <h2>Backtest Plots</h2>
        {charts_html}
      </div>
    </div>

    <div class="tables">
      <div class="panel">
        <h2>Stitched WF Trade Portfolio Metrics</h2>
        {trade_portfolio_metrics_html}
        <h2>Alpha Config</h2>
        {alpha_config_html}
      </div>

      <div class="panel">
        <h2>Stitched WF Trade Yearly Metrics</h2>
        {trade_portfolio_yearly_html}
      </div>

      <div class="panel">
        <h2>Alpha Health Summary</h2>
        {alpha_summary_html}
      </div>

      <div class="panel">
        <h2>Stitched WF Trade Components</h2>
        {trade_component_html}
      </div>

      <div class="panel">
        <h2>Stitched WF Trade Component Errors</h2>
        {trade_errors_html}
      </div>

      <div class="panel">
        <h2>Source Summary CSV Files</h2>
        {source_files_html}
      </div>
    </div>
  </div>
</body>
</html>"""


def write_report(args, input_dir, output_dir, files, alpha_filter, output_name=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df = add_param_key(load_all_summaries(files, alpha_filter=alpha_filter))
    recent_df, windows_df = filter_recent_rounds(
        all_df,
        args.date_column,
        args.window,
        exclude_single_day_trade=args.exclude_single_day_trade,
    )
    recent_df = add_param_key(recent_df)
    latest_df = latest_per_alpha(recent_df)
    trade_portfolio_df, trade_component_equity_df, trade_component_df, trade_error_df, trade_portfolio_metrics = build_portfolio_backtest(
        recent_df
    )
    alpha_summary_df = build_alpha_summary(recent_df, trade_component_df)
    trade_portfolio_yearly_df = calculate_portfolio_yearly_metrics(
        trade_portfolio_df,
        int(trade_portfolio_metrics.get("annualizer", 365 * 24)),
    )

    selected_csv_path = output_dir / "wf_round_best_portfolio_selected.csv"
    recent_df.to_csv(selected_csv_path, index=False)

    latest_csv_path = output_dir / "wf_round_best_portfolio_latest.csv"
    latest_df.to_csv(latest_csv_path, index=False)

    portfolio_csv_path = output_dir / "wf_round_best_portfolio_timeseries.csv"
    trade_portfolio_df.to_csv(portfolio_csv_path)

    component_csv_path = output_dir / "wf_round_best_portfolio_components.csv"
    trade_component_df.to_csv(component_csv_path, index=False)

    errors_csv_path = output_dir / "wf_round_best_portfolio_errors.csv"
    trade_error_df.to_csv(errors_csv_path, index=False)

    trade_portfolio_csv_path = output_dir / "wf_round_best_trade_portfolio_timeseries.csv"
    trade_portfolio_df.to_csv(trade_portfolio_csv_path)

    trade_component_csv_path = output_dir / "wf_round_best_trade_portfolio_components.csv"
    trade_component_df.to_csv(trade_component_csv_path, index=False)

    trade_errors_csv_path = output_dir / "wf_round_best_trade_portfolio_errors.csv"
    trade_error_df.to_csv(trade_errors_csv_path, index=False)

    html_report = build_html_report(
        recent_df=recent_df,
        latest_df=latest_df,
        alpha_summary_df=alpha_summary_df,
        trade_portfolio_df=trade_portfolio_df,
        trade_component_equity_df=trade_component_equity_df,
        trade_component_df=trade_component_df,
        trade_portfolio_metrics=trade_portfolio_metrics,
        trade_portfolio_yearly_df=trade_portfolio_yearly_df,
        trade_error_df=trade_error_df,
        input_dir=input_dir,
        output_csv_name=selected_csv_path.name,
        date_column=args.date_column,
        window=args.window,
        max_component_traces=args.max_component_traces,
    )
    html_path = output_dir / (output_name or args.output_name)
    html_path.write_text(html_report, encoding="utf-8")

    print(f"input csv files: {len(files)}")
    print(f"alphas: {recent_df['alpha_id'].nunique()}")
    print(f"selected rows: {len(recent_df)}")
    print("selected source files:")
    for source_file in sorted(recent_df["source_file"].dropna().astype(str).unique()):
        print(f"  {source_file}")
    print("portfolio mode: stitched trade windows")
    print(f"component backtests: {len(trade_component_df)}")
    print(f"skipped trade windows: {trade_portfolio_metrics.get('skipped_components')}")
    print(f"component errors: {len(trade_error_df)}")
    print(f"stitched trade portfolio SR: {trade_portfolio_metrics.get('SR')}")
    print(f"stitched trade portfolio MDD: {trade_portfolio_metrics.get('MDD')}")
    print(f"selected rows saved: {selected_csv_path}")
    print(f"latest portfolio saved: {latest_csv_path}")
    print(f"portfolio timeseries saved: {portfolio_csv_path}")
    print(f"component backtests saved: {component_csv_path}")
    print(f"component errors saved: {errors_csv_path}")
    print(f"stitched trade aliases saved: {trade_portfolio_csv_path}, {trade_component_csv_path}, {trade_errors_csv_path}")
    print(f"html report saved: {html_path}")
    return html_path, trade_portfolio_metrics


def discover_alpha_ids_from_summaries(files):
    frames = []
    for path in files:
        df = read_summary_csv(path)
        if not df.empty and "alpha_id" in df.columns:
            frames.append(df[["alpha_id"]])
    if not frames:
        raise ValueError("no alpha ids found in walk-forward summary files")
    alpha_ids = pd.concat(frames, ignore_index=True)["alpha_id"].dropna().astype(str).unique()
    return sorted(alpha_ids)


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    files = discover_summary_files(input_dir)
    alpha_filter = target_alpha_ids(args.target_alpha_id)

    if alpha_filter is None:
        alpha_ids = discover_alpha_ids_from_summaries(files)
        base_output_dir = (args.output_dir or input_dir).resolve()
        print(f"TARGET_ALPHA_ID=None: generating separate reports for {len(alpha_ids)} alphas")
        generated = []
        for alpha_id in alpha_ids:
            alpha_output_dir = base_output_dir / alpha_id
            output_name = f"{alpha_id}_wf_round_best_portfolio.html"
            html_path, metrics = write_report(
                args,
                input_dir,
                alpha_output_dir,
                files,
                {alpha_id},
                output_name=output_name,
            )
            generated.append(
                {
                    "alpha_id": alpha_id,
                    "html_path": html_path,
                    "SR": metrics.get("SR"),
                    "MDD": metrics.get("MDD"),
                }
            )

        print("\nindividual html reports:")
        for item in generated:
            print(f"  {item['alpha_id']}: SR={item['SR']} MDD={item['MDD']} -> {item['html_path']}")
        return

    base_output_dir = (args.output_dir or input_dir).resolve()
    generated = []
    for alpha_id in sorted(alpha_filter):
        alpha_output_dir = base_output_dir / alpha_id
        output_name = f"{alpha_id}_wf_round_best_portfolio.html"
        html_path, metrics = write_report(
            args,
            input_dir,
            alpha_output_dir,
            files,
            {alpha_id},
            output_name=output_name,
        )
        generated.append({"alpha_id": alpha_id, "html_path": html_path, "SR": metrics.get("SR"), "MDD": metrics.get("MDD")})

    print("\nindividual html reports:")
    for item in generated:
        print(f"  {item['alpha_id']}: SR={item['SR']} MDD={item['MDD']} -> {item['html_path']}")


if __name__ == "__main__":
    main()

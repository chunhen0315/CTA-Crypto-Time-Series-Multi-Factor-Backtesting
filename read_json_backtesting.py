import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import ast  # for literal_eval
import re
import numpy as np
from lib import entry_exit_logic_lib, models_lib, transformation_lib
import warnings
import os
from config import (
    JSON_FILE,
    DATA_FOLDER as CONFIG_DATA_FOLDER,
    PRICE_FOLDER,
    OUTPUT_FOLDER_BACKTEST as OUTPUT_FOLDER,
    OUTPUT_CSV,
    PARAM_THRESHOLD_1,
    PARAM_THRESHOLD_2,
    PARAM_WINDOW,
    TARGET_ALPHA_ID,
    ASSET_PRICE_SOURCE,
    BT_START_DATE,
    BT_END_DATE,
    FT_START_DATE,
    FT_END_DATE,
    VT_START_DATE,
    VT_END_DATE,
)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_FOLDER = Path(CONFIG_DATA_FOLDER)
price_folder = PRICE_FOLDER
START_DATE = BT_START_DATE
END_DATE = BT_END_DATE
START_DATE_FT = FT_START_DATE
END_DATE_FT = FT_END_DATE
START_DATE_VT = VT_START_DATE
END_DATE_VT = VT_END_DATE

TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d"
}

def topic_to_filename(topic: str) -> str:
    topic = topic.replace("|", "_")
    topic = topic.replace("/", "_").replace("-", "_")
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


def timeframe_to_timedelta(timeframe: str):
    mapping = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
    }
    return mapping.get(timeframe)


def infer_expected_gap(times):
    time_sorted = pd.Series(pd.to_datetime(times, errors="coerce")).dropna().drop_duplicates().sort_values()
    gaps = time_sorted.diff().dropna()
    return gaps.median() if not gaps.empty else None


def build_qc_result(df, source, value_col, alpha_id=None, period_name=None, expected_start=None, expected_end=None, expected_gap=None):
    if df is None or df.empty:
        return {
            "alpha_id": alpha_id,
            "period": period_name,
            "source": source,
            "rows": 0,
            "invalid_time": 0,
            "duplicated_time": 0,
            "missing_rows_est": 0,
            "max_gap": "NA",
            "start": "NA",
            "end": "NA",
            "starts_late": "NA",
            "ends_early": "NA",
            "nan_rate": "NA",
        }

    if isinstance(df.index, pd.DatetimeIndex):
        times = pd.Series(df.index)
    else:
        times = pd.to_datetime(df["time"], errors="coerce") if "time" in df.columns else pd.Series(dtype="datetime64[ns]")

    invalid_time = int(times.isna().sum())
    time_sorted = times.dropna().drop_duplicates().sort_values()
    duplicated_time = int(times.dropna().duplicated().sum())
    gaps = time_sorted.diff().dropna()
    expected_gap = expected_gap or infer_expected_gap(time_sorted)
    missing_rows = 0
    if expected_gap is not None and expected_gap > pd.Timedelta(0):
        missing_rows = int(sum(max(0, round(gap / expected_gap) - 1) for gap in gaps))

    start_time = time_sorted.min() if not time_sorted.empty else pd.NaT
    end_time = time_sorted.max() if not time_sorted.empty else pd.NaT
    max_gap = gaps.max() if not gaps.empty else pd.NaT
    nan_rate = float(df[value_col].isna().mean()) if value_col in df.columns else np.nan

    return {
        "alpha_id": alpha_id,
        "period": period_name,
        "source": source,
        "rows": len(df),
        "invalid_time": invalid_time,
        "duplicated_time": duplicated_time,
        "missing_rows_est": missing_rows,
        "max_gap": str(max_gap) if pd.notna(max_gap) else "NA",
        "start": str(start_time) if pd.notna(start_time) else "NA",
        "end": str(end_time) if pd.notna(end_time) else "NA",
        "starts_late": bool(pd.notna(start_time) and expected_start is not None and start_time > pd.to_datetime(expected_start)),
        "ends_early": bool(pd.notna(end_time) and expected_end is not None and end_time < pd.to_datetime(expected_end)),
        "nan_rate": round(nan_rate, 4) if not np.isnan(nan_rate) else "NA",
    }


def load_price_df(trade_asset: str, timeframe: str, price_delay: int, price_folder: str, alpha_id=None, period_name=None, expected_start=None, expected_end=None):
    if trade_asset not in ASSET_PRICE_SOURCE:
        raise ValueError(f"No price source defined for asset: {trade_asset}")

    exchange = ASSET_PRICE_SOURCE[trade_asset]
    symbol = f"{trade_asset}USDT"
    filename = f"{exchange}_1m_{symbol}.csv"
    file_path = os.path.join(price_folder, filename)

    if not os.path.exists(file_path):
        print(f"[PRICE] File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df['time'] = parse_time_column(df['start_time'])
    df = df.set_index('time')
    df = df[['close']].astype(float)
    if price_delay != 0:
        df['close'] = df['close'].shift(price_delay)

    RESOLUTION = TIMEFRAME_MAP.get(timeframe)
    if RESOLUTION is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    df = df.resample(RESOLUTION).last()
    qc_result = build_qc_result(
        df,
        f"{period_name or ''} price {trade_asset} {timeframe}".strip(),
        "close",
        alpha_id=alpha_id,
        period_name=period_name,
        expected_start=expected_start,
        expected_end=expected_end,
        expected_gap=timeframe_to_timedelta(timeframe),
    )

    return df

def apply_diff_from_formula(df: pd.DataFrame, alpha_formula: str):
    return transformation_lib.apply_diff_from_formula(df, alpha_formula)

def compute_alpha_signal(df: pd.DataFrame, formula: str):
    return transformation_lib.compute_alpha_signal(df, formula)


def calculate_trade_win_rate(df):
    boundary_mask = (df['pos'] != df['pos_t-1']) & (df['pos_t-1'] != 0)
    trade_boundary_cumu = df.loc[boundary_mask, 'cumu']
    if trade_boundary_cumu.empty:
        return 0.0

    trade_pnl = trade_boundary_cumu.diff()
    trade_pnl.iloc[0] = trade_boundary_cumu.iloc[0]
    return round((trade_pnl > 0).mean(), 4)
    
def calculate_metrics(merged_df, window, thresholds):

    merged_df['chg'] = merged_df['close'].pct_change()
    merged_df['pos_t-1'] = merged_df['pos'].shift(1).fillna(0)
    merged_df['trades'] = (merged_df['pos_t-1'] - merged_df['pos']).abs()
    merged_df['pnl'] = (merged_df['pos_t-1'] * merged_df['chg']- merged_df['trades'] * 0.00035)
    merged_df['pnl'] = merged_df['pnl'].fillna(0)
    merged_df['cumu'] = merged_df['pnl'].cumsum()
    merged_df['dd'] = merged_df['cumu'] - merged_df['cumu'].cummax()

    pnl_std = merged_df['pnl'].std()

    SR = (round(merged_df['pnl'].mean() / pnl_std * np.sqrt(365 * 24), 4)if pnl_std != 0 else np.nan)
    MDD = round(merged_df['dd'].min(), 4)
    AR = round(merged_df['pnl'].mean() * 365 * 24, 4)
    CR = round(AR / abs(MDD), 4) if MDD != 0 else np.nan
    TR = round(merged_df['cumu'].iloc[-1], 4)
    num_trades = int(merged_df['trades'].sum())
    TPI = (round(num_trades / (len(merged_df) - window) * 100, 2) if len(merged_df) > window else np.nan)
    SR_MDD = round(SR / abs(MDD), 2) if MDD != 0 else np.nan
    VAR = round(merged_df['pnl'].quantile(0.05) * 100, 4)
    win_rate = calculate_trade_win_rate(merged_df)
    drawdown_periods = (merged_df['dd'] < 0).astype(int)
    drawdown_periods = drawdown_periods.groupby((drawdown_periods != drawdown_periods.shift()).cumsum()).cumsum()
    longest_drawdown = int(drawdown_periods.max())

    return {
        "SR": SR,
        "MDD": MDD,
        "AR": AR,
        "CR": CR,
        "TR": TR,
        "TPI": TPI,
        "SR_MDD": SR_MDD,
        "VAR": VAR,
        "win_rate": win_rate,
        "num_trades": num_trades,
        "LDD": longest_drawdown,
    }

def load_csv(topic: str, column_name: str, datasource_key: str, alpha_id=None, period_name=None, expected_start=None, expected_end=None):
    file_path = DATA_FOLDER / topic_to_filename(topic)
    if not file_path.exists():
        print(f"CSV not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df['time'] = parse_time_column(df['start_time'])

    if column_name in df.columns:
        df = df[[column_name, 'time']].rename(columns={column_name: datasource_key})

    elif 'o' in df.columns:
        def get_key_from_dict(x, key):
            if pd.isnull(x):
                return None
            try:
                return ast.literal_eval(x).get(key, None)
            except Exception:
                return None

        df[datasource_key] = df['o'].apply(lambda x: get_key_from_dict(x, column_name))
        df = df[['time', datasource_key]]

    else:
        print(f"Column '{column_name}' not found in CSV {file_path}")
        return None

    qc_result = build_qc_result(
        df,
        f"{period_name or ''} {alpha_id or ''} factor {datasource_key}".strip(),
        datasource_key,
        alpha_id=alpha_id,
        period_name=period_name,
        expected_start=expected_start,
        expected_end=expected_end,
        expected_gap=infer_expected_gap(df["time"]),
    )

    return df


def standardize_alpha_output_columns(df, datasource_keys, price_delay):
    df = df.copy()
    rename_map = {}
    if len(datasource_keys) >= 1 and datasource_keys[0] in df.columns:
        rename_map[datasource_keys[0]] = "factor_1"
    if len(datasource_keys) >= 2 and datasource_keys[1] in df.columns:
        rename_map[datasource_keys[1]] = "factor_2"
    df = df.rename(columns=rename_map)

    if "data" not in df.columns and "factor" in df.columns:
        insert_at = df.columns.get_loc("factor")
        df.insert(insert_at, "data", df["factor"])

    df["price_shift_candle_minute"] = int(price_delay)

    leading_columns = [
        "factor_1",
        "transform_1",
        "factor_2",
        "transform_2",
        "data",
        "factor",
    ]
    trailing_columns = [
        "signal",
        "pos",
        "close",
        "price_shift_candle_minute",
        "chg",
        "pos_t-1",
        "trades",
        "pnl",
        "cumu",
        "dd",
    ]
    ordered_columns = [col for col in leading_columns if col in df.columns]
    reserved_columns = set(leading_columns + trailing_columns)
    ordered_columns.extend(
        col for col in df.columns
        if col not in reserved_columns and not str(col).startswith("Unnamed")
    )
    ordered_columns.extend(
        col for col in trailing_columns
        if col in df.columns and col not in ordered_columns
    )
    return df[ordered_columns]


def load_alpha_data(
    json_file: str,
    START_DATE: str,
    END_DATE: str,
    period_name: str,
    target_alpha_id=None,
):
    with open(json_file, "r") as f:
        alphas = json.load(f)

    if target_alpha_id is not None:
        target_ids = target_alpha_id if isinstance(target_alpha_id, (list, tuple, set)) else [target_alpha_id]
        target_ids = set(target_ids)
        alphas = [alpha for alpha in alphas if alpha.get("alpha_id") in target_ids]
        if not alphas:
            raise ValueError(f"Alpha ID not found: {target_alpha_id}")

    alpha_dfs = {}
    results = [] 
    PRICE_CACHE = {}

    for alpha in alphas:
        alpha_id = alpha.get("alpha_id")
        ds_str = alpha.get("datasource_structure", "{}")
        try:
            ds = json.loads(ds_str.replace("'", '"'))
        except json.JSONDecodeError:
            print(f"Failed to parse datasource_structure for {alpha_id}")
            continue
        datasource_keys = list(ds.keys())

        merged_df = None
        skip_alpha_period = False

        # SINGLE DATA PROCESSING & MERGED
        for key, val in ds.items():
            topic = val.get("topic")
            column_name = val.get("column")
            if topic and column_name:
                df = load_csv(
                    topic,
                    column_name,
                    key,
                    alpha_id=alpha_id,
                    period_name=period_name,
                    expected_start=START_DATE,
                    expected_end=END_DATE,
                )
                if df is None:
                    skip_alpha_period = True
                    break

                df = df.dropna(subset=[key])

                if merged_df is None:
                    merged_df = df
                else:
                    merged_df = pd.merge(merged_df, df, on='time', how='inner')
                        

        if skip_alpha_period:
            print(f"[SKIP] {alpha_id} {period_name}: datasource QC failed or missing")
            continue

        if merged_df is not None:
            merged_df = merged_df.drop(columns=[c for c in merged_df.columns if 'start_time' in c])
            merged_df = merged_df.set_index('time')

            alpha_formula = alpha.get("alpha_formula", "")
            if alpha_formula:
                merged_df, rewritten_formula = apply_diff_from_formula(
                    merged_df, alpha_formula
                )
                # Keep production-style names in the exported CSV. Formula
                # diff columns remain available under their descriptive
                # names as well (for example, ``taker_buy_volume_diff``).
                for transform_number, datasource_key in enumerate(datasource_keys[:2], start=1):
                    source_column = f"{datasource_key}_diff"
                    if source_column in merged_df.columns:
                        merged_df[f"transform_{transform_number}"] = merged_df[source_column]

                for col in merged_df.columns:
                    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

                alpha_signal = compute_alpha_signal(merged_df, rewritten_formula)
                if alpha_signal is not None:
                    merged_df["factor"] = alpha_signal

                merged_df['factor'] = merged_df['factor'].replace([np.inf, -np.inf], np.nan)
                if abs(merged_df["factor"].min()) < 1.0:
                    merged_df["factor"] = merged_df["factor"].round(10)
                else:
                    merged_df["factor"] = merged_df["factor"].round(10)

                

                if alpha.get("data_preprocessing") == "diff":
                    merged_df["transform"] = merged_df["factor"].diff()
                    merged_df = merged_df.dropna(subset=["transform"])
                    merged_df["factor"] = merged_df["transform"]

                merged_df = merged_df.dropna()

                # SIGNAL
                WINDOW = int(PARAM_WINDOW) if PARAM_WINDOW is not None else int(alpha.get("window", alpha.get("rolling_window_1", 0)))
                MODEL = alpha.get("model", None)
                if "factor" in merged_df.columns and MODEL is not None:
                    merged_df = models_lib.choose_model(merged_df, WINDOW, MODEL)
                    merged_df = merged_df.dropna(subset=["signal"])
                    if "signal" not in merged_df.columns:
                        raise KeyError(f"'signal' column missing after model for alpha {alpha_id}")

                # ENTRY EXIT LOGIC
                entry_exit_logic = alpha.get("entry_exit_logic", "").lower()
                if "_long" in entry_exit_logic:
                    LOGIC = entry_exit_logic.replace("_long", "")
                    SIDE = "long"
                elif "_short" in entry_exit_logic:
                    LOGIC = entry_exit_logic.replace("_short", "")
                    SIDE = "short"
                else:
                    LOGIC = entry_exit_logic
                    SIDE = "both"

                long_entry_threshold = alpha.get("long_entry_threshold", 0.0)
                long_exit_threshold  = alpha.get("long_exit_threshold", 0.0)
                short_entry_threshold = alpha.get("short_entry_threshold", 0.0)
                short_exit_threshold  = alpha.get("short_exit_threshold", 0.0)

                entry_threshold = max(long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold)
                exit_threshold = min(long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold)
                entry_threshold = float(PARAM_THRESHOLD_1) if PARAM_THRESHOLD_1 is not None else entry_threshold
                exit_threshold = float(PARAM_THRESHOLD_2) if PARAM_THRESHOLD_2 is not None else exit_threshold

                merged_df = entry_exit_logic_lib.signal_logic_2(merged_df, entry_threshold, exit_threshold, LOGIC, SIDE)

                # REINDEX & MERGE PRICE
                timeframe = alpha.get("timeframe", "1h")
                RESOLUTION = TIMEFRAME_MAP.get(timeframe)
                if RESOLUTION is None:
                    raise ValueError(f"Unsupported timeframe: {timeframe}")

                full_range = pd.date_range(start=pd.to_datetime(START_DATE), end=pd.to_datetime(END_DATE), freq=RESOLUTION)
                merged_df = merged_df.loc[START_DATE:END_DATE].copy()
                merged_df = merged_df.reindex(full_range)
                merged_df.index.name = "time"
                merged_df = merged_df.ffill()

                trade_asset = alpha.get("trade_asset")
                PRICE_DELAY = alpha.get("shift_backtest_candle_minute", 0)

                key = (trade_asset, timeframe, PRICE_DELAY)

                if key not in PRICE_CACHE:
                    PRICE_CACHE[key] = load_price_df(
                        trade_asset,
                        timeframe,
                        PRICE_DELAY,
                        price_folder,
                        alpha_id=alpha_id,
                        period_name=period_name,
                        expected_start=START_DATE,
                        expected_end=END_DATE,
                    )

                price_df = PRICE_CACHE[key]
                if price_df is None:
                    print(f"[SKIP] {alpha_id} {period_name}: price QC failed or missing")
                    continue
                merged_df = pd.merge(price_df[['close']], merged_df, left_index=True, right_index=True, how='inner')

                metrics = calculate_metrics(
                    merged_df=merged_df,
                    window=WINDOW,
                    thresholds={"entry": entry_threshold, "exit": exit_threshold}
                )
                merged_df = standardize_alpha_output_columns(merged_df, datasource_keys, PRICE_DELAY)

                # Add period info here
                metrics_row = {
                    "period": period_name,
                    "start_date": START_DATE,
                    "end_date": END_DATE,
                    "alpha_id": alpha_id,
                    "custom_id": alpha.get("custom_id"),
                    "model": alpha.get("model"),
                    "timeframe": alpha.get("timeframe"),
                    "window": WINDOW,
                    "entry_exit_logic": alpha.get("entry_exit_logic"),
                    "entry_threshold": entry_threshold,
                    "exit_threshold": exit_threshold,
                    **metrics
                }

                results.append(metrics_row)
                alpha_dfs[alpha_id] = merged_df

    return alpha_dfs, results


if __name__ == "__main__":

    ALL_RESULTS = []

    PERIODS = [
        ("IS", START_DATE, END_DATE),
        ("FT", START_DATE_FT, END_DATE_FT),
        ("VT", START_DATE_VT, END_DATE_VT),
    ]

    for period_name, start, end in PERIODS:
        alpha_data, metrics_rows = load_alpha_data(
            JSON_FILE,
            START_DATE=start,
            END_DATE=end,
            period_name=period_name,
            target_alpha_id=TARGET_ALPHA_ID
        )
        ALL_RESULTS.extend(metrics_rows)

    if not ALL_RESULTS:
        print("[SKIP] No alpha results generated because data QC failed or no data loaded.")
        raise SystemExit(0)

    df_all = pd.DataFrame(ALL_RESULTS)
    df_all = df_all.sort_values(by="alpha_id").reset_index(drop=True)
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAll periods appended and sorted by alpha_id into {OUTPUT_CSV}")


    OUTPUT_FOLDER = Path(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    df_all = pd.DataFrame(ALL_RESULTS)
    df_all = df_all.sort_values(by="alpha_id").reset_index(drop=True)

    csv_path = OUTPUT_FOLDER / "alpha_metrics_all_periods.csv"
    df_all.to_csv(csv_path, index=False)

    print(f"\nCSV saved to: {csv_path.resolve()}")

    with open(JSON_FILE, "r") as f:
        original_alphas = json.load(f)

    if TARGET_ALPHA_ID is not None:
        target_ids = TARGET_ALPHA_ID if isinstance(TARGET_ALPHA_ID, (list, tuple, set)) else [TARGET_ALPHA_ID]
        target_ids = set(target_ids)
        original_alphas = [a for a in original_alphas if a.get("alpha_id") in target_ids]

    ALPHA_META = {}

    for a in original_alphas:
        ALPHA_META[a["alpha_id"]] = {
            "alpha_formula": a.get("alpha_formula"),
            "manual_alpha_formula": a.get("manual_alpha_formula"),
            "data_preprocessing": a.get("data_preprocessing"),
            "preprocessing_window": a.get("preprocessing_window"),
            "datasource_structure": a.get("datasource_structure"),
            "data_asset": a.get("data_asset"),
            "trade_asset": a.get("trade_asset"),
            "shift_backtest_candle_minute": a.get("shift_backtest_candle_minute", 0),
            "fees": a.get("fees"),
        }

    PERIOD_ORDER = ["IS", "FT", "VT"]
    grouped = {}

    for _, row in df_all.iterrows():
        alpha_id = row["alpha_id"]

        if alpha_id not in grouped:

            meta = ALPHA_META.get(alpha_id, {})

            grouped[alpha_id] = {
                "alpha_id": alpha_id,
                "custom_id": row["custom_id"],
                "model": row["model"],
                "timeframe": row["timeframe"],
                "entry_exit_logic": row["entry_exit_logic"],
                "window": row["window"],
                "entry_threshold": row["entry_threshold"],
                "exit_threshold": row["exit_threshold"],

                # 🔥 NEW FIELDS
                "alpha_formula": meta.get("alpha_formula"),
                "manual_alpha_formula": meta.get("manual_alpha_formula"),
                "data_preprocessing": meta.get("data_preprocessing"),
                "preprocessing_window": meta.get("preprocessing_window"),
                "datasource_structure": meta.get("datasource_structure"),
                "data_asset": meta.get("data_asset"),
                "trade_asset": meta.get("trade_asset"),
                "shift_backtest_candle_minute": meta.get("shift_backtest_candle_minute"),
                "fees": meta.get("fees"),

                "periods": []
            }

        grouped[alpha_id]["periods"].append({
            "period": row["period"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "SR": row["SR"],
            "MDD": row["MDD"],
            "AR": row["AR"],
            "CR": row["CR"],
            "TR": row["TR"],
            "TPI": row["TPI"],
            "SR_MDD": row["SR_MDD"],
            "VAR": row["VAR"],
            "win_rate": row["win_rate"],
            "num_trades": row["num_trades"],
            "LDD": row["LDD"]
        })


    for alpha_id in grouped:
        grouped[alpha_id]["periods"] = sorted(
            grouped[alpha_id]["periods"],
            key=lambda x: PERIOD_ORDER.index(x["period"])
        )


    json_folder = OUTPUT_FOLDER / "alpha_json"
    json_folder.mkdir(exist_ok=True)

    for alpha_id, data in grouped.items():
        output_path = json_folder / f"{alpha_id}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\nAlpha JSON result -> {output_path.resolve()}")
        print(json.dumps(data, indent=4))

    print(f"JSON files saved to: {json_folder.resolve()}")
    print(f"Generated {len(grouped)} alpha JSON files.")

    for period_name, start, end in PERIODS:
        alpha_data, metrics_rows = load_alpha_data(
            JSON_FILE,
            START_DATE=start,
            END_DATE=end,
            period_name=period_name,
            target_alpha_id=TARGET_ALPHA_ID
        )

        ALL_RESULTS.extend(metrics_rows)

        for alpha_id, df in alpha_data.items():
            if period_name == "VT":
                vt_path = json_folder / f"{alpha_id}.csv"
                df.to_csv(vt_path)
                print(f"VT CSV saved -> {vt_path.resolve()}")

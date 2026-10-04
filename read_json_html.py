import json
import os
import re
import ast
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from scipy.stats import probplot, ttest_ind
from lib import entry_exit_logic_lib, models_lib, transformation_lib
from config import (
    JSON_FILE,
    DATA_FOLDER as CONFIG_DATA_FOLDER,
    PRICE_FOLDER,
    OUTPUT_FOLDER_HTML as OUTPUT_FOLDER,
    TARGET_ALPHA_ID,
    ASSET_PRICE_SOURCE,
    BT_START_DATE,
    BT_END_DATE,
    FT_START_DATE,
    FT_END_DATE,
    VT_START_DATE,
    VT_END_DATE,
    START_DATE,
    END_DATE,
    WINDOW_STEP,
    THRESHOLD_STEP,
    WINDOW_HEATMAP_DELTA,
    WINDOW_HEATMAP_STEPS,
    THRESHOLD_HEATMAP_STEPS,
    PARAM_WINDOW,
    PARAM_THRESHOLD_1,
    PARAM_THRESHOLD_2,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# =========================================================
# CONFIGURATION — adjust paths as needed
# =========================================================
DATA_FOLDER = Path(CONFIG_DATA_FOLDER)

TIMEFRAME_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h",   "2h": "2h",   "4h": "4h",     "1d": "1D"
}

MAX_NAN_RATE = 0.03

METRIC_ANNUALIZER_MAP = {
    "1m": 365 * 24 * 60, "5m": 365 * 24 * 60 / 5,
    "15m": 365 * 24 * 60 / 15, "30m": 365 * 24 * 60 / 30,
    "1h": 365 * 24, "2h": 365 * 12, "4h": 365 * 6, "1d": 365
}

# =========================================================
# HELPERS
# =========================================================
def topic_to_filename(topic: str) -> str:
    topic = topic.replace("|", "_").replace("/", "_").replace("-", "_")
    topic = topic.replace("?", "_").replace("&", "_").replace("=", "_")
    topic = "".join(c for c in topic if c.isalnum() or c == "_")
    return topic + ".csv"


def parse_start_time_column(values):
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.notna().all():
        parsed = pd.to_datetime(numeric_values, unit="ms", utc=True)
    else:
        parsed = pd.to_datetime(values, utc=True)

    return parsed.dt.tz_localize(None)


def nan_rate_failed(df, value_col, source):
    if value_col not in df.columns:
        return False
    nan_rate = float(df[value_col].isna().mean())
    if nan_rate > MAX_NAN_RATE:
        print(f"[SKIP] {source} nan_rate={nan_rate:.2%} > {MAX_NAN_RATE:.0%}")
        return True
    return False


def apply_param_overrides(window, threshold_1, threshold_2):
    if PARAM_WINDOW is not None:
        window = int(PARAM_WINDOW)
    if PARAM_THRESHOLD_1 is not None:
        threshold_1 = float(PARAM_THRESHOLD_1)
    if PARAM_THRESHOLD_2 is not None:
        threshold_2 = float(PARAM_THRESHOLD_2)
    return window, threshold_1, threshold_2


def load_price_df(trade_asset: str, timeframe: str, price_delay: int):
    exchange = ASSET_PRICE_SOURCE.get(trade_asset)
    if not exchange:
        raise ValueError(f"No price source for asset: {trade_asset}")
    symbol   = f"{trade_asset}USDT"
    filename = f"{exchange}_1m_{symbol}.csv"
    filepath = os.path.join(PRICE_FOLDER, filename)
    if not os.path.exists(filepath):
        print(f"[PRICE] File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    df['time'] = parse_start_time_column(df['start_time'])
    df = df.set_index('time')[['close']].astype(float)
    if price_delay != 0:
        df['close'] = df['close'].shift(price_delay)

    resolution = TIMEFRAME_MAP.get(timeframe)
    if not resolution:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    df = df.resample(resolution).last()
    if nan_rate_failed(df, "close", f"price {trade_asset} {timeframe}"):
        return None
    return df


def load_csv(topic: str, column_name: str, datasource_key: str):

    file_path = DATA_FOLDER / topic_to_filename(topic)
    if not file_path.exists():
        print(f"[DATA] CSV not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df['time'] = parse_start_time_column(df['start_time'])

    # Skip metadata columns — grab all actual data columns
    skip_cols = {'start_time', 'datetime', 'time'}
    data_cols = [c for c in df.columns if c not in skip_cols]

    if column_name in data_cols:
        # Exact match — use it directly
        df = df[['time', column_name]].rename(columns={column_name: datasource_key})

    elif 'o' in data_cols:
        # JSON dict column — extract by key
        def get_key(x):
            if pd.isnull(x):
                return None
            try:
                return ast.literal_eval(x).get(column_name, None)
            except Exception:
                return None
        df[datasource_key] = df['o'].apply(get_key)
        df = df[['time', datasource_key]]

    elif len(data_cols) == 1:
        # Only one data column — just use it regardless of name
        actual_col = data_cols[0]
        print(f"[INFO] Column '{column_name}' not found, using only available column '{actual_col}' for {datasource_key}")
        df = df[['time', actual_col]].rename(columns={actual_col: datasource_key})

    else:
        # Multiple columns — try partial match on column_name
        partial = [c for c in data_cols if column_name in c or c in column_name]
        if partial:
            actual_col = partial[0]
            print(f"[INFO] Column '{column_name}' not found, using partial match '{actual_col}' for {datasource_key}")
            df = df[['time', actual_col]].rename(columns={actual_col: datasource_key})
        else:
            print(f"[WARN] Column '{column_name}' not found in {file_path.name}. Available: {data_cols}")
            return None

    if nan_rate_failed(df, datasource_key, f"factor {datasource_key}"):
        return None

    return df


def apply_diff_from_formula(df: pd.DataFrame, alpha_formula: str):
    return transformation_lib.apply_diff_from_formula(df, alpha_formula)


def compute_alpha_signal(df: pd.DataFrame, formula: str):
    return transformation_lib.compute_alpha_signal(df, formula)


# =========================================================
# METRICS
# =========================================================
def calculate_t_test(bt_df, ft_df):
    bt_pnl = bt_df['pnl'].dropna()
    ft_pnl = ft_df['pnl'].dropna()
    if len(bt_pnl) == 0 or len(ft_pnl) == 0:
        return np.nan, np.nan
    t, p = ttest_ind(bt_pnl, ft_pnl, equal_var=False)
    return round(t, 4), round(p, 4)


def calculate_trade_win_rate(df):
    trades_df = extract_trade_records(df)
    if trades_df.empty:
        return 0.0
    return round((trades_df['trade_return'] > 0).mean(), 4)


def extract_trade_records(df):
    boundary_mask = (df['pos'] != df['pos_t-1']) & (df['pos_t-1'] != 0)
    trade_boundaries = df.loc[boundary_mask, ['cumu', 'pos_t-1']].copy()
    if trade_boundaries.empty:
        return pd.DataFrame(columns=['side', 'trade_return'])

    trade_boundaries['trade_return'] = trade_boundaries['cumu'].diff()
    trade_boundaries.iloc[0, trade_boundaries.columns.get_loc('trade_return')] = trade_boundaries['cumu'].iloc[0]
    trade_boundaries['side'] = trade_boundaries['pos_t-1'].map({1.0: 'Long', -1.0: 'Short'}).fillna('Unknown')
    return trade_boundaries[['side', 'trade_return']]


def calculate_side_metrics(df, side_value, side_label, annualizer):
    side_mask = df['pos_t-1'] == side_value
    side_pnl = df['pnl'].where(side_mask, 0.0).fillna(0.0)
    side_cumu = side_pnl.cumsum()
    side_dd = side_cumu - side_cumu.cummax()
    side_std = side_pnl.std()

    trades_df = extract_trade_records(df)
    side_trades = trades_df[trades_df['side'] == side_label]

    return {
        'Side': side_label,
        'SR': round(side_pnl.mean() / side_std * np.sqrt(annualizer), 4) if pd.notna(side_std) and side_std != 0 else np.nan,
        'AR': round(side_pnl.mean() * annualizer, 4),
        'MDD': round(side_dd.min(), 4) if not side_dd.empty else np.nan,
        'trade_count': int(len(side_trades)),
        'win_rate': round((side_trades['trade_return'] > 0).mean(), 4) if not side_trades.empty else 0.0,
        'avg_trade_return': round(side_trades['trade_return'].mean(), 4) if not side_trades.empty else np.nan,
        'cumu_series': side_cumu,
    }


def calculate_regime_metrics(df, annualizer, label, trades_df=None):
    if df.empty:
        return {"Regime": label, "SR": np.nan, "AR": np.nan, "MDD": np.nan, "trade_count": 0}

    regime_pnl = df['pnl'].fillna(0.0)
    regime_cumu = regime_pnl.cumsum()
    regime_dd = regime_cumu - regime_cumu.cummax()
    regime_std = regime_pnl.std()
    if trades_df is None:
        trades_df = extract_trade_records(df)
    else:
        trades_df = trades_df.loc[trades_df.index.isin(df.index)]

    return {
        "Regime": label,
        "SR": round(regime_pnl.mean() / regime_std * np.sqrt(annualizer), 4) if pd.notna(regime_std) and regime_std != 0 else np.nan,
        "AR": round(regime_pnl.mean() * annualizer, 4),
        "MDD": round(regime_dd.min(), 4) if not regime_dd.empty else np.nan,
        "trade_count": int(len(trades_df)),
    }


def build_long_short_summary(df, annualizer):
    long_metrics = calculate_side_metrics(df, 1.0, 'Long', annualizer)
    short_metrics = calculate_side_metrics(df, -1.0, 'Short', annualizer)
    summary_df = pd.DataFrame([
        {k: v for k, v in long_metrics.items() if k != 'cumu_series'},
        {k: v for k, v in short_metrics.items() if k != 'cumu_series'},
    ])
    return summary_df, long_metrics['cumu_series'], short_metrics['cumu_series']


def build_regime_dataset(df, btc_price_df, annualizer):
    if df.empty or btc_price_df is None or btc_price_df.empty:
        return pd.DataFrame()

    regime_df = df.copy()
    regime_df['btc_close'] = btc_price_df['close'].reindex(regime_df.index).ffill()
    bars_per_day = max(1, int(round(annualizer / 365)))
    ma_window = max(2, 200 * bars_per_day)
    vol_window = max(2, 30 * bars_per_day)
    vol_threshold_window = max(2, 365 * bars_per_day)

    regime_df['btc_ma_200d'] = regime_df['btc_close'].rolling(ma_window, min_periods=ma_window).mean()
    regime_df['btc_return'] = regime_df['btc_close'].pct_change()
    regime_df['btc_vol_30d'] = regime_df['btc_return'].rolling(vol_window, min_periods=vol_window).std() * np.sqrt(annualizer)
    regime_df['trend_valid'] = regime_df['btc_ma_200d'].notna()
    regime_df['trend_above_200d'] = regime_df['trend_valid'] & (regime_df['btc_close'] > regime_df['btc_ma_200d'])
    regime_df['trend_below_200d'] = regime_df['trend_valid'] & (regime_df['btc_close'] < regime_df['btc_ma_200d'])
    regime_df['vol_median'] = (
        regime_df['btc_vol_30d']
        .rolling(vol_threshold_window, min_periods=vol_threshold_window)
        .median()
        .shift(1)
    )
    regime_df['vol_valid'] = regime_df['btc_vol_30d'].notna() & regime_df['vol_median'].notna()
    regime_df['high_vol'] = regime_df['vol_valid'] & (regime_df['btc_vol_30d'] >= regime_df['vol_median'])
    regime_df['low_vol'] = regime_df['vol_valid'] & (regime_df['btc_vol_30d'] < regime_df['vol_median'])

    return regime_df


def simulate_threshold_positions(df, entry_threshold, exit_threshold, logic, side, long_allowed, short_allowed):
    """Re-run threshold entries while applying past-only regime permissions."""
    signal = pd.to_numeric(df.get('signal'), errors='coerce')
    if signal is None:
        return np.zeros(len(df)), 0

    long_allowed = pd.Series(long_allowed, index=df.index).fillna(False).astype(bool)
    short_allowed = pd.Series(short_allowed, index=df.index).fillna(False).astype(bool)
    if side == 'long':
        short_allowed[:] = False
    elif side == 'short':
        long_allowed[:] = False

    reverse_logic = logic in {'trend_reverse', 'mr_reverse', 'fast_reverse'}
    mean_reversion_logic = logic in {'mr', 'mr_reverse'}
    fast_logic = logic in {'fast', 'fast_reverse', 'trend_price_regime'}
    supported_logic = logic in {
        'trend', 'trend_reverse', 'mr', 'mr_reverse',
        'fast', 'fast_reverse', 'trend_price_regime'
    }

    if not supported_logic:
        base_position = pd.to_numeric(df.get('pos', 0), errors='coerce').fillna(0).to_numpy()
        allowed = np.where(base_position > 0, long_allowed.to_numpy(), short_allowed.to_numpy())
        return np.where(allowed, base_position, 0.0), 0

    positions = np.zeros(len(df), dtype=float)
    current_position = 0.0
    forced_exits = 0
    previous_signal = np.nan

    for idx, current_signal in enumerate(signal.to_numpy()):
        allow_long = bool(long_allowed.iloc[idx])
        allow_short = bool(short_allowed.iloc[idx])

        if current_position == 1.0 and not allow_long:
            current_position = 0.0
            forced_exits += 1
        elif current_position == -1.0 and not allow_short:
            current_position = 0.0
            forced_exits += 1

        if not np.isfinite(current_signal):
            positions[idx] = current_position
            previous_signal = current_signal
            continue

        if reverse_logic:
            long_trigger = current_signal <= exit_threshold
            short_trigger = current_signal >= entry_threshold
        else:
            long_trigger = current_signal >= entry_threshold
            short_trigger = current_signal <= exit_threshold

        if long_trigger:
            current_position = 1.0 if allow_long else 0.0
        elif short_trigger:
            current_position = -1.0 if allow_short else 0.0
        elif fast_logic:
            current_position = 0.0
        elif mean_reversion_logic and np.isfinite(previous_signal) and current_signal * previous_signal < 0:
            current_position = 0.0

        positions[idx] = current_position
        previous_signal = current_signal

    return positions, forced_exits


def calculate_regime_robustness(df, btc_price_df, annualizer, window, fees, entry_threshold, exit_threshold, logic, side):
    regime_df = build_regime_dataset(df, btc_price_df, annualizer)
    if regime_df.empty:
        columns = ["Simulation", "SR", "AR", "MDD", "TR", "completed_trades", "win_rate", "exposure_pct", "turnover", "forced_exits"]
        return pd.DataFrame(columns=columns), regime_df, pd.DataFrame()

    never = pd.Series(False, index=regime_df.index)
    bull_high = regime_df['trend_above_200d'] & regime_df['high_vol']
    bull_low = regime_df['trend_above_200d'] & regime_df['low_vol']
    bear_high = regime_df['trend_below_200d'] & regime_df['high_vol']
    bear_low = regime_df['trend_below_200d'] & regime_df['low_vol']
    simulations = {
        "Baseline": (pd.to_numeric(regime_df['pos'], errors='coerce').fillna(0).to_numpy(), 0),
        "200D Trend Filter": simulate_threshold_positions(
            regime_df, entry_threshold, exit_threshold, logic, side,
            regime_df['trend_above_200d'], regime_df['trend_below_200d']
        ),
        "High Vol Only": simulate_threshold_positions(
            regime_df, entry_threshold, exit_threshold, logic, side,
            regime_df['high_vol'], regime_df['high_vol']
        ),
        "Low Vol Only": simulate_threshold_positions(
            regime_df, entry_threshold, exit_threshold, logic, side,
            regime_df['low_vol'], regime_df['low_vol']
        ),
        "Bull >200D Long Only": simulate_threshold_positions(
            regime_df, entry_threshold, exit_threshold, logic, side,
            regime_df['trend_above_200d'], never
        ),
        "Bear <200D Short Only": simulate_threshold_positions(
            regime_df, entry_threshold, exit_threshold, logic, side,
            never, regime_df['trend_below_200d']
        ),
        "Bull + High Vol Long": simulate_threshold_positions(
            regime_df, entry_threshold, exit_threshold, logic, side, bull_high, never
        ),
        "Bull + Low Vol Long": simulate_threshold_positions(
            regime_df, entry_threshold, exit_threshold, logic, side, bull_low, never
        ),
        "Bear + High Vol Short": simulate_threshold_positions(
            regime_df, entry_threshold, exit_threshold, logic, side, never, bear_high
        ),
        "Bear + Low Vol Short": simulate_threshold_positions(
            regime_df, entry_threshold, exit_threshold, logic, side, never, bear_low
        ),
    }

    rows = []
    curve_df = pd.DataFrame(index=regime_df.index)
    for label, (positions, forced_exits) in simulations.items():
        simulation_df = regime_df.copy()
        simulation_df['pos'] = positions
        metrics, simulation_df = calculate_metrics(simulation_df, window, annualizer, fees)
        completed_trades = extract_trade_records(simulation_df)
        rows.append({
            "Simulation": label,
            "SR": metrics.get("SR"),
            "AR": metrics.get("AR"),
            "MDD": metrics.get("MDD"),
            "TR": metrics.get("TR"),
            "completed_trades": int(len(completed_trades)),
            "win_rate": metrics.get("win_rate"),
            "exposure_pct": round(float((simulation_df['pos_t-1'] != 0).mean()) * 100, 2),
            "turnover": round(float(simulation_df['trades'].sum()), 2),
            "forced_exits": int(forced_exits),
        })
        curve_df[label] = simulation_df['cumu']

    return pd.DataFrame(rows), regime_df, curve_df


def calculate_metrics(df, window, annualizer, fees, ft_df=None):
    df = df.copy()
    df['chg']     = df['close'].pct_change()
    df['pos_t-1'] = df['pos'].shift(1).fillna(0)
    df['trades']  = (df['pos_t-1'] - df['pos']).abs()
    df['pnl']     = df['pos_t-1'] * df['chg'] - df['trades'] * fees
    df['pnl']     = df['pnl'].fillna(0)
    df['cumu']    = df['pnl'].cumsum()
    df['dd']      = df['cumu'] - df['cumu'].cummax()
    df['bah_cumu'] = df['chg'].cumsum()
    df['rolling_sharpe_180'] = (
        df['pnl'].rolling(180 * 24).mean() /
        df['pnl'].rolling(180 * 24).std() * np.sqrt(annualizer)
    )
    df['rolling_sharpe_365'] = (
        df['pnl'].rolling(annualizer).mean() /
        df['pnl'].rolling(annualizer).std() * np.sqrt(annualizer)
    )

    std_pnl   = df['pnl'].std()
    SR        = round(df['pnl'].mean() / std_pnl * np.sqrt(annualizer), 4) if std_pnl != 0 else np.nan
    MDD       = round(df['dd'].min(), 4)
    AR        = round(df['pnl'].mean() * annualizer, 4)
    CR        = round(AR / abs(MDD), 4) if MDD != 0 else np.nan
    TR        = round(df['cumu'].iloc[-1], 4)
    num_trades = int(df['trades'].sum())
    TPI       = round(num_trades / (len(df) - window) * 100, 4) if len(df) > window else np.nan
    VAR       = round(df['pnl'].quantile(0.05) * 100, 4)
    win_rate  = calculate_trade_win_rate(df)
    drawdown_periods = (df['dd'] < 0).astype(int)
    drawdown_periods = drawdown_periods.groupby(
        (drawdown_periods != drawdown_periods.shift()).cumsum()
    ).cumsum()
    longest_drawdown = int(drawdown_periods.max()) if not drawdown_periods.empty else 0
    BaH       = round(df['bah_cumu'].iloc[-1], 4)

    t_stat, p_val = (np.nan, np.nan)
    if ft_df is not None:
        t_stat, p_val = calculate_t_test(df, ft_df)

    metrics = {
        "SR": SR, "CR": CR, "MDD": MDD, "AR": AR,
        "TR": TR, "VAR": VAR, "win_rate": win_rate,
        "num_trades": num_trades, "LDD": longest_drawdown,
        "TPI": TPI, "BaH": BaH,
        "t_statistic": t_stat, "p_value": p_val
    }
    return metrics, df


def calculate_yearly_metrics(df, window, annualizer, fees):
    yearly_rows = []
    if df.empty:
        return pd.DataFrame(columns=["Year", "SR", "CR", "MDD", "AR", "TR", "VAR", "win_rate", "num_trades", "LDD", "TPI", "BaH"])

    for year in sorted(df.index.year.unique()):
        df_year = df[df.index.year == year].copy()
        if df_year.empty:
            continue
        metrics_year, _ = calculate_metrics(df_year, window, annualizer, fees)
        yearly_rows.append(
            {
                "Year": int(year),
                "SR": metrics_year["SR"],
                "CR": metrics_year["CR"],
                "MDD": metrics_year["MDD"],
                "AR": metrics_year["AR"],
                "TR": metrics_year["TR"],
                "VAR": metrics_year["VAR"],
                "win_rate": metrics_year["win_rate"],
                "num_trades": metrics_year["num_trades"],
                "LDD": metrics_year["LDD"],
                "TPI": metrics_year["TPI"],
                "BaH": metrics_year["BaH"],
            }
        )

    return pd.DataFrame(yearly_rows)


def build_stress_metrics_table(metrics_map, classes='metrics-table'):
    metric_keys = ['SR', 'CR', 'MDD', 'AR', 'TR', 'VAR', 'win_rate', 'num_trades', 'LDD', 'TPI', 'BaH']
    table_data = {'Metric': metric_keys}
    for column_name, metrics in metrics_map.items():
        table_data[column_name] = [metrics.get(k, '') for k in metric_keys]
    return pd.DataFrame(table_data).to_html(index=False, classes=classes)


def render_column_heatmap_table(df, classes="metrics-table", precision=4):
    if df.empty:
        return df.to_html(index=False, classes=classes)

    numeric_cols = [col for col in df.columns if col != "Year" and pd.api.types.is_numeric_dtype(df[col])]
    styler = (
        df.style
        .format(precision=precision, na_rep="")
        .background_gradient(subset=numeric_cols, cmap="Greens", axis=0)
    )
    return styler.to_html(index=False, table_attributes=f'class="{classes}"')


def _safe_corr(left, right, method):
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)


def _monthly_ic_ir(source, forward_return, method):
    pair = pd.concat([source.rename('source'), forward_return.rename('forward_return')], axis=1).dropna()
    if pair.empty:
        return np.nan

    monthly_values = []
    for _, month_df in pair.groupby(pair.index.to_period('M')):
        if len(month_df) < 10:
            continue
        value = _safe_corr(month_df['source'], month_df['forward_return'], method)
        if pd.notna(value):
            monthly_values.append(value)

    if len(monthly_values) < 2:
        return np.nan
    monthly_values = pd.Series(monthly_values, dtype=float)
    monthly_std = monthly_values.std(ddof=1)
    if pd.isna(monthly_std) or monthly_std == 0:
        return np.nan
    return monthly_values.mean() / monthly_std * np.sqrt(12)


def calculate_ic_metrics(df, market_price_df=None, horizons=None):
    """Calculate factor/signal IC and entry-event performance over several future horizons."""
    df = df.copy()
    horizons = horizons or [1, 3, 6, 12, 24, 72, 168]
    empty_summary = {"IC": np.nan, "RankIC": np.nan, "IC_IR": np.nan, "RankIC_IR": np.nan}
    summary_columns = [
        "Horizon_Bars", "Observations", "Factor_IC", "Factor_RankIC",
        "Signal_IC", "Signal_RankIC", "Factor_IC_IR", "Factor_RankIC_IR",
        "Signal_IC_IR", "Signal_RankIC_IR", "Q80_Q20_Spread",
        "Entry_Event_Mean_Return", "Entry_Event_Hit_Rate", "Entry_Event_Count"
    ]
    yearly_columns = [
        "Year", "Horizon_Bars", "Factor_IC", "Factor_RankIC",
        "Signal_IC", "Signal_RankIC", "Factor_IC_IR", "Factor_RankIC_IR"
    ]

    if "factor" not in df.columns or "signal" not in df.columns:
        df["rolling_ic_90"] = np.nan
        df["rolling_rank_ic_90"] = np.nan
        return empty_summary, pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=yearly_columns), df

    if market_price_df is not None and not market_price_df.empty and 'close' in market_price_df.columns:
        market_close = pd.to_numeric(market_price_df['close'], errors='coerce').reindex(df.index).ffill()
    elif 'close' in df.columns:
        market_close = pd.to_numeric(df['close'], errors='coerce')
    else:
        df["rolling_ic_90"] = np.nan
        df["rolling_rank_ic_90"] = np.nan
        return empty_summary, pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=yearly_columns), df

    factor = pd.to_numeric(df['factor'], errors='coerce')
    signal = pd.to_numeric(df['signal'], errors='coerce')
    position = pd.to_numeric(df.get('pos', 0), errors='coerce')
    if not isinstance(position, pd.Series):
        position = pd.Series(position, index=df.index, dtype=float)
    entry_direction = position.where((position != position.shift(1).fillna(0)) & (position != 0))

    horizon_rows = []
    yearly_rows = []
    for horizon in horizons:
        horizon = int(horizon)
        forward_return = market_close.shift(-horizon) / market_close - 1
        valid = pd.concat(
            [factor.rename('factor'), signal.rename('signal'), forward_return.rename('forward_return')],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()

        factor_ic = _safe_corr(factor, forward_return, 'pearson')
        factor_rank_ic = _safe_corr(factor, forward_return, 'spearman')
        signal_ic = _safe_corr(signal, forward_return, 'pearson')
        signal_rank_ic = _safe_corr(signal, forward_return, 'spearman')
        factor_ic_ir = _monthly_ic_ir(factor, forward_return, 'pearson')
        factor_rank_ic_ir = _monthly_ic_ir(factor, forward_return, 'spearman')
        signal_ic_ir = _monthly_ic_ir(signal, forward_return, 'pearson')
        signal_rank_ic_ir = _monthly_ic_ir(signal, forward_return, 'spearman')

        spread = np.nan
        if len(valid) >= 10 and valid['signal'].nunique() >= 5:
            q20 = valid['signal'].quantile(0.20)
            q80 = valid['signal'].quantile(0.80)
            top_mean = valid.loc[valid['signal'] >= q80, 'forward_return'].mean()
            bottom_mean = valid.loc[valid['signal'] <= q20, 'forward_return'].mean()
            spread = top_mean - bottom_mean

        event_returns = (entry_direction * forward_return).replace([np.inf, -np.inf], np.nan).dropna()
        horizon_rows.append({
            "Horizon_Bars": horizon,
            "Observations": int(len(valid)),
            "Factor_IC": round(factor_ic, 4) if pd.notna(factor_ic) else np.nan,
            "Factor_RankIC": round(factor_rank_ic, 4) if pd.notna(factor_rank_ic) else np.nan,
            "Signal_IC": round(signal_ic, 4) if pd.notna(signal_ic) else np.nan,
            "Signal_RankIC": round(signal_rank_ic, 4) if pd.notna(signal_rank_ic) else np.nan,
            "Factor_IC_IR": round(factor_ic_ir, 4) if pd.notna(factor_ic_ir) else np.nan,
            "Factor_RankIC_IR": round(factor_rank_ic_ir, 4) if pd.notna(factor_rank_ic_ir) else np.nan,
            "Signal_IC_IR": round(signal_ic_ir, 4) if pd.notna(signal_ic_ir) else np.nan,
            "Signal_RankIC_IR": round(signal_rank_ic_ir, 4) if pd.notna(signal_rank_ic_ir) else np.nan,
            "Q80_Q20_Spread": round(spread, 6) if pd.notna(spread) else np.nan,
            "Entry_Event_Mean_Return": round(event_returns.mean(), 6) if not event_returns.empty else np.nan,
            "Entry_Event_Hit_Rate": round((event_returns > 0).mean(), 4) if not event_returns.empty else np.nan,
            "Entry_Event_Count": int(len(event_returns)),
        })

        valid_years = sorted(valid.index.year.unique()) if not valid.empty else []
        for year in valid_years:
            year_mask = valid.index.year == year
            year_df = valid.loc[year_mask]
            yearly_rows.append({
                "Year": int(year),
                "Horizon_Bars": horizon,
                "Factor_IC": round(_safe_corr(year_df['factor'], year_df['forward_return'], 'pearson'), 4),
                "Factor_RankIC": round(_safe_corr(year_df['factor'], year_df['forward_return'], 'spearman'), 4),
                "Signal_IC": round(_safe_corr(year_df['signal'], year_df['forward_return'], 'pearson'), 4),
                "Signal_RankIC": round(_safe_corr(year_df['signal'], year_df['forward_return'], 'spearman'), 4),
                "Factor_IC_IR": round(_monthly_ic_ir(year_df['factor'], year_df['forward_return'], 'pearson'), 4),
                "Factor_RankIC_IR": round(_monthly_ic_ir(year_df['factor'], year_df['forward_return'], 'spearman'), 4),
            })

    horizon_df = pd.DataFrame(horizon_rows, columns=summary_columns)
    yearly_ic_df = pd.DataFrame(yearly_rows, columns=yearly_columns)

    df['fwd_return_1'] = market_close.shift(-1) / market_close - 1
    df["rolling_ic_90"] = factor.rolling(90).corr(df["fwd_return_1"])
    factor_rank = factor.rank()
    return_rank = df["fwd_return_1"].rank()
    df["rolling_rank_ic_90"] = factor_rank.rolling(90).corr(return_rank)

    if not horizon_df.empty:
        first_row = horizon_df.iloc[0]
        ic_summary = {
            "IC": first_row["Factor_IC"],
            "RankIC": first_row["Factor_RankIC"],
            "IC_IR": first_row["Factor_IC_IR"],
            "RankIC_IR": first_row["Factor_RankIC_IR"],
        }
    else:
        ic_summary = empty_summary
    return ic_summary, horizon_df, yearly_ic_df, df


# =========================================================
# PARAMETER SWEEP
# =========================================================
def run_backtest(df, window, entry_threshold, exit_threshold, logic, side, model, price_df=None):
    df = models_lib.choose_model(df.copy(), window, model)

    # trend_price_regime needs price during position generation, before the later PnL price join.
    # Use a temporary column so downstream joins can still add the normal 'close' column cleanly.
    if logic == 'trend_price_regime':
        if price_df is None or 'close' not in price_df.columns:
            raise ValueError("trend_price_regime requires price_df with a 'close' column")
        df['regime_close'] = price_df['close'].reindex(df.index).ffill()

    df = df.dropna().copy()
    df = entry_exit_logic_lib.signal_logic_2(df, entry_threshold, exit_threshold, logic, side)
    if 'regime_close' in df.columns:
        df = df.drop(columns=['regime_close'])
    return df


def parameter_sweep(df, base_window, base_t1, base_t2, logic, side, model, annualizer, fees, price_df, resolution):
    results = []
    windows     = [base_window + i * WINDOW_STEP for i in range(-3, 4) if base_window + i * WINDOW_STEP > 0]
    thresholds1 = [round(base_t1 + i * THRESHOLD_STEP, 4) for i in range(-3, 4)]
    thresholds2 = [round(base_t2 + i * THRESHOLD_STEP, 4) for i in range(-3, 4)]

    for w in windows:
        for t1 in thresholds1:
            for t2 in thresholds2:
                try:
                    bt = run_backtest(df.copy(), w, t1, t2, logic, side, model, price_df=price_df)
                    full_range = pd.date_range(BT_START_DATE, BT_END_DATE, freq=resolution)
                    bt = bt.reindex(full_range).ffill()
                    bt = bt.join(price_df[['close']], how='inner')
                    m, _ = calculate_metrics(bt, w, annualizer, fees)
                    results.append({
                        "SR": m["SR"], "CR": m["CR"], "MDD": m["MDD"],
                        "AR": m["AR"], "TR": m["TR"], "num_trades": m["num_trades"],
                        "rolling_window": w,
                        "threshold_1": t1, "threshold_2": t2
                    })
                except Exception as e:
                    print(f"[SWEEP] w={w} t1={t1} t2={t2} error: {e}")
    return pd.DataFrame(results)


def calculate_sweep_statistics(sweep_df):
    if sweep_df.empty or 'SR' not in sweep_df.columns:
        return pd.DataFrame({'Metric': [], 'Value': []})

    sr           = sweep_df['SR'].dropna()
    best         = sr.max()
    mean_sr      = sr.mean()
    pos_ratio    = (sr > 0).sum() / len(sr) if len(sr) > 0 else 0
    std_sr       = sr.std()
    mean_to_best = mean_sr / best if best != 0 else 0

    score  = 0
    if std_sr != 0:
        score = (1 / std_sr) * 0.4 + mean_to_best * 0.3 + pos_ratio * 0.3
    status = "PASS" if score >= 1 else "FAIL"

    rows = [
        ("best_param_sr",   round(best, 4)),
        ("mean",            round(mean_sr, 4)),
        ("median",          round(sr.median(), 4)),
        ("std",             round(std_sr, 4)),
        ("min",             round(sr.min(), 4)),
        ("max",             round(sr.max(), 4)),
        ("25th percentile", round(sr.quantile(0.25), 4)),
        ("75th percentile", round(sr.quantile(0.75), 4)),
        ("mean / best param_sr", round(mean_to_best, 4)),
        ("num of negative sr",   int((sr < 0).sum())),
        ("num of positive sr",   int((sr > 0).sum())),
        ("total num of permutation", len(sr)),
        ("num of positive sr / total", round(pos_ratio, 4)),
        ("score",  round(score, 4)),
        ("status", status),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# =========================================================
# PLOTTING
# =========================================================
def find_max_drawdown_period(df):
    df = df.copy()
    df['cumu_max'] = df['cumu'].cummax()
    df['drawdown'] = df['cumu'] - df['cumu_max']
    end_idx   = df['drawdown'].idxmin()
    start_idx = df.loc[:end_idx, 'cumu'].idxmax()
    return start_idx, end_idx


def find_longest_drawdown_period(df):
    dd_mask = df['dd'] < 0
    if not dd_mask.any():
        return None, None

    group_id = (dd_mask != dd_mask.shift()).cumsum()
    run_lengths = dd_mask.groupby(group_id).sum()
    longest_group = run_lengths[run_lengths > 0].idxmax()
    longest_mask = group_id == longest_group
    indices = df.index[longest_mask]
    return indices[0], indices[-1]


def build_signal_timeseries(df_plot, upper_threshold, lower_threshold):
    fig = go.Figure()

    if "signal" not in df_plot.columns:
        fig.add_annotation(
            text="Model signal column is unavailable.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(height=400, title="Model Signal Time Series")
        return fig

    signal = pd.to_numeric(df_plot["signal"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    fig.add_trace(
        go.Scattergl(
            x=df_plot.index,
            y=signal,
            mode="lines",
            name="Model Signal",
            line=dict(color="#1565c0", width=1.2),
            hovertemplate="Time=%{x}<br>Signal=%{y:.6g}<extra></extra>",
        )
    )

    reference_lines = [
        (upper_threshold, "Upper threshold", "#2e7d32"),
        (lower_threshold, "Lower threshold", "#c62828"),
        (0.0, "Zero", "#424242"),
    ]
    plotted_values = set()
    for value, label, color in reference_lines:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value) or value in plotted_values:
            continue
        plotted_values.add(value)
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=color,
            line_width=1,
            annotation_text=f"{label}: {value:g}",
            annotation_position="top left",
        )

    fig.update_layout(
        height=420,
        title="Model Signal Time Series",
        xaxis_title="Time",
        yaxis_title="Signal",
        showlegend=True,
    )
    return fig


def build_position_timeseries(df_plot):
    pos_df = df_plot.copy()
    pos_df["year"] = pos_df.index.year
    traces = []
    for yr in sorted(pos_df["year"].dropna().unique()):
        sub = pos_df[pos_df["year"] == yr]
        traces.append(go.Scatter(
            x=sub.index, y=sub["pos"], mode="lines", name=f"Position {yr}",
            line=dict(shape="hv")
        ))
    fig = go.Figure(traces)
    fig.update_layout(
        height=350,
        title="Position Time Series",
        xaxis_title="Time",
        yaxis_title="Position",
        legend_title="Year"
    )
    return fig


def build_cumu_pnl_histograms(df_plot):
    pnl_df = df_plot.copy()
    pnl_df["year"] = pnl_df.index.year

    yearly = pnl_df.groupby("year")["pnl"].sum().reset_index()

    half_year = pnl_df["pnl"].resample("6ME").sum().rename("pnl").reset_index()
    half_year.columns = ["period_end", "pnl"]
    half_year["label"] = pd.to_datetime(half_year["period_end"]).dt.strftime("%Y-%m")

    quarter = pnl_df["pnl"].resample("3ME").sum().rename("pnl").reset_index()
    quarter.columns = ["period_end", "pnl"]
    quarter["label"] = pd.to_datetime(quarter["period_end"]).dt.strftime("%Y-%m")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Cumulative PnL by Year", "Cumulative PnL by 6 Months", "Cumulative PnL by 3 Months"),
        vertical_spacing=0.12
    )
    fig.add_trace(go.Bar(x=yearly["year"].astype(str), y=yearly["pnl"], name="Yearly PnL"), row=1, col=1)
    fig.add_trace(go.Bar(x=half_year["label"], y=half_year["pnl"], name="6M PnL"), row=2, col=1)
    fig.add_trace(go.Bar(x=quarter["label"], y=quarter["pnl"], name="3M PnL"), row=3, col=1)
    fig.update_layout(height=1100, title="Cumulative PnL Histograms", showlegend=False)
    fig.update_yaxes(title_text="PnL", row=1, col=1)
    fig.update_yaxes(title_text="PnL", row=2, col=1)
    fig.update_yaxes(title_text="PnL", row=3, col=1)
    return fig


def build_monthly_cumu_pnl_heatmap(df_plot):
    heat_df = df_plot.copy()
    monthly = heat_df["pnl"].resample("M").sum().to_frame("monthly_pnl")
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = monthly.pivot(index="year", columns="month", values="monthly_pnl").reindex(columns=month_order)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=[str(y) for y in pivot.index],
        text=np.round(pivot.values, 4),
        texttemplate="%{text}",
        hovertemplate="Year %{y}<br>Month %{x}<br>PnL %{z:.4f}<extra></extra>",
        colorscale="Greens"
    ))
    fig.update_layout(
        height=450,
        title="Monthly Cumulative PnL Heatmap",
        xaxis_title="Month",
        yaxis_title="Year"
    )
    return fig


def build_parameter_sweep_heatmap(sweep_df, center_window, metric="SR"):
    if sweep_df.empty:
        return go.Figure()

    heat_df = sweep_df[sweep_df["rolling_window"] == center_window].copy()
    pivot = heat_df.pivot(index="threshold_2", columns="threshold_1", values=metric).sort_index(ascending=True)

    if pivot.empty:
        return go.Figure()

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[round(v, 4) for v in pivot.columns],
            y=[round(v, 4) for v in pivot.index],
            text=np.round(pivot.values, 4),
            texttemplate="%{text}",
            hovertemplate="threshold_2 %{y}<br>threshold_1 %{x}<br>" + metric + " %{z:.4f}<extra></extra>",
            colorscale="Greens",
        )
    )
    fig.update_layout(
        height=450,
        title=f"Parameter Sweep Heatmap ({metric}) - parameter window={center_window}",
        xaxis_title="threshold_1",
        yaxis_title="threshold_2",
    )
    return fig


def evaluate_parameter_heatmap_windows(base_df, thresholds1, thresholds2, logic, side, model, annualizer, fees, price_df, resolution, center_window, metric="SR"):
    heatmap_figs = []
    window_offsets = [0]
    for step in range(1, WINDOW_HEATMAP_STEPS + 1):
        window_offsets.extend([-step, step])
    target_windows = [
        center_window + offset * WINDOW_HEATMAP_DELTA
        for offset in window_offsets
        if center_window + offset * WINDOW_HEATMAP_DELTA > 0
    ]

    for window in target_windows:
        rows = []
        for t1 in thresholds1:
            for t2 in thresholds2:
                try:
                    bt = run_backtest(base_df.copy(), window, t1, t2, logic, side, model, price_df=price_df)
                    full_range = pd.date_range(BT_START_DATE, BT_END_DATE, freq=resolution)
                    bt = bt.reindex(full_range).ffill()
                    bt = bt.join(price_df[['close']], how='inner')
                    metrics, _ = calculate_metrics(bt, window, annualizer, fees)
                    rows.append({
                        "threshold_1": t1,
                        "threshold_2": t2,
                        metric: metrics.get(metric)
                    })
                except Exception:
                    continue

        window_df = pd.DataFrame(rows)
        if window_df.empty:
            continue

        pivot = window_df.pivot(index="threshold_2", columns="threshold_1", values=metric).sort_index(ascending=True)
        if pivot.empty:
            continue

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[round(v, 4) for v in pivot.columns],
                y=[round(v, 4) for v in pivot.index],
                text=np.round(pivot.values, 4),
                texttemplate="%{text}",
                hovertemplate="threshold_2 %{y}<br>threshold_1 %{x}<br>" + metric + " %{z:.4f}<extra></extra>",
                colorscale="Greens",
            )
        )
        label = "parameter window" if window == center_window else f"parameter window {'+' if window > center_window else '-'} {abs(window - center_window)}"
        fig.update_layout(
            height=420,
            title=f"Parameter Sweep Heatmap ({metric}) - {label} ({window})",
            xaxis_title="threshold_1",
            yaxis_title="threshold_2",
        )
        heatmap_figs.append({"label": f"{label} ({window})", "fig": fig})

    return heatmap_figs


def build_rolling_ic_plot(df_plot):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["rolling_ic_90"],
            mode="lines",
            name="Rolling 90 IC",
            line=dict(color="teal"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["rolling_rank_ic_90"],
            mode="lines",
            name="Rolling 90 Rank IC",
            line=dict(color="orange"),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        height=400,
        title="Rolling IC",
        xaxis_title="Time",
        yaxis_title="IC",
    )
    return fig


def build_multi_horizon_ic_plot(ic_horizon_df):
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
        subplot_titles=("IC by Forward Horizon", "Entry-Signal Forward Performance"),
    )
    if ic_horizon_df.empty:
        fig.add_annotation(text="No multi-horizon IC data available.", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=480, title="Multi-Horizon IC and Signal Performance")
        return fig

    horizons = ic_horizon_df['Horizon_Bars']
    for column, label, color, dash in [
        ('Factor_IC', 'Factor IC', '#1565c0', 'solid'),
        ('Factor_RankIC', 'Factor RankIC', '#26a69a', 'dash'),
        ('Signal_IC', 'Signal IC', '#ef6c00', 'solid'),
        ('Signal_RankIC', 'Signal RankIC', '#8e24aa', 'dash'),
    ]:
        fig.add_trace(
            go.Scatter(
                x=horizons,
                y=ic_horizon_df[column],
                mode='lines+markers',
                name=label,
                line=dict(color=color, dash=dash),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=horizons,
            y=ic_horizon_df['Entry_Event_Mean_Return'],
            name='Entry Event Mean Return',
            marker_color='#2e7d32',
            opacity=0.70,
        ),
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=ic_horizon_df['Entry_Event_Hit_Rate'],
            mode='lines+markers',
            name='Entry Event Hit Rate',
            line=dict(color='#c62828'),
        ),
        row=1,
        col=2,
        secondary_y=True,
    )
    fig.add_hline(y=0, line_dash='dash', line_color='#424242', row=1, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='#424242', row=1, col=2, secondary_y=False)
    fig.add_hline(y=0.5, line_dash='dot', line_color='#c62828', row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text='Forward Horizon (bars)', row=1, col=1)
    fig.update_xaxes(title_text='Forward Horizon (bars)', row=1, col=2)
    fig.update_yaxes(title_text='Correlation', row=1, col=1)
    fig.update_yaxes(title_text='Signed Forward Return', row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text='Hit Rate', range=[0, 1], row=1, col=2, secondary_y=True)
    fig.update_layout(height=500, title='Multi-Horizon IC and Signal Performance', title_x=0.5)
    return fig


def build_regime_strategy_comparison_plot(curve_df):
    fig = go.Figure()
    primary_simulations = ['Baseline', '200D Trend Filter', 'High Vol Only', 'Low Vol Only']
    colors = {
        'Baseline': '#1565c0',
        '200D Trend Filter': '#2e7d32',
        'High Vol Only': '#c62828',
        'Low Vol Only': '#8e24aa',
    }
    plot_columns = [column for column in primary_simulations if column in curve_df.columns]
    if not plot_columns:
        plot_columns = list(curve_df.columns)
    for column in plot_columns:
        fig.add_trace(
            go.Scatter(
                x=curve_df.index,
                y=curve_df[column],
                mode='lines',
                name=column,
                line=dict(color=colors.get(column)),
            )
        )
    if curve_df.empty:
        fig.add_annotation(text='No regime simulation curves available.', x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        height=460,
        title='Regime-Gated Strategy Simulation',
        xaxis_title='Time',
        yaxis_title='Cumulative PnL',
    )
    return fig


def build_long_short_cumu_plot(df_plot, annualizer):
    long_short_df, long_cumu, short_cumu = build_long_short_summary(df_plot, annualizer)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=long_cumu, mode="lines", name="Long Only", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=short_cumu, mode="lines", name="Short Only", line=dict(color="firebrick")))
    fig.update_layout(
        height=420,
        title="Long / Short Split Cumulative PnL",
        xaxis_title="Time",
        yaxis_title="Cumulative PnL",
    )
    return fig, long_short_df


def add_regime_background(fig, x_index, state_series, row, col, positive_color, negative_color):
    if state_series is None or state_series.empty:
        return

    state_series = state_series.dropna().astype(bool)
    if state_series.empty:
        return

    group_id = (state_series != state_series.shift()).cumsum()
    for _, segment in state_series.groupby(group_id):
        color = positive_color if bool(segment.iloc[0]) else negative_color
        fig.add_vrect(
            x0=segment.index[0],
            x1=segment.index[-1],
            fillcolor=color,
            opacity=0.10,
            layer="below",
            line_width=0,
            row=row,
            col=col,
        )


def build_regime_classification_plot(regime_plot_df):
    available_rows = []
    if not regime_plot_df.empty:
        available_rows.append(("BTC Trend (200D MA)", "btc_close", "btc_ma_200d"))
        if 'btc_vol_30d' in regime_plot_df.columns:
            available_rows.append(("BTC Volatility (30D)", "btc_vol_30d", "vol_median"))

    if not available_rows:
        fig = go.Figure()
        fig.update_layout(height=300, title="Regime Classification")
        return fig

    fig = make_subplots(
        rows=len(available_rows),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[item[0] for item in available_rows],
    )

    current_row = 1
    for title, value_col, threshold_col in available_rows:
        sub_df = regime_plot_df.dropna(subset=[value_col]).copy()
        if sub_df.empty:
            current_row += 1
            continue

        if value_col == 'btc_close':
            fig.add_trace(go.Scatter(x=sub_df.index, y=sub_df[value_col], mode="lines", name="BTC Close", line=dict(color="black")), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=sub_df.index, y=sub_df['btc_ma_200d'], mode="lines", name="BTC 200D MA", line=dict(color="green", dash="dash")), row=current_row, col=1)
            add_regime_background(fig, sub_df.index, sub_df['trend_above_200d'], current_row, 1, "#2e7d32", "#c62828")
        elif value_col == 'btc_vol_30d':
            fig.add_trace(go.Scatter(x=sub_df.index, y=sub_df[value_col], mode="lines", name="BTC 30D Vol", line=dict(color="#1565c0")), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=sub_df.index, y=sub_df[threshold_col], mode="lines", name="Past 365D Vol Median", line=dict(color="#ef6c00", dash="dash")), row=current_row, col=1)
            add_regime_background(fig, sub_df.index, sub_df['high_vol'], current_row, 1, "#ef6c00", "#90caf9")

        current_row += 1

    fig.update_layout(height=max(320, 260 * len(available_rows)), title="Regime Classification")
    return fig


def build_transformation_qq_plot(df_plot, max_points=5000):
    """Show the factor transformation and signal normality side by side."""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Transformation: Raw Factor vs Model Signal",
            "Q-Q Plot: Transformed Signal vs Normal",
        ),
    )

    required_columns = ["factor", "signal"]
    if any(column not in df_plot.columns for column in required_columns):
        fig.add_annotation(
            text="Raw factor or transformed signal is unavailable.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(height=500, title="Factor Transformation Diagnostics")
        return fig

    diagnostic_df = df_plot[required_columns].copy()
    for column in required_columns:
        diagnostic_df[column] = pd.to_numeric(diagnostic_df[column], errors="coerce")
    diagnostic_df = diagnostic_df.replace([np.inf, -np.inf], np.nan).dropna()

    if diagnostic_df.empty:
        fig.add_annotation(
            text="No finite factor/signal observations are available.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(height=500, title="Factor Transformation Diagnostics")
        return fig

    if len(diagnostic_df) > max_points:
        sample_positions = np.linspace(0, len(diagnostic_df) - 1, max_points, dtype=int)
        diagnostic_df = diagnostic_df.iloc[sample_positions]

    fig.add_trace(
        go.Scattergl(
            x=diagnostic_df["factor"],
            y=diagnostic_df["signal"],
            mode="markers",
            name="Factor to Signal",
            marker=dict(color="#1565c0", size=5, opacity=0.45),
            hovertemplate="Raw factor=%{x:.6g}<br>Signal=%{y:.6g}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    theoretical_quantiles, ordered_signal = probplot(
        diagnostic_df["signal"].to_numpy(),
        dist="norm",
        fit=False,
    )
    theoretical_quantiles = np.asarray(theoretical_quantiles, dtype=float)
    ordered_signal = np.asarray(ordered_signal, dtype=float)

    fig.add_trace(
        go.Scattergl(
            x=theoretical_quantiles,
            y=ordered_signal,
            mode="markers",
            name="Signal Quantiles",
            marker=dict(color="#ef6c00", size=5, opacity=0.55),
            hovertemplate="Normal quantile=%{x:.4f}<br>Signal quantile=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    if len(theoretical_quantiles) >= 2:
        slope, intercept = np.polyfit(theoretical_quantiles, ordered_signal, 1)
        reference_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
        reference_y = intercept + slope * reference_x
        fig.add_trace(
            go.Scatter(
                x=reference_x,
                y=reference_y,
                mode="lines",
                name="Normal Reference",
                line=dict(color="#424242", dash="dash"),
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Raw Factor", row=1, col=1)
    fig.update_yaxes(title_text="Model Signal", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Normal Quantile", row=1, col=2)
    fig.update_yaxes(title_text="Ordered Signal Quantile", row=1, col=2)
    fig.update_layout(
        height=500,
        title="Factor Transformation Diagnostics",
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
    )
    return fig


def create_combined_plot(df_plot, annualizer):
    fig = make_subplots(
        rows=6, cols=1,
        vertical_spacing=0.08,
        subplot_titles=(
            "Cumulative PnL and Price with Max Drawdown Period",
            "Drawdown", "Full Period PnL", "Final Year PnL",
            "Roll-180-Day SR", "Roll-365-Day SR"
        ),
        specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * 5,
        row_heights=[0.3, 0.14, 0.14, 0.14, 0.14, 0.14]
    )

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['cumu'],  name='Cumulative PnL', line=dict(color='blue')),  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], name='Close Price',    line=dict(color='black')), row=1, col=1, secondary_y=True)

    start_idx, end_idx = find_max_drawdown_period(df_plot)
    fig.add_vrect(x0=start_idx, x1=end_idx, fillcolor="red", opacity=0.2,
                  layer="below", line_width=0,
                  annotation_text="Max Drawdown", annotation_position="top left",
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['dd'], fill='tozeroy',
                             line=dict(color='red'), name='Drawdown'), row=2, col=1)
    ldd_start_idx, ldd_end_idx = find_longest_drawdown_period(df_plot)
    if ldd_start_idx is not None and ldd_end_idx is not None:
        fig.add_vrect(x0=ldd_start_idx, x1=ldd_end_idx, fillcolor="orange", opacity=0.2,
                      layer="below", line_width=0,
                      annotation_text="Longest DD", annotation_position="top left",
                      row=2, col=1)

    pnl_nonzero = df_plot.loc[df_plot['pnl'] != 0, 'pnl']
    fig.add_trace(go.Histogram(x=pnl_nonzero, nbinsx=500, marker_color='green', name='Full PnL'), row=3, col=1)

    final_year = df_plot.tail(annualizer)
    final_year = final_year[final_year['pnl'] != 0]
    fig.add_trace(go.Histogram(x=final_year['pnl'], nbinsx=365, marker_color='blue', name='Final Year PnL'), row=4, col=1)

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['rolling_sharpe_180'], line=dict(color='purple'), name='Roll-180d SR'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['rolling_sharpe_365'], line=dict(color='darkgreen'), name='Roll-365d SR'), row=6, col=1)

    for row in [5, 6]:
        for y0, color in [(2, 'green'), (0, 'black')]:
            fig.add_shape(type='line', x0=df_plot.index[0], x1=df_plot.index[-1],
                          y0=y0, y1=y0, line=dict(color=color, dash='dash'), row=row, col=1)

    fig.update_layout(height=1800, showlegend=True,
                      title_text="Backtest Results Dashboard", title_x=0.5)
    fig.update_yaxes(title_text="Cumulative PnL", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Close Price",    row=1, col=1, secondary_y=True)
    return fig


# =========================================================
# HTML BUILDER
# =========================================================
import pandas as pd

def build_html(alpha, metrics_bt, metrics_ft, metrics_vt, yearly_metrics_df, ic_summary, ic_horizon_df, yearly_ic_df, sweep_df, sweep_stats_df,
               vt_stress_metrics_html, long_short_df, regime_df, fig_main, fig_position, fig_hist, fig_heatmap, fig_sweep_heatmaps, fig_ic,
               fig_long_short, fig_regime_classification, fig_regime_strategy, fig_transformation_qq, fig_signal, window, t1, t2):

    alpha_id = alpha['alpha_id']

    metric_keys = ['SR', 'CR', 'MDD', 'AR', 'TR', 'VAR', 'win_rate', 'num_trades', 'LDD', 'TPI', 'BaH', 't_statistic', 'p_value']
    metrics_table_data = {
        'Metric':     metric_keys,
        'BT Period':  [metrics_bt.get(k, '') for k in metric_keys],
        'FT Period':  [metrics_ft.get(k, '') for k in metric_keys],
        'VT Period':  [metrics_vt.get(k, '') for k in metric_keys],
    }

    metrics_html     = pd.DataFrame(metrics_table_data).to_html(index=False, classes='metrics-table')
    yearly_metrics_html = render_column_heatmap_table(yearly_metrics_df, classes='metrics-table', precision=3)
    ic_html = f'<div class="wide-table">{render_column_heatmap_table(ic_horizon_df, classes="metrics-table")}</div>'
    yearly_ic_html = f'<div class="wide-table">{render_column_heatmap_table(yearly_ic_df, classes="metrics-table")}</div>'
    long_short_html = render_column_heatmap_table(long_short_df, classes='metrics-table')
    regime_html = f'<div class="wide-table">{render_column_heatmap_table(regime_df, classes="metrics-table")}</div>'
    sweep_html       = sweep_df.to_html(index=False, classes='sweep-table')
    sweep_stats_html = sweep_stats_df.to_html(index=False, classes='stats-table')
    plotly_main_html = fig_main.to_html(full_html=False, include_plotlyjs='cdn')
    plotly_position_html = fig_position.to_html(full_html=False, include_plotlyjs=False)
    plotly_hist_html = fig_hist.to_html(full_html=False, include_plotlyjs=False)
    plotly_heatmap_html = fig_heatmap.to_html(full_html=False, include_plotlyjs=False)
    plotly_ic_html = fig_ic.to_html(full_html=False, include_plotlyjs=False)
    plotly_long_short_html = fig_long_short.to_html(full_html=False, include_plotlyjs=False)
    plotly_regime_classification_html = fig_regime_classification.to_html(full_html=False, include_plotlyjs=False)
    plotly_regime_strategy_html = fig_regime_strategy.to_html(full_html=False, include_plotlyjs=False)
    plotly_transformation_qq_html = fig_transformation_qq.to_html(full_html=False, include_plotlyjs=False)
    plotly_signal_html = fig_signal.to_html(full_html=False, include_plotlyjs=False)

    if fig_sweep_heatmaps:
        dropdown_options = "".join(
            f'<option value="sweep-heatmap-{idx}">{item["label"]}</option>'
            for idx, item in enumerate(fig_sweep_heatmaps)
        )
        sweep_heatmap_blocks = "".join(
            f'<div id="sweep-heatmap-{idx}" class="sweep-heatmap-panel" style="display: {"block" if idx == 0 else "none"};">'
            f'{item["fig"].to_html(full_html=False, include_plotlyjs=False)}'
            f'</div>'
            for idx, item in enumerate(fig_sweep_heatmaps)
        )
        plotly_sweep_heatmap_html = f"""
        <div class="table-container">
          <h2>Parameter Sweep Heatmap</h2>
          <div style="padding: 0 12px 12px 12px;">
            <label for="sweep-heatmap-select"><strong>Window slice:</strong></label>
            <select id="sweep-heatmap-select" style="margin-left: 8px; padding: 4px 8px;">
              {dropdown_options}
            </select>
          </div>
          {sweep_heatmap_blocks}
        </div>
        """
    else:
        plotly_sweep_heatmap_html = ""

    info_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in {
            "alpha_id": alpha_id,
            "custom_id": alpha.get('custom_id'),
            "alpha_formula": alpha.get('alpha_formula'),
            "manual_alpha_formula": alpha.get("manual_alpha_formula"),
            "data_preprocessing": alpha.get("data_preprocessing"),
            "preprocessing_window": alpha.get("preprocessing_window"),
            "data_asset": alpha.get("data_asset"),
            "trade_asset": alpha.get("trade_asset"),
            "shift_backtest_candle_minute": alpha.get("shift_backtest_candle_minute", 0),
            "fees": alpha.get("fees"),
            "model": alpha.get('model'),
            "entry_exit_logic": alpha.get('entry_exit_logic'),
            "timeframe": alpha.get('timeframe'),
            "window": window,
            "threshold_1": t1,
            "threshold_2": t2,
            "datasource_structure": alpha.get("datasource_structure"),
        }.items()
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <title>Backtest Report — {alpha_id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
    .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; height: 100vh; }}
    .plots {{ grid-column: 1; overflow-y: auto; }}
    
    .tables {{ 
        grid-column: 2; 
        overflow-y: auto; 
        padding: 10px; 
        max-width: 800px; 
    }}

    .table-container {{ margin-bottom: 30px; }}
    .wide-table {{ overflow-x: auto; width: 100%; }}
    .wide-table .metrics-table {{ min-width: 1450px; table-layout: auto; }}

    .metrics-table, .sweep-table, .stats-table, .info-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      table-layout: fixed;
    }}

    .info-table td:first-child {{
        width: 30%;
        font-weight: bold;
        background-color: #f9f9f9;
    }}

    .info-table td:last-child {{
        width: 70%;
        word-wrap: break-word;
        text-align: left;
    }}

    .metrics-table th, .metrics-table td,
    .sweep-table th,   .sweep-table td,
    .stats-table th,   .stats-table td,
    .info-table th,    .info-table td {{
      border: 1px solid #ddd;
      padding: 8px 10px;
      text-align: center;
      vertical-align: top;
      word-break: break-word;
    }}

    .metrics-table th, .sweep-table th, .stats-table th, .info-table th {{
      background-color: #4CAF50;
      color: white;
    }}

    h2 {{ text-align: center; color: #333; }}

    .sweep-table-container {{ overflow-x: auto; }}

    .pass {{ color: green; font-weight: bold; }}
    .fail {{ color: red; font-weight: bold; }}
  </style>
</head>

<body>
<div class="dashboard">
  <div class="plots">
      <h2>Backtest Plots — {alpha_id}</h2>
      {plotly_main_html}
      {plotly_transformation_qq_html}
      {plotly_signal_html}
      {plotly_position_html}
      {plotly_hist_html}
      {plotly_heatmap_html}
      {plotly_sweep_heatmap_html}
      {plotly_ic_html}
      {plotly_long_short_html}
      {plotly_regime_strategy_html}
      {plotly_regime_classification_html}
  </div>

  <div class="tables">
      <div class="table-container">
        <h2>Alpha Info</h2>
        <table class="info-table">
          <thead>
            <tr>
              <th>Field</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {info_rows}
          </tbody>
        </table>
      </div>

      <h2>Overfitting Test</h2>
      <div class="table-container">
        <h2>Metrics (BT vs FT vs VT)</h2>
        {metrics_html}
      </div>

      <div class="table-container">
        <h2>Parameter Sweep Statistics</h2>
        {sweep_stats_html}
      </div>

      <h2>Robustness Test</h2>
      <div class="table-container">
        <h2>Metrics (Full-Date Stress)</h2>
        {vt_stress_metrics_html}
      </div>

      <div class="table-container">
        <h2>Yearly Metrics</h2>
        {yearly_metrics_html}
      </div>

      <div class="table-container">
        <h2>Long / Short Split</h2>
        {long_short_html}
      </div>

      <h2>Alpha Decay / Regime Test</h2>
      <div class="table-container">
        <h2>Regime-Gated Strategy Simulations</h2>
        {regime_html}
      </div>

      <div class="table-container">
        <h2>Multi-Horizon IC & Entry Performance</h2>
        {ic_html}
      </div>

      <div class="table-container">
        <h2>Yearly Multi-Horizon IC</h2>
        {yearly_ic_html}
      </div>
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {{
  document.querySelectorAll('.stats-table tbody tr').forEach(row => {{
    const cells = row.querySelectorAll('td');
    if (cells.length > 0 && cells[0].textContent === 'status') {{
        cells[1].classList.add(
            cells[1].textContent === 'PASS' ? 'pass' : 'fail'
        );
    }}
  }});

  const sweepSelect = document.getElementById('sweep-heatmap-select');
  if (sweepSelect) {{
    sweepSelect.addEventListener('change', function() {{
      document.querySelectorAll('.sweep-heatmap-panel').forEach(panel => {{
        panel.style.display = 'none';
      }});
      const selected = document.getElementById(this.value);
      if (selected) {{
        selected.style.display = 'block';
      }}
    }});
  }}
}});
</script>

</body>
</html>"""


def get_sweep_status(sweep_stats_df):
    if sweep_stats_df.empty:
        return ""
    row = sweep_stats_df[sweep_stats_df["Metric"] == "status"]
    if row.empty:
        return ""
    return row["Value"].iloc[0]


def build_summary_html(summary_rows):
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_table_html = "<p>No reports were generated.</p>"
    else:
        summary_df["alpha_sort_num"] = summary_df["alpha_id"].str.extract(r"(\d+)$").astype(float)
        summary_df = summary_df.sort_values(by=["alpha_sort_num", "alpha_id"], ascending=[True, True]).reset_index(drop=True)
        display_df = summary_df.copy()
        display_df["Report"] = display_df.apply(
            lambda row: f'<a href="{row["report_file"]}">{row["alpha_id"]}</a>',
            axis=1
        )
        display_df["remark"] = display_df["remark"].apply(
            lambda x: '<span class="remark-failed">FAILED</span>' if x == "FAILED" else '<span class="remark-passed">PASSED</span>'
        )
        display_df = display_df[
            [
                "Report", "custom_id", "model", "timeframe", "window", "threshold_1", "threshold_2",
                "BT_SR", "FT_SR", "VT_SR", "BT_MDD", "FT_MDD", "VT_MDD",
                "IC", "RankIC", "rolling_180d_sr", "current_drawdown", "sweep_status", "remark", "remark_reason"
            ]
        ].rename(
            columns={
                "custom_id": "Custom ID",
                "model": "Model",
                "timeframe": "Timeframe",
                "window": "Window",
                "threshold_1": "Threshold 1",
                "threshold_2": "Threshold 2",
                "BT_SR": "BT SR",
                "FT_SR": "FT SR",
                "VT_SR": "VT SR",
                "BT_MDD": "BT MDD",
                "FT_MDD": "FT MDD",
                "VT_MDD": "VT MDD",
                "IC": "IC",
                "RankIC": "RankIC",
                "rolling_180d_sr": "Rolling 180d SR",
                "current_drawdown": "Current Drawdown",
                "sweep_status": "Sweep Status",
                "remark": "Remark",
                "remark_reason": "Reason",
            }
        )
        summary_table_html = display_df.to_html(index=False, classes="summary-table", escape=False)

    return f"""<!DOCTYPE html>
<html>
<head>
  <title>Alpha Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; background: #f5f7f5; color: #222; }}
    .container {{ max-width: 1600px; margin: 0 auto; }}
    h1 {{ margin-top: 0; }}
    .summary-wrap {{ overflow-x: auto; background: white; padding: 16px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
    .summary-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .summary-table th, .summary-table td {{ border: 1px solid #d9e2d9; padding: 8px 10px; text-align: center; }}
    .summary-table th {{ background: #2e7d32; color: white; position: sticky; top: 0; }}
    .summary-table tr:nth-child(even) {{ background: #f8fbf8; }}
    a {{ color: #1b5e20; text-decoration: none; font-weight: 600; }}
    a:hover {{ text-decoration: underline; }}
    .remark-failed {{ color: #c62828; font-weight: 700; }}
    .remark-passed {{ color: #2e7d32; font-weight: 700; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Alpha Summary</h1>
    <div class="summary-wrap">
      {summary_table_html}
    </div>
  </div>
</body>
</html>"""

# =========================================================
# MAIN
# =========================================================
def main():
    with open(JSON_FILE, "r", encoding="utf-8-sig") as f:
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

    if TARGET_ALPHA_ID is not None:
        target_ids = TARGET_ALPHA_ID if isinstance(TARGET_ALPHA_ID, (list, tuple, set)) else [TARGET_ALPHA_ID]
        target_ids = set(target_ids)
        alphas = [alpha for alpha in alphas if alpha.get("alpha_id") in target_ids]
        if not alphas:
            raise ValueError(f"Alpha ID not found: {TARGET_ALPHA_ID}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    PRICE_CACHE = {}
    summary_rows = []

    for alpha in alphas:
        alpha_id = alpha.get("alpha_id", "unknown")
        print(f"\n{'='*60}\nProcessing {alpha_id} ...")

        try:
            # ---- Parse datasource_structure ----
            ds_raw = alpha.get("datasource_structure", "{}")
            try:
                ds = json.loads(ds_raw.replace("'", '"'))
            except Exception:
                ds = ast.literal_eval(ds_raw)

            # ---- Load & merge data ----
            merged_df = None
            skip_alpha = False
            for key, val in ds.items():
                topic  = val.get("topic")
                col    = val.get("column")
                df_src = load_csv(topic, col, key)
                if df_src is None:
                    skip_alpha = True
                    break
                df_src = df_src.dropna(subset=[key])
                merged_df = df_src if merged_df is None else pd.merge(merged_df, df_src, on='time', how='inner')

            if skip_alpha:
                print(f"[SKIP] Datasource QC failed or missing for {alpha_id}")
                continue

            if merged_df is None:
                print(f"[SKIP] No data loaded for {alpha_id}")
                continue

            merged_df = merged_df.drop(columns=[c for c in merged_df.columns if 'start_time' in c])
            merged_df = merged_df.set_index('time')

            # ---- Compute alpha signal ----
            alpha_formula = alpha.get("manual_alpha_formula") or alpha.get("alpha_formula", "")
            merged_df, rewritten = apply_diff_from_formula(merged_df, alpha_formula)
            for c in merged_df.columns:
                merged_df[c] = pd.to_numeric(merged_df[c], errors='coerce')

            signal = compute_alpha_signal(merged_df, rewritten)
            if signal is None:
                print(f"[SKIP] Signal computation failed for {alpha_id}")
                continue

            merged_df['factor'] = signal
            merged_df['factor'] = merged_df['factor'].replace([np.inf, -np.inf], np.nan)
            merged_df['factor'] = (merged_df['factor'].round(10)
                                   if abs(merged_df['factor'].min()) < 1.0
                                   else merged_df['factor'].round(10))

            if alpha.get("data_preprocessing") == "diff":
                merged_df['factor'] = merged_df['factor'].diff()
                merged_df = merged_df.dropna(subset=['factor'])

            merged_df = merged_df.dropna()

            # ---- Setup params ----
            timeframe  = alpha.get("timeframe", "1h")
            resolution = TIMEFRAME_MAP.get(timeframe)
            annualizer = METRIC_ANNUALIZER_MAP.get(timeframe, 365 * 24)
            window     = int(alpha.get("window", alpha.get("rolling_window_1", 0)))
            model      = alpha.get("model")
            fees       = float(alpha.get("fees", 0.035)) / 100

            entry_exit = alpha.get("entry_exit_logic", "").lower()
            if "_long" in entry_exit:
                logic, side = entry_exit.replace("_long", ""), "long"
            elif "_short" in entry_exit:
                logic, side = entry_exit.replace("_short", ""), "short"
            else:
                logic, side = entry_exit, "both"

            # t1 = float(alpha.get("long_entry_threshold",  0))
            # t2 = float(alpha.get("short_entry_threshold", 0))
            long_entry_threshold = float(alpha.get("long_entry_threshold", 0.0))
            long_exit_threshold  = float(alpha.get("long_exit_threshold", 0.0))
            short_entry_threshold = float(alpha.get("short_entry_threshold", 0.0))
            short_exit_threshold  = float(alpha.get("short_exit_threshold", 0.0))

            t1 = max(long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold)
            t2 = min(long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold)
            window, t1, t2 = apply_param_overrides(window, t1, t2)

            # ---- Price ----
            trade_asset  = alpha.get("trade_asset", "BTC")
            price_delay  = int(alpha.get("shift_backtest_candle_minute", 0))
            price_key    = (trade_asset, timeframe, price_delay)
            if price_key not in PRICE_CACHE:
                PRICE_CACHE[price_key] = load_price_df(trade_asset, timeframe, price_delay)
            price_df = PRICE_CACHE[price_key]
            if price_df is None:
                print(f"[SKIP] No price data for {alpha_id}")
                continue

            # Compute model/position on full available history before slicing periods.
            # This keeps rolling/EMA warmup consistent with live alpha generation.
            signal_df = run_backtest(merged_df.copy(), window, t1, t2, logic, side, model, price_df=price_df)

            # ---- BT ----
            df_bt = signal_df.loc[BT_START_DATE:BT_END_DATE].copy()
            full_range_bt = pd.date_range(BT_START_DATE, BT_END_DATE, freq=resolution)
            df_bt = df_bt.reindex(full_range_bt).ffill()
            df_bt = df_bt.join(price_df[['close']], how='inner')

            # ---- FT ----
            df_ft = signal_df.loc[FT_START_DATE:FT_END_DATE].copy()
            full_range_ft = pd.date_range(FT_START_DATE, FT_END_DATE, freq=resolution)
            df_ft = df_ft.reindex(full_range_ft).ffill()
            df_ft = df_ft.join(price_df[['close']], how='inner')

            # ---- VT ----
            df_vt = signal_df.loc[VT_START_DATE:VT_END_DATE].copy()
            full_range_vt = pd.date_range(VT_START_DATE, VT_END_DATE, freq=resolution)
            df_vt = df_vt.reindex(full_range_vt).ffill()
            df_vt = df_vt.join(price_df[['close']], how='inner')

            # ---- Full chart window ----
            df_full = signal_df.loc[START_DATE:END_DATE].copy()
            full_range_all = pd.date_range(START_DATE, END_DATE, freq=resolution)
            df_full = df_full.reindex(full_range_all).ffill()
            df_full = df_full.join(price_df[['close']], how='inner')

            metrics_ft, df_ft = calculate_metrics(df_ft, window, annualizer, fees)
            metrics_vt, df_vt = calculate_metrics(df_vt, window, annualizer, fees)
            metrics_bt, df_bt = calculate_metrics(df_bt, window, annualizer, fees, ft_df=df_ft)
            metrics_full, df_full = calculate_metrics(df_full, window, annualizer, fees)
            yearly_metrics_df = calculate_yearly_metrics(df_full, window, annualizer, fees)
            btc_price_key = ("BTC", timeframe, 0)
            if btc_price_key not in PRICE_CACHE:
                PRICE_CACHE[btc_price_key] = load_price_df("BTC", timeframe, 0)
            btc_price_df = PRICE_CACHE[btc_price_key]
            ic_summary, ic_horizon_df, yearly_ic_df, df_full = calculate_ic_metrics(df_full, btc_price_df)

            # ---- Sweep ----
            sweep_df    = parameter_sweep(merged_df.loc[BT_START_DATE:BT_END_DATE].copy(),
                                          window, t1, t2, logic, side, model,
                                          annualizer, fees, price_df, resolution)
            sweep_stats = calculate_sweep_statistics(sweep_df)
            long_short_df, _, _ = build_long_short_summary(df_full, annualizer)
            regime_df, regime_plot_df, regime_curve_df = calculate_regime_robustness(
                df_full, btc_price_df, annualizer, window, fees, t1, t2, logic, side
            )

            stress_fee = 0.06 / 100
            metrics_full_fee_006, _ = calculate_metrics(df_full.copy(), window, annualizer, stress_fee)

            doubled_price_delay = price_delay * 2
            doubled_price_key = (trade_asset, timeframe, doubled_price_delay)
            if doubled_price_key not in PRICE_CACHE:
                PRICE_CACHE[doubled_price_key] = load_price_df(trade_asset, timeframe, doubled_price_delay)
            doubled_price_df = PRICE_CACHE[doubled_price_key]
            if doubled_price_df is None:
                metrics_full_shift_x2 = {k: np.nan for k in ['SR', 'CR', 'MDD', 'AR', 'TR', 'VAR', 'win_rate', 'num_trades', 'LDD', 'TPI', 'BaH']}
            else:
                df_full_shift_x2 = signal_df.loc[START_DATE:END_DATE].copy()
                df_full_shift_x2 = df_full_shift_x2.reindex(full_range_all).ffill()
                df_full_shift_x2 = df_full_shift_x2.join(doubled_price_df[['close']], how='inner')
                metrics_full_shift_x2, _ = calculate_metrics(df_full_shift_x2, window, annualizer, fees)

            vt_stress_metrics_html = build_stress_metrics_table(
                {
                    'Full Base': metrics_full,
                    'Full Fees 0.06': metrics_full_fee_006,
                    'Full Shift x2': metrics_full_shift_x2,
                }
            )

            # ---- Plot & HTML ----
            fig_main = create_combined_plot(df_full, annualizer)
            fig_transformation_qq = build_transformation_qq_plot(df_full)
            fig_signal = build_signal_timeseries(df_full, t1, t2)
            fig_position = build_position_timeseries(df_full)
            fig_hist = build_cumu_pnl_histograms(df_full)
            fig_heatmap = build_monthly_cumu_pnl_heatmap(df_full)
            fig_long_short, _ = build_long_short_cumu_plot(df_full, annualizer)
            fig_regime_strategy = build_regime_strategy_comparison_plot(regime_curve_df)
            fig_regime_classification = build_regime_classification_plot(regime_plot_df)
            thresholds1 = [
                round(t1 + offset * THRESHOLD_STEP, 4)
                for offset in range(-THRESHOLD_HEATMAP_STEPS, THRESHOLD_HEATMAP_STEPS + 1)
            ]
            thresholds2 = [
                round(t2 + offset * THRESHOLD_STEP, 4)
                for offset in range(-THRESHOLD_HEATMAP_STEPS, THRESHOLD_HEATMAP_STEPS + 1)
            ]
            fig_sweep_heatmaps = evaluate_parameter_heatmap_windows(
                merged_df.loc[BT_START_DATE:BT_END_DATE].copy(),
                thresholds1,
                thresholds2,
                logic,
                side,
                model,
                annualizer,
                fees,
                price_df,
                resolution,
                window,
                metric="SR"
            )
            if not fig_sweep_heatmaps:
                fig_sweep_heatmaps = [{
                    "label": f"parameter window ({window})",
                    "fig": build_parameter_sweep_heatmap(sweep_df, window, metric="SR"),
                }]
            fig_ic = build_multi_horizon_ic_plot(ic_horizon_df)
            html = build_html(alpha, metrics_bt, metrics_ft, metrics_vt, yearly_metrics_df, ic_summary, ic_horizon_df, yearly_ic_df, sweep_df, sweep_stats,
                              vt_stress_metrics_html, long_short_df, regime_df, fig_main, fig_position, fig_hist, fig_heatmap, fig_sweep_heatmaps, fig_ic,
                              fig_long_short, fig_regime_classification, fig_regime_strategy, fig_transformation_qq, fig_signal, window, t1, t2)

            out_path = os.path.join(OUTPUT_FOLDER, f"{alpha_id}.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)

            summary_rows.append(
                {
                    "rolling_180d_sr": df_vt["rolling_sharpe_180"].iloc[-1] if "rolling_sharpe_180" in df_vt.columns and not df_vt.empty else np.nan,
                    "current_drawdown": df_vt["dd"].iloc[-1] if "dd" in df_vt.columns and not df_vt.empty else np.nan,
                    "alpha_id": alpha_id,
                    "custom_id": alpha.get("custom_id", ""),
                    "model": model,
                    "timeframe": timeframe,
                    "window": window,
                    "threshold_1": t1,
                    "threshold_2": t2,
                    "BT_SR": metrics_bt.get("SR"),
                    "FT_SR": metrics_ft.get("SR"),
                    "VT_SR": metrics_vt.get("SR"),
                    "BT_MDD": metrics_bt.get("MDD"),
                    "FT_MDD": metrics_ft.get("MDD"),
                    "VT_MDD": metrics_vt.get("MDD"),
                    "IC": ic_summary.get("IC"),
                    "RankIC": ic_summary.get("RankIC"),
                    "sweep_status": get_sweep_status(sweep_stats),
                    "remark": "FAILED" if (
                        ("rolling_sharpe_180" in df_vt.columns and not df_vt.empty and pd.notna(df_vt["rolling_sharpe_180"].iloc[-1]) and df_vt["rolling_sharpe_180"].iloc[-1] < 0)
                        or ("dd" in df_vt.columns and not df_vt.empty and pd.notna(df_vt["dd"].iloc[-1]) and pd.notna(metrics_vt.get("MDD")) and df_vt["dd"].iloc[-1] < metrics_vt.get("MDD"))
                    ) else "PASSED",
                    "remark_reason": "; ".join(
                        reason for reason, condition in [
                            ("Rolling 180d SR < 0", "rolling_sharpe_180" in df_vt.columns and not df_vt.empty and pd.notna(df_vt["rolling_sharpe_180"].iloc[-1]) and df_vt["rolling_sharpe_180"].iloc[-1] < 0),
                            ("Current drawdown < historical MDD", "dd" in df_vt.columns and not df_vt.empty and pd.notna(df_vt["dd"].iloc[-1]) and pd.notna(metrics_vt.get("MDD")) and df_vt["dd"].iloc[-1] < metrics_vt.get("MDD")),
                        ] if condition
                    ) or "Within limits",
                    "report_file": os.path.basename(out_path),
                }
            )

            print(f"[OK] Report saved -> {out_path}")
            print(f"     BT SR={metrics_bt['SR']}  FT SR={metrics_ft['SR']}")

        except Exception as e:
            print(f"[ERROR] {alpha_id}: {e}")
            import traceback; traceback.print_exc()

    summary_html = build_summary_html(summary_rows)
    summary_path = os.path.join(OUTPUT_FOLDER, "alpha_summary.html")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_html)
    print(f"[OK] Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()

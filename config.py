from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(r"C:\Users\User\OneDrive\Documents\DA\Temp_Backtest\doidoi_backtest\backtest_CQ\monitor")
BACKTEST_DIR = BASE_DIR / "backtest_feb"


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# Shared paths
ALPHA_LIST_CSV = BASE_DIR / "csv_alpha_list.csv"
ALPHA_LIST_JSON = BASE_DIR / "alpha.json"
JSON_FILE = str(ALPHA_LIST_JSON)
PRICE_FOLDER = str(BASE_DIR / "price")
DATA_FOLDER = str(BASE_DIR / "data_bt")
OUTPUT_FOLDER_BACKTEST = str(BACKTEST_DIR)
OUTPUT_FOLDER_HTML = str(BACKTEST_DIR / "alpha_json" / "Check")
OUTPUT_CSV = "alpha_metrics_result_all_periods.csv"

for directory in (
    PRICE_FOLDER,
    DATA_FOLDER,
    OUTPUT_FOLDER_BACKTEST,
    OUTPUT_FOLDER_HTML,
):
    ensure_directory(directory)


# Data fetching
API_KEY = ""
FETCH_START = datetime(year=2021, month=1, day=1, tzinfo=timezone.utc)
FETCH_END = datetime(year=2026, month=8, day=1, tzinfo=timezone.utc)
MAX_CONCURRENT_REQUESTS = 5
PRICE_TOPICS = [
    "bybit-linear|candle?interval=1m&symbol=BTCUSDT",
    "binance-linear|candle?interval=1m&symbol=ETHUSDT",
]

ASSET_PRICE_SOURCE = {
    "BTC": "bybit",
    "ETH": "binance",
}

TARGET_ALPHA_ID = None
# ["TURTLE003_00098", "TURTLE003_00018", "TURTLE003_00008","TURTLE003_00062","TURTLE003_00142","TURTLE003_00080","TURTLE003_00085","TURTLE003_00009"]

# TARGET_ALPHA_ID = [
# "TURTLE003_00008", "TURTLE003_00009", "TURTLE003_00013", "TURTLE003_00015", "TURTLE003_00018", "TURTLE003_00039", "TURTLE003_00043", "TURTLE003_00054", "TURTLE003_00057", "TURTLE003_00061", "TURTLE003_00062", "TURTLE003_00069", "TURTLE003_00072", "TURTLE003_00078", "TURTLE003_00080", "TURTLE003_00085", "TURTLE003_00087", "TURTLE003_00090", "TURTLE003_00098", "TURTLE003_00100", "TURTLE003_00140", "TURTLE003_00141", "TURTLE003_00142", "TURTLE003_00143", "TURTLE003_00144", "TURTLE003_00145", "TURTLE003_00147"
# ]

# Shared date windows
BT_START_DATE = "2021-01-01"
BT_END_DATE = "2024-10-01"
FT_START_DATE = "2024-10-01"
FT_END_DATE = "2025-10-01"
VT_START_DATE = "2025-10-01"
VT_END_DATE = "2026-10-04" 

START_DATE = "2021-01-01"
END_DATE = "2026-10-04"


# Permutation/grid defaults
TOP_N_RESULTS = 20
SR_DRIFT_THRESHOLD = 0.10
MDD_FILTER_THRESHOLD = -0.50
PARAM_WINDOW = None #770
PARAM_THRESHOLD_1 = None #0.7
PARAM_THRESHOLD_2 = None #-0.25
WINDOW_RANGE = (100, 800, 20)
THRESHOLD_1_RANGE = (0.5, 4.1, 0.20)
THRESHOLD_2_RANGE = (-4.0, -0.5, 0.20)
WINDOW_STEP = 50
THRESHOLD_STEP = 0.2
WINDOW_SWEEP_STEPS = 3
THRESHOLD_SWEEP_STEPS = 3


# TPE search params
TPE_SEED = 42
TPE_N_TRIALS = 1000
TPE_WINDOW_MIN = 100
TPE_WINDOW_MAX = 800
TPE_WINDOW_STEP = 10
TPE_MIN_SR = 1.5
TPE_MIN_FT_SR = 0.0
TPE_FT_SR_TOLERANCE = 0.50
TPE_MIN_MDD = -0.5
TPE_MIN_SINGLE_SIDE_TPI = 1.5
TPE_MIN_BOTH_SIDE_TPI = 3.0
TPE_MAX_TPI = 100.0
TPE_PLATEAU_WINDOW_DELTA = 50
TPE_PLATEAU_THRESHOLD_DELTA = 0.25
TPE_PLATEAU_MIN_NEIGHBORS = 5
TPE_MODEL_SEARCH_MODE = "categorical"  # "categorical" or "per_model"
TPE_DEFAULT_MODEL_OPTIONS = None #[
    # 'zscore','kurtosis_zscore', 'quantilev1_zscore', 'percentilerank_zscore', 'L2_zscore'
    # 'skew_zscore', 'volatilityv0_zscore', 'variance_zscore', 
    # 'zlmadiff_zscore', 'cci_zscore', 'decay_zscore', 'pn_zscore', 
    # 'L2_zscore', 'rvi_zscore', 'percentilerank_zscore', 
    # 'smadiff_zscore', 'dema_zscore', 'swmadiff_zscore', 
    # 'cmo_zscore', 'coppock_zscore', 'er_zscore', 'pgo_zscore', 
    # 'kurtskew_zscore'
# ]
TPE_LOGIC_OPTIONS = None #[
#     "trend",
#     "trend_reverse",
#     "mr",
#     "mr_reverse",
#     "fast",
#     "fast_reverse",
# ]
# None = search only each alpha's default side parsed from entry_exit_logic.
TPE_SIDE_OPTIONS = None


# Walk-forward TPE params (36-6-3-3, 18-6-3-3, 9-3-3-3 )
WF_TRAIN_MONTHS = 6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
WF_TEST_MONTHS = 3
WF_TRADE_MONTHS = 3
WF_STEP_MONTHS = 3
WF_pass_rate_threshold = 0.35
WF_CORR_THRESHOLD = 0.65
WF_CORR_MIN_SR_GATE = 1.0
WF_ALLOW_PARTIAL_FINAL_TRADE = True
WF_PLATEAU_WINDOW_DELTA = TPE_PLATEAU_WINDOW_DELTA
WF_PLATEAU_THRESHOLD_DELTA = TPE_PLATEAU_THRESHOLD_DELTA
WF_PLATEAU_MIN_NEIGHBORS = TPE_PLATEAU_MIN_NEIGHBORS
WF_USE_BASE_MODEL = False
WF_USE_BASE_LOGIC = False
WF_ENQUEUE_BASE_TRIAL = False
WF_REQUIRE_TRAIN_TEST_PASS_FOR_SELECTION = True
WF_L1Y_MONTHS = 12
WF_L1Y_MIN_SR = 0.5
WF_L1Y_MIN_MDD = -0.50
WF_L1Y_MIN_TPI = 1.0
WF_L1Y_MIN_TRADES = 10


# PBO / CSCV audit params
PBO_N_TRIALS = 200
PBO_N_BLOCKS = 10
PBO_MAX_SPLITS = None
PBO_SEED = TPE_SEED
PBO_START_DATE = START_DATE
PBO_END_DATE = END_DATE


# HTML heatmap neighborhood
WINDOW_HEATMAP_DELTA = 20
WINDOW_HEATMAP_STEPS = 3
THRESHOLD_HEATMAP_STEPS = 6


# Correlation portfolio report
CORRELATION_ALPHA_IDS = None #[
#     "TURTLE003_00008",
# ]
CORRELATION_THRESHOLD = 0.65
CORRELATION_USE_ABSOLUTE = False
CORRELATION_MIN_OBSERVATIONS = 100
CORRELATION_RESAMPLE = None
CORRELATION_PARAM_SOURCE = "json"
CORRELATION_START_DATE = START_DATE
CORRELATION_END_DATE = END_DATE
CORRELATION_OUTPUT_FOLDER = str(BACKTEST_DIR / "correlation_portfolio")
CORRELATION_OUTPUT_HTML = "correlation_portfolio_report.html"
ensure_directory(CORRELATION_OUTPUT_FOLDER)

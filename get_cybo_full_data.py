import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import colorlog
import cybotrade_datasource
import pandas as pd
from dateutil.relativedelta import relativedelta

from config import DATA_FOLDER, JSON_FILE


API_KEY = ""

# Cybotrade factor-data configuration.
OUTPUT_FOLDER = Path(DATA_FOLDER)
NAN_FOLDER = OUTPUT_FOLDER / "nan"
NAN_ROW_THRESHOLD = 0.03
FREQUENCY = "h"
CYBO_START_TIME = datetime(2021, 1, 1, tzinfo=timezone.utc)
CYBO_END_TIME = datetime(2026, 10, 4, tzinfo=timezone.utc)
PRICE_START_TIME = datetime(year=2021, month=1, day=1, tzinfo=timezone.utc)
PRICE_END_TIME = datetime(year=2026, month=10, day=4, tzinfo=timezone.utc)

# Optional endpoints that should always be fetched in addition to those found in
# JSON_FILE. You can paste topics here, or provide them at run time with
# --endpoint.
MANUAL_ENDPOINTS = [
    # "glassnode|derivatives/futures_funding_rate_perpetual_all_v2?a=BTC&i=1h",
]

# Price-data configuration.
PRICE_FOLDER_PATH = (
    r"C:\Users\User\OneDrive\Documents\DA\Temp_Backtest\doidoi_backtest"
    r"\backtest_CQ\monitor\price"
)

PRICE_TOPICS = [
    "bybit-linear|candle?interval=1m&symbol=BTCUSDT",
    # "binance-linear|candle?interval=1m&symbol=ETHUSDT"
]

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
NAN_FOLDER.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch Cybotrade datasource endpoints and price data into CSV files."
    )
    parser.add_argument(
        "--endpoint",
        "--topic",
        dest="endpoints",
        action="append",
        default=[],
        metavar="TOPIC",
        help=(
            "Endpoint/topic to fetch in addition to config endpoints. "
            "Repeat this option to fetch multiple endpoints."
        ),
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Do not read endpoints from JSON_FILE; fetch only manual/CLI endpoints.",
    )
    return parser.parse_args()


def topic_to_filename(topic: str) -> str:
    topic = topic.replace("|", "_")
    topic = topic.replace("/", "_").replace("-", "_")
    topic = topic.replace("?", "_").replace("&", "_").replace("=", "_")
    topic = "".join(c for c in topic if c.isalnum() or c == "_")
    return topic + ".csv"


def parse_timestamp_series(df: pd.DataFrame) -> pd.Series:
    if "start_time" in df.columns:
        numeric_start_time = pd.to_numeric(df["start_time"], errors="coerce")
        if numeric_start_time.notna().any():
            sample = numeric_start_time.dropna().median()
            if sample >= 1e17:
                unit = "ns"
            elif sample >= 1e14:
                unit = "us"
            elif sample >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            return pd.to_datetime(
                numeric_start_time, unit=unit, utc=True, errors="coerce"
            )

    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    raise ValueError("CSV has no start_time or datetime column")


def normalize_hourly_grid(df: pd.DataFrame):
    if df.empty:
        return df, 0, 0, 0.0

    df = df.copy()
    df["_time"] = parse_timestamp_series(df)
    df = df.dropna(subset=["_time"]).sort_values("_time")
    df = df.drop_duplicates(subset=["_time"], keep="last")

    original_index = pd.DatetimeIndex(df["_time"])
    full_index = pd.date_range(
        original_index.min(), original_index.max(), freq=FREQUENCY
    )
    missing_count = len(full_index.difference(original_index))

    df = df.set_index("_time").reindex(full_index)
    df.index.name = "_time"

    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    df["start_time"] = (
        (df.index - epoch) // pd.Timedelta(milliseconds=1)
    ).astype("int64")
    df["datetime"] = df.index.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S")

    ordered_cols = ["start_time"]
    data_cols = [c for c in df.columns if c not in {"start_time", "datetime"}]
    ordered_cols.extend(data_cols)
    ordered_cols.append("datetime")
    df = df[ordered_cols].reset_index(drop=True)

    nan_rows = int(df[data_cols].isna().any(axis=1).sum()) if data_cols else 0
    nan_ratio = nan_rows / len(df) if len(df) else 0.0
    return df, missing_count, nan_rows, nan_ratio


def checked_save_csv(df: pd.DataFrame, file_name: Path):
    df, missing_count, nan_rows, nan_ratio = normalize_hourly_grid(df)
    target = NAN_FOLDER / file_name.name if nan_ratio > NAN_ROW_THRESHOLD else file_name

    if target != file_name and file_name.exists():
        file_name.unlink()

    df.to_csv(target, index=False)

    status = "quarantined" if target != file_name else "saved"
    print(
        f"{status.upper()} {target} | "
        f"missing_timestamps={missing_count} nan_rows={nan_rows} "
        f"nan_ratio={nan_ratio:.2%}"
    )
    return {
        "status": status,
        "file_name": str(target),
        "missing_timestamps": missing_count,
        "nan_rows": nan_rows,
        "nan_ratio": nan_ratio,
    }


def latest_timestamp(df: pd.DataFrame):
    parsed = parse_timestamp_series(df)
    return parsed.max()


async def fetch_and_save(topic: str, start_time: datetime, end_time: datetime):
    file_name = OUTPUT_FOLDER / topic_to_filename(topic)

    if file_name.exists():
        try:
            existing_df = pd.read_csv(file_name)
            if not existing_df.empty and "start_time" in existing_df.columns:
                last_ts = latest_timestamp(existing_df)
                if last_ts >= end_time - pd.Timedelta(days=1):
                    print(f"Skipping {topic} (already up to date)")
                    result = checked_save_csv(existing_df, file_name)
                    result["topic"] = topic
                    result["status"] = (
                        "skipped" if result["status"] == "saved" else result["status"]
                    )
                    return result
                print(
                    f"Re-fetching {topic} "
                    f"(data ends at {last_ts}, need up to {end_time})"
                )
            else:
                print(f"Re-fetching {topic} (empty or missing timestamp column)")
        except Exception as error:
            print(f"Re-fetching {topic} (failed to read existing file: {error})")

    data = await cybotrade_datasource.query_paginated(
        api_key=API_KEY,
        topic=topic,
        start_time=start_time,
        end_time=end_time,
    )

    df = pd.DataFrame(data)
    result = checked_save_csv(df, file_name)
    result["topic"] = topic
    return result


def load_config_topics(json_file):
    with open(json_file, "r") as file:
        alphas = json.load(file)

    topics = set()
    for alpha in alphas:
        ds_raw = alpha.get("datasource_structure", {})
        try:
            if isinstance(ds_raw, str):
                ds = json.loads(ds_raw.replace("'", '"'))
            elif isinstance(ds_raw, dict):
                ds = ds_raw
            else:
                raise TypeError(
                    f"unsupported datasource_structure type: {type(ds_raw).__name__}"
                )
        except (json.JSONDecodeError, TypeError) as error:
            print(
                "Failed to parse datasource_structure for "
                f"{alpha.get('custom_id')}: {error}"
            )
            continue

        for val in ds.values():
            if not isinstance(val, dict):
                continue
            topic = val.get("topic")
            if topic:
                topics.add(topic.strip())

    return topics


async def fetch_cybo_data(args):
    unique_topics = {
        topic.strip()
        for topic in [*MANUAL_ENDPOINTS, *args.endpoints]
        if topic and topic.strip()
    }

    if not args.skip_config:
        unique_topics.update(load_config_topics(JSON_FILE))

    if not unique_topics:
        raise SystemExit(
            "No endpoints found. Add MANUAL_ENDPOINTS, use --endpoint, "
            "or run without --skip-config."
        )

    sorted_topics = sorted(unique_topics)
    print(f"Found {len(sorted_topics)} unique topics to fetch.")

    tasks = [
        fetch_and_save(topic, CYBO_START_TIME, CYBO_END_TIME)
        for topic in sorted_topics
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    saved_count = 0
    skipped_count = 0
    quarantined_count = 0
    failed_topics = []

    for topic, result in zip(sorted_topics, results):
        if isinstance(result, Exception):
            failed_topics.append((topic, str(result)))
            continue

        status = result.get("status")
        if status == "saved":
            saved_count += 1
        elif status == "skipped":
            skipped_count += 1
        elif status == "quarantined":
            quarantined_count += 1

    print("\nFetch summary:")
    print(f"Saved: {saved_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Quarantined: {quarantined_count}")
    print(f"Failed: {len(failed_topics)}")

    if failed_topics:
        print("\nFailed topics:")
        for topic, error in failed_topics:
            print(f"- {topic}: {error}")


def setup_logger():
    log_format = "%(log_color)s%(asctime)s - %(levelname)s - %(message)s%(reset)s"
    color_formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    console_handler.stream.reconfigure(encoding="utf-8")
    file_handler = logging.FileHandler("../cybotrade_query.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def month_chunks(start, end):
    cur = start
    while cur < end:
        nxt = min(cur + relativedelta(years=1), end)
        yield cur, nxt
        cur = nxt


async def fetch_price_data(logger):
    price_query_start = datetime.now(timezone.utc)
    logger.info("Starting data query process...")
    logger.info(f"Fetching data from {PRICE_START_TIME} to {PRICE_END_TIME}")
    os.makedirs(PRICE_FOLDER_PATH, exist_ok=True)

    for topic in PRICE_TOPICS:
        exchange = topic.split("|")[0].split("-")[0]
        symbol = topic.split("&")[-1].split("=")[-1]
        interval = topic.split("?")[1].split("&")[0].split("=")[-1]
        filename = f"{exchange}_{interval}_{symbol}.csv"
        file_path = os.path.join(PRICE_FOLDER_PATH, filename)

        first_write = True
        total_records = 0

        for chunk_start, chunk_end in month_chunks(
            PRICE_START_TIME, PRICE_END_TIME
        ):
            try:
                logger.info(f"Querying {topic} | {chunk_start} -> {chunk_end}")
                data = await cybotrade_datasource.query_paginated(
                    api_key=API_KEY,
                    topic=topic,
                    start_time=chunk_start,
                    end_time=chunk_end,
                )
                if data:
                    df = pd.DataFrame(data)
                    df.to_csv(
                        file_path,
                        mode="w" if first_write else "a",
                        header=first_write,
                        index=False,
                    )
                    first_write = False
                    total_records += len(data)
                    logger.info(
                        f"  +{len(data)} records (running total: {total_records})"
                    )
                else:
                    logger.warning(f"  No data for {chunk_start} -> {chunk_end}")
                del data
            except Exception as error:
                logger.error(
                    f"Error for {topic} chunk {chunk_start}->{chunk_end}: {error}",
                    exc_info=True,
                )

        logger.info(
            f"Done {topic}: {total_records} total records saved to {file_path}"
        )

    time_taken = datetime.now(timezone.utc) - price_query_start
    logger.info(f"Total time taken: {time_taken}")


async def main(args):
    await fetch_cybo_data(args)
    logger = setup_logger()
    await fetch_price_data(logger)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))

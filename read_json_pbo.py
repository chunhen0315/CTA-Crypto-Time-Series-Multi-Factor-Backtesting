import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import read_json_tpe_permutation as tpe

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


OUTPUT_DIR = Path(config.OUTPUT_FOLDER_BACKTEST) / "pbo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PBO_N_TRIALS = int(getattr(config, "PBO_N_TRIALS", min(getattr(config, "TPE_N_TRIALS", 200), 200)))
PBO_N_BLOCKS = int(getattr(config, "PBO_N_BLOCKS", 10))
PBO_MAX_SPLITS = getattr(config, "PBO_MAX_SPLITS", None)
PBO_START_DATE = getattr(config, "PBO_START_DATE", config.START_DATE)
PBO_END_DATE = getattr(config, "PBO_END_DATE", config.END_DATE)
PBO_SEED = int(getattr(config, "PBO_SEED", getattr(config, "TPE_SEED", 42)))


def date_str(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def selected_alphas(alphas):
    target = config.TARGET_ALPHA_ID
    if target is None:
        return alphas

    if isinstance(target, str):
        value = target.strip()
        if not value or value.lower() in {"none", "all"}:
            return alphas
        target_ids = [value]
    else:
        target_ids = [str(item).strip() for item in target if item is not None and str(item).strip()]

    target_set = set(target_ids)
    selected = [alpha for alpha in alphas if alpha.get("alpha_id") in target_set]
    if not selected:
        raise ValueError(f"Alpha ID not found: {config.TARGET_ALPHA_ID}")
    return selected


def value_grid(start, stop, step):
    count = int(round((stop - start) / step)) + 1
    return [round(start + idx * step, 4) for idx in range(max(count, 0))]


def sample_candidates(strategy, rng):
    model_options = tpe.build_model_options(strategy["model"])
    logic_options = tpe.build_logic_options(strategy["logic"])
    side_options = tpe.build_side_options(strategy["side"])
    window_options = list(range(tpe.WINDOW_MIN, tpe.WINDOW_MAX + 1, tpe.WINDOW_STEP))

    candidates = [
        {
            "candidate_id": 0,
            "model": strategy["model"],
            "logic": strategy["logic"],
            "side": strategy["side"],
            "window": int(strategy["window"]),
            "threshold_1": float(strategy["t1"]),
            "threshold_2": float(strategy["t2"]),
            "candidate_source": "json_base",
        }
    ]
    seen = {
        (
            candidates[0]["model"],
            candidates[0]["logic"],
            candidates[0]["side"],
            candidates[0]["window"],
            candidates[0]["threshold_1"],
            candidates[0]["threshold_2"],
        )
    }

    attempts = 0
    max_attempts = max(PBO_N_TRIALS * 20, 1000)
    while len(candidates) < PBO_N_TRIALS and attempts < max_attempts:
        attempts += 1
        model = rng.choice(model_options)
        logic = rng.choice(logic_options)
        side = rng.choice(side_options)
        window = int(rng.choice(window_options))
        t1_min, t1_max, t2_min, t2_max, step = tpe.threshold_bounds_for_model(model)
        t1 = float(rng.choice(value_grid(t1_min, t1_max, step)))
        t2 = float(rng.choice(value_grid(t2_min, t2_max, step)))
        if t1 <= t2:
            continue

        key = (model, logic, side, window, t1, t2)
        if key in seen:
            continue

        seen.add(key)
        candidates.append(
            {
                "candidate_id": len(candidates),
                "model": model,
                "logic": logic,
                "side": side,
                "window": window,
                "threshold_1": t1,
                "threshold_2": t2,
                "candidate_source": "random_search_space",
            }
        )

    return pd.DataFrame(candidates)


def build_time_blocks(price_df):
    if PBO_N_BLOCKS < 4 or PBO_N_BLOCKS % 2 != 0:
        raise ValueError("PBO_N_BLOCKS must be an even integer >= 4")

    start_ts = pd.Timestamp(PBO_START_DATE)
    end_ts = pd.Timestamp(PBO_END_DATE)
    index = price_df.loc[start_ts:end_ts].dropna(subset=["close"]).index
    if len(index) < PBO_N_BLOCKS:
        raise ValueError(f"Not enough price rows for {PBO_N_BLOCKS} PBO blocks")

    blocks = []
    for block_id, block_index in enumerate(np.array_split(index, PBO_N_BLOCKS), start=1):
        if len(block_index) == 0:
            continue
        blocks.append(
            {
                "block_id": block_id,
                "block_start": block_index[0],
                "block_end": block_index[-1],
                "row_count": len(block_index),
            }
        )

    if len(blocks) != PBO_N_BLOCKS:
        raise ValueError(f"Expected {PBO_N_BLOCKS} blocks, got {len(blocks)}")
    return blocks


def metrics_to_prefixed_row(prefix, metrics):
    return {
        f"{prefix}_SR": metrics["SR"],
        f"{prefix}_CR": metrics["CR"],
        f"{prefix}_MDD": metrics["MDD"],
        f"{prefix}_AR": metrics["AR"],
        f"{prefix}_TR": metrics["TR"],
        f"{prefix}_TPI": metrics["TPI"],
        f"{prefix}_num_trades": metrics["num_trades"],
    }


def evaluate_candidates_by_block(alpha_id, candidates_df, blocks, factor_df, price_df, strategy):
    rows = []
    metric_cols = [
        "BLOCK_SR",
        "BLOCK_CR",
        "BLOCK_MDD",
        "BLOCK_AR",
        "BLOCK_TR",
        "BLOCK_TPI",
        "BLOCK_num_trades",
    ]
    total = len(candidates_df) * len(blocks)
    with tqdm(total=total, desc=f"{alpha_id} PBO blocks", unit="eval") as pbar:
        for _, candidate in candidates_df.iterrows():
            for block in blocks:
                row = {
                    "alpha_id": alpha_id,
                    "candidate_id": int(candidate["candidate_id"]),
                    "block_id": int(block["block_id"]),
                    "block_start": date_str(block["block_start"]),
                    "block_end": date_str(block["block_end"]),
                    "block_rows": int(block["row_count"]),
                    "model": candidate["model"],
                    "logic": candidate["logic"],
                    "side": candidate["side"],
                    "window": int(candidate["window"]),
                    "threshold_1": float(candidate["threshold_1"]),
                    "threshold_2": float(candidate["threshold_2"]),
                    "candidate_source": candidate["candidate_source"],
                    "error": "",
                }
                try:
                    metrics, _ = tpe.evaluate_period(
                        factor_df,
                        price_df,
                        block["block_start"],
                        block["block_end"],
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
                    row["error"] = str(exc)
                else:
                    row.update(metrics_to_prefixed_row("BLOCK", metrics))
                rows.append(row)
                pbar.update(1)
    block_metrics_df = pd.DataFrame(rows)
    for col in metric_cols:
        if col not in block_metrics_df.columns:
            block_metrics_df[col] = np.nan
    return block_metrics_df


def limit_splits(split_list, rng):
    if PBO_MAX_SPLITS is None:
        return split_list

    max_splits = int(PBO_MAX_SPLITS)
    if max_splits <= 0 or len(split_list) <= max_splits:
        return split_list

    selected = rng.choice(len(split_list), size=max_splits, replace=False)
    return [split_list[int(idx)] for idx in sorted(selected)]


def pbo_from_block_metrics(block_metrics_df, rng):
    sr_matrix = block_metrics_df.pivot(index="candidate_id", columns="block_id", values="BLOCK_SR")
    sr_matrix = sr_matrix.apply(pd.to_numeric, errors="coerce")
    block_ids = list(sr_matrix.columns)
    half = len(block_ids) // 2
    split_sets = list(combinations(block_ids, half))
    split_sets = limit_splits(split_sets, rng)

    split_rows = []
    for split_id, train_blocks in enumerate(split_sets, start=1):
        train_blocks = list(train_blocks)
        test_blocks = [block_id for block_id in block_ids if block_id not in train_blocks]
        train_scores = sr_matrix[train_blocks].mean(axis=1, skipna=True)
        test_scores = sr_matrix[test_blocks].mean(axis=1, skipna=True)
        train_scores[sr_matrix[train_blocks].notna().sum(axis=1) < len(train_blocks)] = np.nan
        test_scores[sr_matrix[test_blocks].notna().sum(axis=1) < len(test_blocks)] = np.nan
        valid_train = train_scores.dropna()
        if valid_train.empty:
            continue

        selected_candidate = int(valid_train.idxmax())
        selected_train_sr = train_scores.loc[selected_candidate]
        selected_oos_sr = test_scores.loc[selected_candidate]
        valid_oos = test_scores.dropna()
        if pd.isna(selected_oos_sr) or len(valid_oos) < 2:
            continue

        rank_desc = float(valid_oos.rank(ascending=False, method="average").loc[selected_candidate])
        oos_percentile = (len(valid_oos) - rank_desc + 1.0) / (len(valid_oos) + 1.0)
        oos_percentile = min(max(oos_percentile, 1e-6), 1.0 - 1e-6)
        logit_lambda = float(np.log(oos_percentile / (1.0 - oos_percentile)))

        split_rows.append(
            {
                "split_id": split_id,
                "train_blocks": ",".join(str(item) for item in train_blocks),
                "test_blocks": ",".join(str(item) for item in test_blocks),
                "selected_candidate_id": selected_candidate,
                "selected_train_sr": round(float(selected_train_sr), 4),
                "selected_oos_sr": round(float(selected_oos_sr), 4),
                "oos_rank_desc": round(rank_desc, 4),
                "oos_candidate_count": int(len(valid_oos)),
                "oos_percentile": round(float(oos_percentile), 6),
                "logit_lambda": round(logit_lambda, 6),
                "is_oos_below_median": bool(logit_lambda < 0),
            }
        )

    return pd.DataFrame(split_rows)


def summarize_pbo(alpha, candidates_df, block_metrics_df, splits_df, blocks):
    alpha_id = alpha.get("alpha_id", "")
    if splits_df.empty:
        pbo = np.nan
    else:
        pbo = round(float(splits_df["is_oos_below_median"].mean()), 4)

    sr_matrix = block_metrics_df.pivot(index="candidate_id", columns="block_id", values="BLOCK_SR")
    candidate_mean_sr = sr_matrix.mean(axis=1, skipna=True)
    best_full_candidate_id = int(candidate_mean_sr.idxmax()) if candidate_mean_sr.notna().any() else None
    best_full_mean_sr = candidate_mean_sr.loc[best_full_candidate_id] if best_full_candidate_id is not None else np.nan

    selected_counts = (
        splits_df["selected_candidate_id"].value_counts().rename_axis("candidate_id").reset_index(name="pbo_selected_count")
        if not splits_df.empty
        else pd.DataFrame(columns=["candidate_id", "pbo_selected_count"])
    )
    candidates_with_counts = candidates_df.merge(selected_counts, on="candidate_id", how="left")
    candidates_with_counts["pbo_selected_count"] = candidates_with_counts["pbo_selected_count"].fillna(0).astype(int)
    if best_full_candidate_id is not None:
        best_candidate = candidates_with_counts[candidates_with_counts["candidate_id"] == best_full_candidate_id].iloc[0].to_dict()
    else:
        best_candidate = {}

    return {
        "alpha_id": alpha_id,
        "pbo": pbo,
        "pbo_pct": round(pbo * 100, 2) if not pd.isna(pbo) else np.nan,
        "split_count": int(len(splits_df)),
        "candidate_count": int(len(candidates_df)),
        "block_count": int(len(blocks)),
        "block_start": date_str(blocks[0]["block_start"]) if blocks else "",
        "block_end": date_str(blocks[-1]["block_end"]) if blocks else "",
        "median_oos_percentile": round(float(splits_df["oos_percentile"].median()), 6) if not splits_df.empty else np.nan,
        "median_logit_lambda": round(float(splits_df["logit_lambda"].median()), 6) if not splits_df.empty else np.nan,
        "mean_selected_train_sr": round(float(splits_df["selected_train_sr"].mean()), 4) if not splits_df.empty else np.nan,
        "mean_selected_oos_sr": round(float(splits_df["selected_oos_sr"].mean()), 4) if not splits_df.empty else np.nan,
        "q25_selected_oos_sr": round(float(splits_df["selected_oos_sr"].quantile(0.25)), 4) if not splits_df.empty else np.nan,
        "best_full_candidate_id": best_full_candidate_id,
        "best_full_mean_block_sr": round(float(best_full_mean_sr), 4) if not pd.isna(best_full_mean_sr) else np.nan,
        "best_model": best_candidate.get("model", ""),
        "best_logic": best_candidate.get("logic", ""),
        "best_side": best_candidate.get("side", ""),
        "best_window": best_candidate.get("window", ""),
        "best_threshold_1": best_candidate.get("threshold_1", ""),
        "best_threshold_2": best_candidate.get("threshold_2", ""),
        "pbo_n_trials_config": PBO_N_TRIALS,
        "pbo_n_blocks_config": PBO_N_BLOCKS,
        "pbo_max_splits_config": "" if PBO_MAX_SPLITS is None else PBO_MAX_SPLITS,
    }, candidates_with_counts


def run_pbo(alpha):
    alpha_id = alpha.get("alpha_id")
    print(f"\nPBO CSCV audit for {alpha_id}")

    factor_df, alpha_formula = tpe.build_factor_dataframe(alpha)
    strategy = tpe.get_strategy_config(alpha)
    if strategy["resolution"] is None:
        raise ValueError(f"Unsupported timeframe: {strategy['timeframe']}")

    price_df = tpe.load_price_df(
        trade_asset=alpha.get("trade_asset", "BTC"),
        timeframe=strategy["timeframe"],
        price_delay=int(alpha.get("shift_backtest_candle_minute", 0)),
    )

    rng = np.random.default_rng(PBO_SEED)
    blocks = build_time_blocks(price_df)
    candidates_df = sample_candidates(strategy, rng)
    if len(candidates_df) < 2:
        raise ValueError("PBO needs at least 2 candidate parameter sets")

    block_metrics_df = evaluate_candidates_by_block(alpha_id, candidates_df, blocks, factor_df, price_df, strategy)
    splits_df = pbo_from_block_metrics(block_metrics_df, rng)
    summary_row, candidates_with_counts = summarize_pbo(alpha, candidates_df, block_metrics_df, splits_df, blocks)

    alpha_dir = OUTPUT_DIR / alpha_id
    alpha_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = alpha_dir / f"{alpha_id}_pbo_candidates.csv"
    block_metrics_path = alpha_dir / f"{alpha_id}_pbo_block_metrics.csv"
    splits_path = alpha_dir / f"{alpha_id}_pbo_splits.csv"
    summary_path = alpha_dir / f"{alpha_id}_pbo_summary.csv"

    candidates_with_counts.to_csv(candidates_path, index=False)
    block_metrics_df.to_csv(block_metrics_path, index=False)
    splits_df.to_csv(splits_path, index=False)
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    print(f"formula: {alpha_formula}")
    print(
        f"base: model={strategy['model']} logic={strategy['logic']} side={strategy['side']} "
        f"window={strategy['window']} t1={strategy['t1']} t2={strategy['t2']}"
    )
    print(
        f"PBO={summary_row['pbo']} ({summary_row['pbo_pct']}%) "
        f"splits={summary_row['split_count']} candidates={summary_row['candidate_count']} "
        f"mean selected OOS SR={summary_row['mean_selected_oos_sr']} "
        f"q25 selected OOS SR={summary_row['q25_selected_oos_sr']}"
    )
    print(f"saved summary: {summary_path}")
    return summary_row


def main():
    with open(config.JSON_FILE, "r", encoding="utf-8") as file:
        alphas = json.load(file)

    summary_rows = []
    errors = []
    for alpha in selected_alphas(alphas):
        alpha_id = alpha.get("alpha_id", "")
        try:
            summary_rows.append(run_pbo(alpha))
        except Exception as exc:
            print(f"\nERROR {alpha_id}: {exc}")
            errors.append({"alpha_id": alpha_id, "error": str(exc)})

    if summary_rows:
        summary_path = OUTPUT_DIR / "pbo_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nPBO summary saved: {summary_path}")

    if errors:
        error_path = OUTPUT_DIR / "pbo_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"PBO errors saved: {error_path}")


if __name__ == "__main__":
    main()

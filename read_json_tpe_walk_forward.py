import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

import read_json_tpe_permutation as tpe
from config import (
    END_DATE,
    JSON_FILE,
    OUTPUT_FOLDER_BACKTEST,
    START_DATE,
    TARGET_ALPHA_ID,
    TOP_N_RESULTS,
    WF_ALLOW_PARTIAL_FINAL_TRADE as ALLOW_PARTIAL_FINAL_TRADE,
    WF_ENQUEUE_BASE_TRIAL as ENQUEUE_BASE_TRIAL,
    WF_L1Y_MIN_MDD as L1Y_MIN_MDD,
    WF_L1Y_MIN_SR as L1Y_MIN_SR,
    WF_L1Y_MIN_TPI as L1Y_MIN_TPI,
    WF_L1Y_MIN_TRADES as L1Y_MIN_TRADES,
    WF_L1Y_MONTHS as L1Y_MONTHS,
    WF_PLATEAU_MIN_NEIGHBORS as PLATEAU_MIN_NEIGHBORS,
    WF_PLATEAU_THRESHOLD_DELTA as PLATEAU_THRESHOLD_DELTA,
    WF_PLATEAU_WINDOW_DELTA as PLATEAU_WINDOW_DELTA,
    WF_STEP_MONTHS as WF_STEP_MONTHS,
    WF_TEST_MONTHS as TEST_MONTHS,
    WF_TRADE_MONTHS as TRADE_MONTHS,
    WF_TRAIN_MONTHS as TRAIN_MONTHS,
    WF_USE_BASE_LOGIC as USE_BASE_LOGIC,
    WF_USE_BASE_MODEL as USE_BASE_MODEL,
    WF_pass_rate_threshold as WFA_pass_rate_threshold
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"Fixed parameter .* is out of range for distribution", category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


OUTPUT_DIR = Path(OUTPUT_FOLDER_BACKTEST) / "tpe_walk_forward"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def metrics_to_row(prefix, metrics):
    return {
        f"{prefix}_SR": metrics["SR"],
        f"{prefix}_CR": metrics["CR"],
        f"{prefix}_MDD": metrics["MDD"],
        f"{prefix}_AR": metrics["AR"],
        f"{prefix}_TR": metrics["TR"],
        f"{prefix}_TPI": metrics["TPI"],
        f"{prefix}_num_trades": metrics["num_trades"],
    }


def unique_options(options):
    return [item for idx, item in enumerate(options) if item and item not in options[:idx]]


def target_alpha_ids():
    if TARGET_ALPHA_ID is None:
        return None

    if isinstance(TARGET_ALPHA_ID, str):
        value = TARGET_ALPHA_ID.strip()
        if not value or value.lower() in {"none", "all"}:
            return None
        return [value]

    values = [str(item).strip() for item in TARGET_ALPHA_ID if item is not None and str(item).strip()]
    return values or None


def selected_wf_alphas(alphas):
    target_ids = target_alpha_ids()
    if target_ids is None:
        return alphas

    target_set = set(target_ids)
    selected = [alpha for alpha in alphas if alpha.get("alpha_id") in target_set]
    found_set = {alpha.get("alpha_id") for alpha in selected}
    missing = [alpha_id for alpha_id in target_ids if alpha_id not in found_set]

    if missing:
        print(f"warning: alpha id not found and skipped: {missing}")
    if not selected:
        raise ValueError(f"No TARGET_ALPHA_ID found: {TARGET_ALPHA_ID}")

    return selected


def date_str(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def period_end(start_date, months):
    return pd.Timestamp(start_date) + pd.DateOffset(months=months) - pd.Timedelta(days=1)


def infer_data_end(factor_df, price_df):
    configured_end = pd.Timestamp(END_DATE).normalize()
    factor_end = pd.Timestamp(factor_df.index.max()).normalize()
    price_end = pd.Timestamp(price_df.index.max()).normalize()
    return min(configured_end, factor_end, price_end)


def build_wf_rounds(data_end):
    data_end = pd.Timestamp(data_end).normalize()
    base_train_start = pd.Timestamp(START_DATE).normalize()
    rounds = []
    offset_months = 0

    while True:
        train_start = base_train_start + pd.DateOffset(months=offset_months)
        train_end = period_end(train_start, TRAIN_MONTHS)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = period_end(test_start, TEST_MONTHS)
        trade_start = test_end + pd.Timedelta(days=1)
        planned_trade_end = period_end(trade_start, TRADE_MONTHS)

        if trade_start > data_end:
            break
        if not ALLOW_PARTIAL_FINAL_TRADE and planned_trade_end > data_end:
            break

        trade_end = min(planned_trade_end, data_end)
        rounds.append(
            {
                "round": len(rounds) + 1,
                "train_start": date_str(train_start),
                "train_end": date_str(train_end),
                "test_start": date_str(test_start),
                "test_end": date_str(test_end),
                "trade_start": date_str(trade_start),
                "trade_end": date_str(trade_end),
            }
        )
        offset_months += WF_STEP_MONTHS

    return rounds


def build_wf_model_options(base_model):
    if USE_BASE_MODEL and tpe.DEFAULT_MODEL_OPTIONS is None:
        return unique_options([base_model])
    options = []
    if USE_BASE_MODEL:
        options.append(base_model)
    if tpe.DEFAULT_MODEL_OPTIONS is not None:
        options.extend(tpe.DEFAULT_MODEL_OPTIONS)
    options = unique_options(options)
    return options or unique_options([base_model])


def build_wf_logic_options(base_logic):
    if USE_BASE_LOGIC and tpe.LOGIC_OPTIONS is None:
        return unique_options([base_logic])
    options = []
    if USE_BASE_LOGIC:
        options.append(base_logic)
    if tpe.LOGIC_OPTIONS is not None:
        options.extend(tpe.LOGIC_OPTIONS)
    options = unique_options(options)
    return options or unique_options([base_logic])


def passes_period_filters(row, prefix):
    tpi_threshold = tpe.tpi_threshold_for_side(row.get("side"))
    return passes_period_filters_with_tpi(row, prefix, tpi_threshold)


def passes_period_filters_with_tpi(row, prefix, tpi_threshold):
    checks = {
        f"{prefix}_SR": tpe.MIN_SR,
        f"{prefix}_MDD": tpe.MIN_MDD,
        f"{prefix}_TPI": tpi_threshold,
    }

    for metric, threshold in checks.items():
        value = row.get(metric)
        if pd.isna(value):
            return False
        if metric.endswith("_MDD"):
            if value <= threshold:
                return False
        elif value <= threshold:
            return False

    return True


def passes_train_test_filters(row):
    return passes_period_filters(row, "TRAIN") and passes_period_filters(row, "TEST")


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def capped_ratio(value, target):
    value = safe_float(value)
    if pd.isna(value) or target == 0:
        return -1.0
    return min(value / target, 1.5)


def soft_fallback_score(row):
    return tpe.selection_scoring_lib.selection_score(
        row,
        "TRAIN",
        "TEST",
        side=row.get("side"),
        min_primary_sr=tpe.MIN_SR,
        min_validation_sr=row.get("TEST_MIN_SR", tpe.MIN_FT_SR),
        max_mdd=tpe.MIN_MDD,
    )


def selected_params_df(candidates_df):
    if candidates_df.empty:
        return candidates_df

    candidates_df = candidates_df.copy()
    filter_col = "FILTER_PASS" if "FILTER_PASS" in candidates_df.columns else "TRAIN_TEST_PASS"
    strict_mask = candidates_df[filter_col].fillna(False)
    pass_sort_cols = [col for col in ["SELECTION_SCORE", "plateau_score", "TEST_SR", "TRAIN_SR"] if col in candidates_df.columns]
    fallback_sort_cols = [filter_col] + pass_sort_cols

    if strict_mask.any():
        selected_df = candidates_df[strict_mask].sort_values(
            pass_sort_cols,
            ascending=[False] * len(pass_sort_cols),
            na_position="last",
        ).head(1).reset_index(drop=True)
        selected_df["SELECTION_MODE"] = "pass_train_test_best_selection_score"
        selected_df["WFA_SELECTED_PASS"] = True
    else:
        selected_df = candidates_df.sort_values(
            fallback_sort_cols,
            ascending=[False] * len(fallback_sort_cols),
            na_position="last",
        ).head(1).reset_index(drop=True)
        if bool(selected_df.iloc[0].get(filter_col, False)):
            mode = "pass_train_test_best_available"
        else:
            mode = "failed_train_test_filter_best_available"
        selected_df["SELECTION_MODE"] = mode
        selected_df["WFA_SELECTED_PASS"] = False

    selected_df["PARAM_SELECTED"] = True
    return selected_df


def round_error_df(alpha, wf_round, exc):
    return pd.DataFrame(
        [
            {
                "alpha_id": alpha.get("alpha_id"),
                "round": wf_round["round"],
                "train_start": wf_round["train_start"],
                "train_end": wf_round["train_end"],
                "test_start": wf_round["test_start"],
                "test_end": wf_round["test_end"],
                "trade_start": wf_round["trade_start"],
                "trade_end": wf_round["trade_end"],
                "SELECTION_MODE": "round_error",
                "WFA_SELECTED_PASS": False,
                "PARAM_SELECTED": False,
                "ROUND_ERROR": str(exc),
            }
        ]
    )


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

    for idx, row in trials_df.iterrows():
        same_family = trials_df[
            (trials_df["model"] == row["model"])
            & (trials_df["logic"] == row["logic"])
            & (trials_df["side"] == row["side"])
            & (trials_df["TRAIN_SR"].notna())
            & ((trials_df["window"] - row["window"]).abs() <= PLATEAU_WINDOW_DELTA)
            & ((trials_df["threshold_1"] - row["threshold_1"]).abs() <= PLATEAU_THRESHOLD_DELTA)
            & ((trials_df["threshold_2"] - row["threshold_2"]).abs() <= PLATEAU_THRESHOLD_DELTA)
        ]

        if same_family.empty:
            continue

        pass_mask = same_family.apply(lambda item: passes_period_filters(item, "TRAIN"), axis=1)
        pass_count = int(pass_mask.sum())
        pass_rate = pass_count / len(same_family)
        q25_sr = same_family["TRAIN_SR"].quantile(0.25)
        mean_sr = same_family["TRAIN_SR"].mean()
        median_sr = same_family["TRAIN_SR"].median()
        worst_mdd = same_family["TRAIN_MDD"].min()
        neighbor_penalty = 0.0 if len(same_family) >= PLATEAU_MIN_NEIGHBORS else 0.25

        trials_df.loc[idx, "plateau_count"] = len(same_family)
        trials_df.loc[idx, "plateau_pass_count"] = pass_count
        trials_df.loc[idx, "plateau_pass_rate"] = round(pass_rate, 4)
        trials_df.loc[idx, "plateau_mean_sr"] = round(mean_sr, 4)
        trials_df.loc[idx, "plateau_median_sr"] = round(median_sr, 4)
        trials_df.loc[idx, "plateau_q25_sr"] = round(q25_sr, 4)
        trials_df.loc[idx, "plateau_worst_mdd"] = round(worst_mdd, 4)
        trials_df.loc[idx, "plateau_score"] = round(
            q25_sr + 0.25 * pass_rate + 0.05 * np.log1p(pass_count) - neighbor_penalty,
            4,
        )

    return trials_df


def select_train_pass_candidates(trials_df):
    train_pass_df = trials_df[trials_df.apply(lambda row: passes_period_filters(row, "TRAIN"), axis=1)].copy()
    if train_pass_df.empty:
        return train_pass_df

    return train_pass_df.sort_values(
        [
            "TRAIN_SR",
            "plateau_score",
            "plateau_q25_sr",
            "plateau_pass_count",
            "TRAIN_MDD",
        ],
        ascending=[False, False, False, False, False],
        na_position="last",
    )


def select_best_available_trial(trials_df):
    sort_cols = [
        "objective",
        "TRAIN_SR",
        "plateau_score",
        "plateau_q25_sr",
        "plateau_pass_count",
        "TRAIN_MDD",
    ]
    return trials_df.sort_values(
        sort_cols,
        ascending=[False, False, False, False, False, False],
        na_position="last",
    ).head(1).copy()


def run_tpe_for_round(alpha, factor_df, price_df, strategy, wf_round):
    alpha_id = alpha.get("alpha_id")
    model_options = build_wf_model_options(strategy["model"])
    logic_options = build_wf_logic_options(strategy["logic"])
    side_options = tpe.build_side_options(strategy["side"])
    trial_rows = []

    def objective(trial, fixed_model=None):
        model = fixed_model or trial.suggest_categorical("model", model_options)
        logic = trial.suggest_categorical("logic", logic_options)
        side = trial.suggest_categorical("side", side_options)
        window = trial.suggest_int("window", tpe.WINDOW_MIN, tpe.WINDOW_MAX, step=tpe.WINDOW_STEP)
        t1, t2 = tpe.suggest_thresholds(trial, model)

        try:
            train_metrics, _ = tpe.evaluate_period(
                factor_df,
                price_df,
                wf_round["train_start"],
                wf_round["train_end"],
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
                    "round": wf_round["round"],
                    "train_start": wf_round["train_start"],
                    "train_end": wf_round["train_end"],
                    "test_start": wf_round["test_start"],
                    "test_end": wf_round["test_end"],
                    "trade_start": wf_round["trade_start"],
                    "trade_end": wf_round["trade_end"],
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

        score = tpe.objective_score(train_metrics, side=side)
        trial_rows.append(
            {
                "alpha_id": alpha_id,
                "round": wf_round["round"],
                "train_start": wf_round["train_start"],
                "train_end": wf_round["train_end"],
                "test_start": wf_round["test_start"],
                "test_end": wf_round["test_end"],
                "trade_start": wf_round["trade_start"],
                "trade_end": wf_round["trade_end"],
                "trial": trial.number,
                "model": model,
                "logic": logic,
                "side": side,
                "window": window,
                "threshold_1": t1,
                "threshold_2": t2,
                "objective": score,
                **metrics_to_row("TRAIN", train_metrics),
            }
        )
        return score

    def run_study(fixed_model=None, study_seed=None, desc=None):
        sampler = optuna.samplers.TPESampler(seed=study_seed or tpe.SEED + int(wf_round["round"]))
        study = optuna.create_study(sampler=sampler, direction="maximize")
        should_enqueue = ENQUEUE_BASE_TRIAL and (fixed_model is None or fixed_model == strategy["model"])
        if should_enqueue:
            study.enqueue_trial(
                {
                    **({} if fixed_model else {"model": strategy["model"]}),
                    "logic": strategy["logic"],
                    "side": strategy["side"],
                    "window": strategy["window"],
                    "threshold_1": strategy["t1"],
                    "threshold_2": strategy["t2"],
                }
            )

        with tqdm(total=tpe.N_TRIALS, desc=desc or f"{alpha_id} WF{wf_round['round']} TPE", unit="trial") as pbar:
            def callback(_study, _trial):
                pbar.update(1)

            study.optimize(lambda trial: objective(trial, fixed_model=fixed_model), n_trials=tpe.N_TRIALS, callbacks=[callback])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if tpe.MODEL_SEARCH_MODE == "per_model":
        base_seed = tpe.SEED + int(wf_round["round"]) * 1000
        for idx, model in enumerate(model_options):
            run_study(fixed_model=model, study_seed=base_seed + idx, desc=f"{alpha_id} WF{wf_round['round']} TPE {model}")
    elif tpe.MODEL_SEARCH_MODE == "categorical":
        run_study()
    else:
        raise ValueError(f"unsupported TPE_MODEL_SEARCH_MODE: {tpe.MODEL_SEARCH_MODE}")

    trials_df = pd.DataFrame(trial_rows)
    if trials_df.empty:
        raise ValueError(f"No completed trials for {alpha_id} round {wf_round['round']}")
    trials_df["trial"] = range(len(trials_df))

    for col in ["TRAIN_SR", "TRAIN_CR", "TRAIN_MDD", "TRAIN_AR", "TRAIN_TR", "TRAIN_TPI", "TRAIN_num_trades"]:
        if col not in trials_df.columns:
            trials_df[col] = np.nan

    trials_df = add_plateau_scores(trials_df)
    selected_df = select_train_pass_candidates(trials_df).copy()
    if selected_df.empty:
        selected_df = select_best_available_trial(trials_df)
        selected_df["TRAIN_FALLBACK_SELECTED"] = True
    else:
        selected_df["TRAIN_FALLBACK_SELECTED"] = False

    eval_cols = [
        "TEST_SR",
        "TEST_CR",
        "TEST_MDD",
        "TEST_AR",
        "TEST_TR",
        "TEST_TPI",
        "TEST_num_trades",
        "TEST_error",
        "TRADE_SR",
        "TRADE_CR",
        "TRADE_MDD",
        "TRADE_AR",
        "TRADE_TR",
        "TRADE_TPI",
        "TRADE_num_trades",
        "TRADE_error",
    ]
    for col in eval_cols:
        trials_df[col] = "" if col.endswith("_error") else np.nan
        selected_df[col] = "" if col.endswith("_error") else np.nan

    for idx, candidate in tqdm(
        selected_df.iterrows(),
        total=len(selected_df),
        desc=f"{alpha_id} WF{wf_round['round']} TEST/TRADE",
        unit="row",
    ):
        try:
            test_metrics, _ = tpe.evaluate_period(
                factor_df,
                price_df,
                wf_round["test_start"],
                wf_round["test_end"],
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
            selected_df.loc[idx, "TEST_error"] = str(exc)
            trials_df.loc[idx, "TEST_error"] = str(exc)
        else:
            for key, value in metrics_to_row("TEST", test_metrics).items():
                selected_df.loc[idx, key] = value
                trials_df.loc[idx, key] = value

        try:
            trade_metrics, _ = tpe.evaluate_period(
                factor_df,
                price_df,
                wf_round["trade_start"],
                wf_round["trade_end"],
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
            selected_df.loc[idx, "TRADE_error"] = str(exc)
            trials_df.loc[idx, "TRADE_error"] = str(exc)
            continue

        for key, value in metrics_to_row("TRADE", trade_metrics).items():
            selected_df.loc[idx, key] = value
            trials_df.loc[idx, key] = value

    selected_df["TRAIN_PASS"] = selected_df.apply(lambda row: tpe.selection_scoring_lib.period_pass(row, "TRAIN", row.get("side"), min_sr=tpe.MIN_SR, max_mdd=tpe.MIN_MDD), axis=1)
    selected_df[["TEST_MIN_SR", "TEST_MAX_SR"]] = selected_df.apply(
        lambda row: pd.Series(tpe.validation_sr_bounds(row, "TRAIN")),
        axis=1,
    )
    selected_df["TEST_SR_WITHIN_TRAIN_TOLERANCE"] = selected_df.apply(
        lambda row: not pd.isna(row.get("TEST_SR"))
        and row.get("TEST_SR") >= row.get("TEST_MIN_SR")
        and (pd.isna(row.get("TEST_MAX_SR")) or row.get("TEST_SR") <= row.get("TEST_MAX_SR")),
        axis=1,
    )
    selected_df["TEST_PASS"] = selected_df.apply(lambda row: tpe.validation_period_pass(row, "TRAIN", "TEST", row.get("side")), axis=1)
    selected_df["TRAIN_TEST_PASS"] = selected_df["TRAIN_PASS"] & selected_df["TEST_PASS"]
    selected_df["FALLBACK_SCORE"] = selected_df.apply(soft_fallback_score, axis=1)
    selected_df["WF_SELECTION_SCORE"] = selected_df["FALLBACK_SCORE"]
    selected_df["SELECTION_SCORE"] = selected_df["WF_SELECTION_SCORE"]
    selected_df["TRADE_PASS"] = selected_df.apply(lambda row: tpe.selection_scoring_lib.period_pass(row, "TRADE", row.get("side"), min_sr=1.0, max_mdd=tpe.MIN_MDD), axis=1)
    selected_df["TRADE_SR_POSITIVE"] = pd.to_numeric(selected_df["TRADE_SR"], errors="coerce").gt(0)
    selected_df["TRADE_AUDIT_ONLY"] = True
    selected_df["SR_DRIFT_TRAIN_TEST"] = selected_df.apply(lambda row: tpe.selection_scoring_lib.sr_drift(row, "TRAIN", "TEST"), axis=1)
    selected_df["MDD_DRIFT_TRAIN_TEST"] = selected_df.apply(lambda row: tpe.selection_scoring_lib.mdd_drift(row, "TRAIN", "TEST"), axis=1)
    selected_df["TPI_DRIFT_TRAIN_TEST"] = selected_df.apply(lambda row: tpe.selection_scoring_lib.tpi_drift(row, "TRAIN", "TEST"), axis=1)
    selected_df["TRAIN_TEST_SR_COMBI"] = selected_df[["TRAIN_SR", "TEST_SR"]].sum(axis=1, min_count=2)
    selected_df["FILTER_PASS"] = selected_df["TRAIN_PASS"] & selected_df["TEST_PASS"]
    selected_df["FILTER_SELECT_PASS"] = selected_df["FILTER_PASS"]
    selected_df = selected_df.sort_values(
        [
            "FILTER_SELECT_PASS",
            "SELECTION_SCORE",
            "TEST_SR",
            "TRAIN_SR",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    selected_df["PARAM_SELECTED"] = False
    selected_indexes = selected_df.index[selected_df["FILTER_SELECT_PASS"]][:1]
    if len(selected_indexes):
        selected_df.loc[selected_indexes, "PARAM_SELECTED"] = True

    return trials_df, selected_df


def add_full_period_metrics(round_best, factor_df, price_df, strategy):
    round_best = round_best.copy().reset_index(drop=True)

    for col in [
        "full_start",
        "full_end",
        "FULL_SR",
        "FULL_CR",
        "FULL_MDD",
        "FULL_AR",
        "FULL_TR",
        "FULL_TPI",
        "FULL_num_trades",
        "FULL_error",
        "FULL_PASS",
    ]:
        round_best[col] = "" if col == "FULL_error" else np.nan

    for idx, row in tqdm(round_best.iterrows(), total=len(round_best), desc="Full period", unit="row"):
        round_best.loc[idx, "full_start"] = START_DATE
        round_best.loc[idx, "full_end"] = END_DATE
        try:
            for required_col in ["window", "threshold_1", "threshold_2", "logic", "side", "model"]:
                if required_col not in row or pd.isna(row[required_col]):
                    raise ValueError("missing selected params")

            full_metrics, _ = tpe.evaluate_period(
                factor_df,
                price_df,
                START_DATE,
                END_DATE,
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
            round_best.loc[idx, "FULL_error"] = str(exc)
            continue

        for key, value in metrics_to_row("FULL", full_metrics).items():
            round_best.loc[idx, key] = value
        round_best.loc[idx, "FULL_PASS"] = passes_period_filters(round_best.loc[idx], "FULL")

    return round_best


def latest_year_pass(row):
    values = {
        "sr": safe_float(row.get("L1Y_SR")),
        "mdd": safe_float(row.get("L1Y_MDD")),
        "tpi": safe_float(row.get("L1Y_TPI")),
        "num_trades": safe_float(row.get("L1Y_num_trades")),
    }
    if any(pd.isna(value) for value in values.values()):
        return False

    return (
        values["sr"] > L1Y_MIN_SR
        and values["mdd"] > L1Y_MIN_MDD
        and values["tpi"] > L1Y_MIN_TPI
        and values["num_trades"] >= L1Y_MIN_TRADES
    )


def add_latest_year_metrics(round_best, factor_df, price_df, strategy, data_end):
    round_best = round_best.copy().reset_index(drop=True)
    l1y_end = pd.Timestamp(data_end).normalize()
    l1y_start = l1y_end - pd.DateOffset(months=L1Y_MONTHS) + pd.Timedelta(days=1)

    for col in [
        "L1Y_start",
        "L1Y_end",
        "L1Y_SR",
        "L1Y_CR",
        "L1Y_MDD",
        "L1Y_AR",
        "L1Y_TR",
        "L1Y_TPI",
        "L1Y_num_trades",
        "L1Y_error",
        "L1Y_PASS",
    ]:
        if col in {"L1Y_start", "L1Y_end", "L1Y_error"}:
            round_best[col] = ""
        elif col == "L1Y_PASS":
            round_best[col] = False
        else:
            round_best[col] = np.nan

    for idx, row in tqdm(round_best.iterrows(), total=len(round_best), desc="Latest 1Y", unit="row"):
        round_best.loc[idx, "L1Y_start"] = date_str(l1y_start)
        round_best.loc[idx, "L1Y_end"] = date_str(l1y_end)
        try:
            for required_col in ["window", "threshold_1", "threshold_2", "logic", "side", "model"]:
                if required_col not in row or pd.isna(row[required_col]):
                    raise ValueError("missing selected params")

            l1y_metrics, _ = tpe.evaluate_period(
                factor_df,
                price_df,
                l1y_start,
                l1y_end,
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
            round_best.loc[idx, "L1Y_error"] = str(exc)
            round_best.loc[idx, "L1Y_PASS"] = False
            continue

        for key, value in metrics_to_row("L1Y", l1y_metrics).items():
            round_best.loc[idx, key] = value
        round_best.loc[idx, "L1Y_PASS"] = latest_year_pass(round_best.loc[idx])

    return round_best


def bool_series(df, column):
    column_name = next((col for col in df.columns if str(col).strip() == column), None)
    if column_name is None:
        return pd.Series(False, index=df.index)

    values = df[column_name]
    if values.dtype == bool:
        return values.fillna(False)

    return values.fillna(False).map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )


def wfa_selected_summary_path(alpha_dir, alpha_id, round_best):
    suffix = f"{TRAIN_MONTHS}_{TEST_MONTHS}_{TRADE_MONTHS}_{WF_STEP_MONTHS}"
    full_pass_suffix = "_full_pass" if bool_series(round_best, "FULL_PASS").any() else ""
    return alpha_dir / f"{alpha_id}_wfa_selected_summary_{suffix}{full_pass_suffix}.csv"


def add_wfa_summary_metrics(round_best):
    round_best = round_best.copy()
    summary_cols = [
        "WFA_OOS_SR_MEDIAN",
        "WFA_OOS_SR_MEAN",
        "WFA_OOS_SR_Q25",
        "WFA_PASS_COUNT",
        "WFA_ROUND_COUNT",
        "WFA_PASS_RATE",
        "WFA_PASS_RATE_FILTER",
    ]

    for col in summary_cols:
        round_best[col] = np.nan

    if round_best.empty:
        return round_best

    oos_sr_values = round_best["TRADE_SR"] if "TRADE_SR" in round_best.columns else pd.Series(np.nan, index=round_best.index)
    oos_sr = pd.to_numeric(oos_sr_values, errors="coerce")
    pass_mask = bool_series(round_best, "TRAIN_PASS") & bool_series(round_best, "TEST_PASS")
    round_count = len(round_best)
    pass_count = int(pass_mask.sum())
    pass_rate = pass_count / round_count if round_count else np.nan

    round_best["WFA_OOS_SR_MEDIAN"] = round(oos_sr.median(), 4) if oos_sr.notna().any() else np.nan
    round_best["WFA_OOS_SR_MEAN"] = round(oos_sr.mean(), 4) if oos_sr.notna().any() else np.nan
    round_best["WFA_OOS_SR_Q25"] = round(oos_sr.quantile(0.25), 4) if oos_sr.notna().any() else np.nan
    round_best["WFA_PASS_COUNT"] = pass_count
    round_best["WFA_ROUND_COUNT"] = round_count
    round_best["WFA_PASS_RATE"] = round(pass_rate, 4) if not pd.isna(pass_rate) else np.nan
    round_best["WFA_PASS_RATE_FILTER"] = pass_rate > WFA_pass_rate_threshold if not pd.isna(pass_rate) else False

    return round_best


def run_walk_forward(alpha):
    alpha_id = alpha.get("alpha_id")
    print(f"\nWalk-forward TPE for {alpha_id}")

    factor_df, alpha_formula = tpe.build_factor_dataframe(alpha)
    strategy = tpe.get_strategy_config(alpha)
    if strategy["resolution"] is None:
        raise ValueError(f"Unsupported timeframe: {strategy['timeframe']}")

    price_df = tpe.load_price_df(
        trade_asset=alpha.get("trade_asset", "BTC"),
        timeframe=strategy["timeframe"],
        price_delay=int(alpha.get("shift_backtest_candle_minute", 0)),
    )
    data_end = infer_data_end(factor_df, price_df)
    wf_rounds = build_wf_rounds(data_end)
    if not wf_rounds:
        raise ValueError(f"No walk-forward rounds available for {alpha_id} through {date_str(data_end)}")

    alpha_dir = OUTPUT_DIR / alpha_id
    alpha_dir.mkdir(parents=True, exist_ok=True)

    all_param_selected = []
    for wf_round in wf_rounds:
        print(
            f"round {wf_round['round']}: train {wf_round['train_start']} to {wf_round['train_end']}, "
            f"test {wf_round['test_start']} to {wf_round['test_end']}, "
            f"trade {wf_round['trade_start']} to {wf_round['trade_end']}"
        )
        try:
            trials_df, selected_df = run_tpe_for_round(alpha, factor_df, price_df, strategy, wf_round)
            param_selected_df = selected_params_df(selected_df)
        except Exception as exc:
            print(f"round {wf_round['round']} selected: failed ({exc})")
            param_selected_df = round_error_df(alpha, wf_round, exc)
        all_param_selected.append(param_selected_df)

        if param_selected_df.empty:
            print(f"round {wf_round['round']} selected: none available")
        else:
            best = param_selected_df.iloc[0]
            if best.get("SELECTION_MODE") == "round_error":
                print(f"round {wf_round['round']} selected: mode=round_error error={best.get('ROUND_ERROR')}")
                continue
            print(
                f"round {wf_round['round']} selected: "
                f"mode={best['SELECTION_MODE']} "
                f"model={best['model']} logic={best['logic']} side={best['side']} "
                f"window={int(best['window'])} t1={best['threshold_1']} t2={best['threshold_2']} "
                f"TRAIN_SR={best['TRAIN_SR']} TEST_SR={best['TEST_SR']} TRADE_SR={best['TRADE_SR']} "
                f"TRAIN_MDD={best['TRAIN_MDD']} TEST_MDD={best['TEST_MDD']} TRADE_MDD={best['TRADE_MDD']} "
                f"TRAIN_TPI={best['TRAIN_TPI']} TEST_TPI={best['TEST_TPI']} TRADE_TPI={best['TRADE_TPI']} "
                f"plateau_score={best['plateau_score']} train_test_pass={best['TRAIN_TEST_PASS']} "
                f"trade_sr_positive={best['TRADE_SR_POSITIVE']} "
                f"selection_score={best['WF_SELECTION_SCORE']} trade_audit_only={best['TRADE_AUDIT_ONLY']}"
            )

    param_selected_summary = pd.concat(all_param_selected, ignore_index=True)
    round_best_full = add_full_period_metrics(param_selected_summary, factor_df, price_df, strategy)
    round_best_full = add_latest_year_metrics(round_best_full, factor_df, price_df, strategy, data_end)
    round_best_full = add_wfa_summary_metrics(round_best_full)
    summary_path = wfa_selected_summary_path(alpha_dir, alpha_id, round_best_full)
    round_best_full.to_csv(summary_path, index=False)

    if round_best_full.empty:
        print("round best full-period backtest: none available")
    else:
        print(
            "round best full-period backtest:\n"
            + round_best_full[
                [
                    "round",
                    "SELECTION_MODE",
                    "model",
                    "logic",
                    "side",
                    "window",
                    "threshold_1",
                    "threshold_2",
                    "TRAIN_SR",
                    "TRAIN_MDD",
                    "TRAIN_TPI",
                    "TEST_SR",
                    "TEST_MDD",
                    "TEST_TPI",
                    "TRADE_SR",
                    "FULL_SR",
                    "FULL_MDD",
                    "FULL_TPI",
                    "L1Y_SR",
                    "L1Y_MDD",
                    "L1Y_TPI",
                    "L1Y_num_trades",
                    "TRAIN_TEST_PASS",
                    "TRADE_SR_POSITIVE",
                    "WF_SELECTION_SCORE",
                    "TRADE_AUDIT_ONLY",
                    "TRADE_PASS",
                    "FULL_PASS",
                    "L1Y_PASS",
                    "WFA_OOS_SR_MEDIAN",
                    "WFA_OOS_SR_MEAN",
                    "WFA_OOS_SR_Q25",
                    "WFA_PASS_RATE",
                    "WFA_PASS_RATE_FILTER",
                ]
            ].to_string(index=False)
        )

    print(f"formula: {alpha_formula}")
    print(
        f"base: model={strategy['model']} logic={strategy['logic']} "
        f"window={strategy['window']} t1={strategy['t1']} t2={strategy['t2']}"
    )
    print(f"summary saved: {summary_path}")

    return round_best_full


def main():
    with open(JSON_FILE, "r", encoding="utf-8") as file:
        alphas = json.load(file)

    selected_alphas = selected_wf_alphas(alphas)
    target_ids = target_alpha_ids()
    target_label = "ALL" if target_ids is None else ", ".join(target_ids)
    print(f"walk-forward target alpha: {target_label}")
    print(f"walk-forward alpha count: {len(selected_alphas)}")

    errors = []
    for alpha in selected_alphas:
        alpha_id = alpha.get("alpha_id", "")
        try:
            run_walk_forward(alpha)
        except Exception as exc:
            print(f"\nERROR {alpha_id}: {exc}")
            errors.append({"alpha_id": alpha_id, "error": str(exc)})

    if errors:
        print(f"errors: {pd.DataFrame(errors).to_string(index=False)}")


if __name__ == "__main__":
    main()

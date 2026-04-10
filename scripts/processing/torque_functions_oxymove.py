import json
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import ttest_1samp, ttest_rel


DEFAULT_BASE_ROOT = r"C:\Program Files\DigiMove\DigiMove\DataAnalysisProject\data\MOXY-bids"
DEFAULT_WINDOW = 30
DEFAULT_PEAK_MIN_INTERVAL = 2.6
TASK_BY_SESSION = {"001": "concentric", "002": "eccentric"}


# -----------------------------------------------------------------------------
# Path and loading helpers
# -----------------------------------------------------------------------------
def _normalize_subject_id(subject_id: str) -> str:
    return str(subject_id).replace("sub-", "")


def _list_subject_ids(base_root: str) -> list[str]:
    return sorted(
        d.replace("sub-", "")
        for d in os.listdir(base_root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(base_root, d))
    )


def build_beh_file_paths(
    subject_id: str,
    session_id: str,
    task: str,
    base_root: str = DEFAULT_BASE_ROOT,
) -> Dict[str, str]:
    """Build BIDS-like paths for torque TSV.GZ, events TSV and sidecar JSON."""
    subject_id = _normalize_subject_id(subject_id)
    base = os.path.join(
        base_root,
        f"sub-{subject_id}",
        f"ses-{session_id}",
        "beh",
        f"sub-{subject_id}_ses-{session_id}_task-{task}",
    )
    return {
        "torque": f"{base}_recording-torque_physio.tsv.gz",
        "events": f"{base}_events.tsv",
        "json": f"{base}_recording-torque_physio.json",
    }


def _find_session_for_task(subject_id: str, task_name: str, base_root: str) -> Optional[str]:
    """Find first session containing all required files for one subject/task."""
    subject_id = _normalize_subject_id(subject_id)
    sub_dir = os.path.join(base_root, f"sub-{subject_id}")
    if not os.path.isdir(sub_dir):
        return None

    for ses_name in sorted(os.listdir(sub_dir)):
        if not ses_name.startswith("ses-"):
            continue
        ses_id = ses_name.replace("ses-", "")
        paths = build_beh_file_paths(subject_id, ses_id, task_name, base_root=base_root)
        if all(os.path.exists(p) for p in paths.values()):
            return ses_id
    return None


def infer_task_from_session(
    subject_id: str,
    session_id: str,
    base_root: str = DEFAULT_BASE_ROOT,
    task_choice: Optional[str] = None,
) -> str:
    """Infer task from events file names for a given subject/session."""
    if task_choice is not None:
        return task_choice

    subject_id = _normalize_subject_id(subject_id)
    beh_dir = os.path.join(base_root, f"sub-{subject_id}", f"ses-{session_id}", "beh")
    prefix = f"sub-{subject_id}_ses-{session_id}_task-"
    suffix = "_events.tsv"

    if os.path.isdir(beh_dir):
        tasks = [
            name[len(prefix):-len(suffix)]
            for name in os.listdir(beh_dir)
            if name.startswith(prefix) and name.endswith(suffix)
        ]
        unique_tasks = sorted(set(tasks))
        if len(unique_tasks) == 1:
            return unique_tasks[0]
        if len(unique_tasks) > 1:
            raise ValueError(
                f"Multiple tasks found for sub-{subject_id} ses-{session_id}: {unique_tasks}. "
                "Set task_choice explicitly."
            )

    task = TASK_BY_SESSION.get(session_id)
    if task is None:
        raise ValueError(
            f"Cannot infer task for session '{session_id}'. Set task_choice explicitly."
        )
    return task


def load_task_data(torque_file: str, events_file: str, json_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load torque signal, events and sampling frequency from one task triplet."""
    df_source = pd.read_csv(torque_file, sep="\t", header=None, names=["torque_Nm"])
    events_source = pd.read_csv(events_file, sep="\t")
    with open(json_file, "r", encoding="utf-8") as f:
        fs_source = float(json.load(f)["SamplingFrequency"])

    df_source["time"] = np.arange(len(df_source), dtype=float) / fs_source
    return df_source, events_source, fs_source


def load_session_data(
    subject_id: str,
    session_id: str,
    task_choice: Optional[str] = None,
    base_root: str = DEFAULT_BASE_ROOT,
) -> Dict[str, Any]:
    """Load one subject/session torque dataset with optional task override."""
    subject_id = _normalize_subject_id(subject_id)
    task = infer_task_from_session(subject_id, session_id, base_root=base_root, task_choice=task_choice)
    paths = build_beh_file_paths(subject_id, session_id, task, base_root=base_root)

    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key.capitalize()} file not found: {path}")

    df_source, events_source, fs_source = load_task_data(paths["torque"], paths["events"], paths["json"])
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "task": task,
        "paths": paths,
        "df": df_source,
        "events": events_source,
        "fs": fs_source,
    }


def load_concentric_session(subject_id: str, session_id: str, base_root: str = DEFAULT_BASE_ROOT):
    """Compatibility helper returning concentric dataframe, fs, Con30 and Con50 events."""
    try:
        session = load_session_data(subject_id, session_id, task_choice="concentric", base_root=base_root)
        events = session["events"]
        return True, session["df"], session["fs"], events[events["trial_type"] == "Con30"], events[events["trial_type"] == "Con50"]
    except Exception as e:
        print(f"Warning: could not load concentric data: {e}")
        return False, pd.DataFrame(), np.nan, pd.DataFrame(), pd.DataFrame()


def load_eccentric_session(subject_id: str, session_id: str, base_root: str = DEFAULT_BASE_ROOT):
    """Compatibility helper returning eccentric dataframe, fs, Ecc30 and Ecc50 events."""
    try:
        session = load_session_data(subject_id, session_id, task_choice="eccentric", base_root=base_root)
        events = session["events"]
        return True, session["df"], session["fs"], events[events["trial_type"] == "Ecc30"], events[events["trial_type"] == "Ecc50"]
    except Exception as e:
        print(f"Warning: could not load eccentric data: {e}")
        return False, pd.DataFrame(), np.nan, pd.DataFrame(), pd.DataFrame()


# -----------------------------------------------------------------------------
# Metrics and signal processing
# -----------------------------------------------------------------------------
def detect_peaks_in_segment(
    segment_data: pd.Series,
    sampling_freq: float,
    height_threshold: float,
    direction: str = "positive",
    peak_min_interval: float = DEFAULT_PEAK_MIN_INTERVAL,
):
    """Detect peaks with a minimum spacing and configurable direction."""
    distance = max(1, int(sampling_freq * peak_min_interval))
    signal = segment_data.to_numpy()

    if direction == "negative":
        peak_indices, _ = find_peaks(-signal, height=height_threshold, distance=distance)
    elif direction == "absolute":
        peak_indices, _ = find_peaks(np.abs(signal), height=height_threshold, distance=distance)
    else:
        peak_indices, _ = find_peaks(signal, height=height_threshold, distance=distance)

    peak_values = signal[peak_indices]
    return np.asarray(peak_values), segment_data.index[peak_indices].tolist()


def process_series(
    df_source: pd.DataFrame,
    events_source: pd.DataFrame,
    label: str,
    threshold: float,
    sampling_freq: float,
    direction: str,
    window: int = DEFAULT_WINDOW,
    onset_offset: float = 0.0,
    peak_min_interval: float = DEFAULT_PEAK_MIN_INTERVAL,
    verbose: bool = True,
):
    """Return mean force per set and all detected peaks for one condition."""
    forces, all_peaks = [], []
    for _, row in events_source.iterrows():
        t0 = row["onset"] + onset_offset
        segment = df_source[(df_source["time"] >= t0) & (df_source["time"] <= t0 + window)]["torque_Nm"]
        peaks, _ = detect_peaks_in_segment(
            segment,
            sampling_freq=sampling_freq,
            height_threshold=threshold,
            direction=direction,
            peak_min_interval=peak_min_interval,
        )

        if len(peaks) > 0:
            m = float(np.mean(peaks))
            forces.append(m)
            all_peaks.extend(peaks.tolist())
            if verbose:
                print(f"{label} set {len(forces)}: {len(peaks)} peaks detected, mean = {m:.3f} Nm")
        elif verbose:
            print(f"{label} set: no peak detected")

    return forces, all_peaks


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    """Return mean/std for values, NaN/NaN for empty input."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr))


def compute_average_mvc(df_source: pd.DataFrame, events_source: pd.DataFrame, mvc_window: int = 6) -> float:
    """Return MVC reference as the highest max torque across MVC sets."""
    mvc_values = []
    for _, row in events_source[events_source["trial_type"] == "MVC"].iterrows():
        start = row["onset"]
        end = start + mvc_window
        segment = df_source[(df_source["time"] >= start) & (df_source["time"] <= end)]
        mvc_values.append(float(segment["torque_Nm"].max()))

    if not mvc_values:
        return np.nan
    return float(np.max(mvc_values))


def compute_mvc_quality_stats(
    df_source: pd.DataFrame,
    events_source: pd.DataFrame,
    mvc_window: int = 6,
    reference_values: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
    tolerance_pct: float = 0.15,
) -> Dict[str, Any]:
    """Compute MVC quality using last MVC within +/- tolerance around best of first 3 MVCs."""
    mvc_events = events_source[events_source["trial_type"] == "MVC"]
    mvc_max_values = []

    for _, row in mvc_events.iterrows():
        start = row["onset"]
        end = start + mvc_window
        segment = df_source[(df_source["time"] >= start) & (df_source["time"] <= end)]
        mvc_max_values.append(float(segment["torque_Nm"].max()))

    print("Max torque for each MVC:", mvc_max_values)

    if len(mvc_max_values) >= 3:
        average_mvc_max = float(np.mean(mvc_max_values[:3]))
    elif mvc_max_values:
        average_mvc_max = float(np.mean(mvc_max_values))
    else:
        average_mvc_max = np.nan

    print(f"Average Max Torque across the 3 first MVCs: {average_mvc_max} Nm")

    values_for_test = list(reference_values) if reference_values is not None else mvc_max_values
    t_stat, p_value = np.nan, np.nan
    group_A = np.array([])
    last = np.nan
    is_subject_eligible = False
    exclusion_reason = "No MVC trial available for quality test."
    best_mvc_first3 = np.nan
    lower_bound = np.nan
    upper_bound = np.nan
    n_reference_mvc = 0

    if len(values_for_test) >= 2:
        reference_values_for_rule = values_for_test[:-1][:3]
        group_A = np.asarray(reference_values_for_rule, dtype=float)
        last = float(values_for_test[-1])
        n_reference_mvc = int(group_A.size)
        best_mvc_first3 = float(np.max(group_A))
        lower_bound = float(best_mvc_first3 * (1.0 - tolerance_pct))
        upper_bound = float(best_mvc_first3 * (1.0 + tolerance_pct))

        print(f"Best MVC among reference trials: {best_mvc_first3:.3f} Nm")
        print(
            f"Acceptance range (+/- {tolerance_pct * 100:.1f}%): "
            f"[{lower_bound:.3f}, {upper_bound:.3f}] Nm"
        )
        print(f"Last MVC: {last:.3f} Nm")

        is_subject_eligible = bool(lower_bound <= last <= upper_bound)
        if is_subject_eligible:
            exclusion_reason = ""
            print("MVC quality check PASSED. Subject can be used.")
        else:
            exclusion_reason = (
                "MVC quality check FAILED: last MVC is outside +/- tolerance "
                "around best reference MVC."
            )
            print("MVC quality check FAILED. Subject should be excluded from analysis.")
    elif len(values_for_test) == 1:
        last = float(values_for_test[0])
        best_mvc_first3 = float(values_for_test[0])
        lower_bound = float(best_mvc_first3 * (1.0 - tolerance_pct))
        upper_bound = float(best_mvc_first3 * (1.0 + tolerance_pct))
        n_reference_mvc = 0
        is_subject_eligible = True
        exclusion_reason = ""
        print("Only one MVC trial available: quality check marked as PASS by convention.")
    else:
        print("No MVC value found: quality check cannot be evaluated.")

    return {
        "mvc_events": mvc_events,
        "mvc_max_values": mvc_max_values,
        "average_mvc_max": average_mvc_max,
        "group_A": group_A,
        "n_reference_mvc": n_reference_mvc,
        "last": last,
        "t_stat": t_stat,
        "p_value": p_value,
        "alpha": alpha,
        "method": "best_first3_with_tolerance",
        "tolerance_pct": tolerance_pct,
        "best_mvc_first3": best_mvc_first3,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "is_subject_eligible": is_subject_eligible,
        "exclusion_reason": exclusion_reason,
    }


def evaluate_subject_mvc_eligibility(
    subject_id: str,
    session_id: str,
    base_root: str = DEFAULT_BASE_ROOT,
    task_choice: Optional[str] = None,
    mvc_window: int = 6,
    alpha: float = 0.05,
    tolerance_pct: float = 0.15,
) -> Dict[str, Any]:
    """Evaluate one subject/session MVC quality and return structured status."""
    subject_id = _normalize_subject_id(subject_id)
    session = load_session_data(
        subject_id=subject_id,
        session_id=session_id,
        task_choice=task_choice,
        base_root=base_root,
    )

    mvc_stats = compute_mvc_quality_stats(
        df_source=session["df"],
        events_source=session["events"],
        mvc_window=mvc_window,
        reference_values=None,
        alpha=alpha,
        tolerance_pct=tolerance_pct,
    )

    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "task": session["task"],
        "n_mvc": int(len(mvc_stats["mvc_max_values"])),
        "average_mvc_max": mvc_stats["average_mvc_max"],
        "best_mvc_first3": mvc_stats["best_mvc_first3"],
        "last_mvc": mvc_stats["last"],
        "lower_bound": mvc_stats["lower_bound"],
        "upper_bound": mvc_stats["upper_bound"],
        "tolerance_pct": mvc_stats["tolerance_pct"],
        "p_value": mvc_stats["p_value"],
        "alpha": mvc_stats["alpha"],
        "is_subject_eligible": mvc_stats["is_subject_eligible"],
        "reason": mvc_stats["exclusion_reason"],
        "mvc_stats": mvc_stats,
    }


def summarize_mvc_eligibility_all_subjects(
    session_id: str,
    base_root: str = DEFAULT_BASE_ROOT,
    task_choice: Optional[str] = None,
    mvc_window: int = 6,
    alpha: float = 0.05,
    tolerance_pct: float = 0.15,
    print_summary: bool = True,
) -> pd.DataFrame:
    """Evaluate MVC quality for all subjects and return a summary table."""
    rows = []
    for subject_id in _list_subject_ids(base_root):
        try:
            result = evaluate_subject_mvc_eligibility(
                subject_id=subject_id,
                session_id=session_id,
                base_root=base_root,
                task_choice=task_choice,
                mvc_window=mvc_window,
                alpha=alpha,
                tolerance_pct=tolerance_pct,
            )
            status = "OK" if result["is_subject_eligible"] else "NOT_OK"
            rows.append(
                {
                    "Subject": result["subject_id"],
                    "Session": result["session_id"],
                    "Task": result["task"],
                    "Status": status,
                    "n_MVC": result["n_mvc"],
                    "Best_MVC_First3_Nm": result["best_mvc_first3"],
                    "Last_MVC_Nm": result["last_mvc"],
                    "Lower_Bound_Nm": result["lower_bound"],
                    "Upper_Bound_Nm": result["upper_bound"],
                    "Tolerance_pct": result["tolerance_pct"] * 100.0,
                    "p_value": result["p_value"],
                    "alpha": result["alpha"],
                    "Average_MVC_Nm": result["average_mvc_max"],
                    "Reason": result["reason"],
                }
            )
        except Exception as e:
            rows.append(
                {
                    "Subject": subject_id,
                    "Session": session_id,
                    "Task": task_choice if task_choice is not None else "auto",
                    "Status": "ERROR",
                    "n_MVC": np.nan,
                    "Best_MVC_First3_Nm": np.nan,
                    "Last_MVC_Nm": np.nan,
                    "Lower_Bound_Nm": np.nan,
                    "Upper_Bound_Nm": np.nan,
                    "Tolerance_pct": tolerance_pct * 100.0,
                    "p_value": np.nan,
                    "alpha": alpha,
                    "Average_MVC_Nm": np.nan,
                    "Reason": str(e),
                }
            )

    summary_df = pd.DataFrame(rows)

    if print_summary:
        print("\n" + "=" * 70)
        print(f"MVC ELIGIBILITY SUMMARY - session {session_id}")
        print("=" * 70)
        for _, row in summary_df.iterrows():
            if np.isfinite(row["Last_MVC_Nm"]) and np.isfinite(row["Lower_Bound_Nm"]) and np.isfinite(row["Upper_Bound_Nm"]):
                rule_txt = (
                    f"last={row['Last_MVC_Nm']:.2f} Nm, "
                    f"range=[{row['Lower_Bound_Nm']:.2f}, {row['Upper_Bound_Nm']:.2f}] Nm"
                )
            else:
                rule_txt = "rule data unavailable"
            print(f"sub-{row['Subject']}: {row['Status']} ({rule_txt})")
        print("=" * 70)

    return summary_df


def summarize_mvc_eligibility_all_sessions(
    base_root: str = DEFAULT_BASE_ROOT,
    mvc_window: int = 6,
    alpha: float = 0.05,
    tolerance_pct: float = 0.15,
    print_summary: bool = True,
) -> pd.DataFrame:
    """Evaluate MVC quality for all subjects and all available sessions."""
    rows = []
    for subject_id in _list_subject_ids(base_root):
        sub_folder = os.path.join(base_root, f"sub-{subject_id}")
        if not os.path.isdir(sub_folder):
            continue

        sessions = sorted(
            d.replace("ses-", "")
            for d in os.listdir(sub_folder)
            if d.startswith("ses-") and os.path.isdir(os.path.join(sub_folder, d))
        )

        for session_id in sessions:
            try:
                result = evaluate_subject_mvc_eligibility(
                    subject_id=subject_id,
                    session_id=session_id,
                    base_root=base_root,
                    task_choice=None,
                    mvc_window=mvc_window,
                    alpha=alpha,
                    tolerance_pct=tolerance_pct,
                )
                status = "OK" if result["is_subject_eligible"] else "NOT_OK"
                rows.append(
                    {
                        "Subject": result["subject_id"],
                        "Session": result["session_id"],
                        "Task": result["task"],
                        "Status": status,
                        "n_MVC": result["n_mvc"],
                    "Best_MVC_First3_Nm": result["best_mvc_first3"],
                    "Last_MVC_Nm": result["last_mvc"],
                    "Lower_Bound_Nm": result["lower_bound"],
                    "Upper_Bound_Nm": result["upper_bound"],
                    "Tolerance_pct": result["tolerance_pct"] * 100.0,
                        "p_value": result["p_value"],
                        "alpha": result["alpha"],
                        "Average_MVC_Nm": result["average_mvc_max"],
                        "Reason": result["reason"],
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "Subject": subject_id,
                        "Session": session_id,
                        "Task": "auto",
                        "Status": "ERROR",
                        "n_MVC": np.nan,
                        "Best_MVC_First3_Nm": np.nan,
                        "Last_MVC_Nm": np.nan,
                        "Lower_Bound_Nm": np.nan,
                        "Upper_Bound_Nm": np.nan,
                        "Tolerance_pct": tolerance_pct * 100.0,
                        "p_value": np.nan,
                        "alpha": alpha,
                        "Average_MVC_Nm": np.nan,
                        "Reason": str(e),
                    }
                )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["Subject", "Session"]).reset_index(drop=True)

    if print_summary:
        print("\n" + "=" * 70)
        print("MVC ELIGIBILITY SUMMARY - ALL SUBJECTS / ALL SESSIONS")
        print("=" * 70)
        if summary_df.empty:
            print("No subject/session data found.")
        else:
            for _, row in summary_df.iterrows():
                if np.isfinite(row["Last_MVC_Nm"]) and np.isfinite(row["Lower_Bound_Nm"]) and np.isfinite(row["Upper_Bound_Nm"]):
                    rule_txt = (
                        f"last={row['Last_MVC_Nm']:.2f} Nm, "
                        f"range=[{row['Lower_Bound_Nm']:.2f}, {row['Upper_Bound_Nm']:.2f}] Nm"
                    )
                else:
                    rule_txt = "rule data unavailable"
                print(
                    f"sub-{row['Subject']} | ses-{row['Session']} | {row['Task']} | "
                    f"{row['Status']} ({rule_txt})"
                )
        print("=" * 70)

    return summary_df


def paired_ttest_mvc_pre_post(
    base_root: str = DEFAULT_BASE_ROOT,
    mvc_window: int = 6,
    tolerance_pct: float = 0.15,
    include_only_eligible: bool = True,
    aggregate_by_subject: bool = True,
    make_plot: bool = True,
    print_stats: bool = True,
) -> Dict[str, Any]:
    """Run paired t-test on MVC pre (best reference) vs post (last MVC) over all subjects."""
    summary_df = summarize_mvc_eligibility_all_sessions(
        base_root=base_root,
        mvc_window=mvc_window,
        tolerance_pct=tolerance_pct,
        print_summary=False,
    )

    if summary_df.empty:
        raise RuntimeError("No MVC data available for paired t-test.")

    pairs_df = summary_df.copy()
    pairs_df = pairs_df[pairs_df["Status"].isin(["OK", "NOT_OK"])].copy()
    pairs_df = pairs_df.dropna(subset=["Best_MVC_First3_Nm", "Last_MVC_Nm"])

    if include_only_eligible:
        pairs_df = pairs_df[pairs_df["Status"] == "OK"].copy()

    if pairs_df.empty:
        raise RuntimeError("No valid pre/post MVC pairs available after filtering.")

    pairs_df["Subject"] = pairs_df["Subject"].astype(str)
    pairs_df["Session"] = pairs_df["Session"].astype(str)

    if aggregate_by_subject:
        used_df = (
            pairs_df.groupby("Subject", as_index=False)
            .agg(
                Pre_MVC_Nm=("Best_MVC_First3_Nm", "mean"),
                Post_MVC_Nm=("Last_MVC_Nm", "mean"),
                n_pairs=("Best_MVC_First3_Nm", "size"),
            )
            .sort_values("Subject")
            .reset_index(drop=True)
        )
        line_labels = [f"sub-{s}" for s in used_df["Subject"]]
    else:
        used_df = pairs_df.rename(
            columns={
                "Best_MVC_First3_Nm": "Pre_MVC_Nm",
                "Last_MVC_Nm": "Post_MVC_Nm",
            }
        )[["Subject", "Session", "Pre_MVC_Nm", "Post_MVC_Nm", "Status"]].copy()
        used_df = used_df.sort_values(["Subject", "Session"]).reset_index(drop=True)
        line_labels = [f"sub-{r.Subject} ses-{r.Session}" for r in used_df.itertuples(index=False)]

    pre = used_df["Pre_MVC_Nm"].to_numpy(dtype=float)
    post = used_df["Post_MVC_Nm"].to_numpy(dtype=float)

    t_stat, p_value = ttest_rel(pre, post)
    mean_pre = float(np.mean(pre))
    mean_post = float(np.mean(post))
    mean_diff = float(np.mean(post - pre))

    if print_stats:
        print("\n" + "=" * 70)
        print("PAIRED T-TEST: MVC PRE VS POST")
        print("=" * 70)
        print(f"N pairs: {len(pre)}")
        print(f"Mean PRE:  {mean_pre:.3f} Nm")
        print(f"Mean POST: {mean_post:.3f} Nm")
        print(f"Mean diff (POST - PRE): {mean_diff:.3f} Nm")
        print(f"t-statistic: {t_stat:.6f}")
        print(f"p-value:     {p_value:.6f}")
        print("=" * 70)

    if make_plot:
        fig, ax = plt.subplots(figsize=(9, 6))
        x = [0, 1]

        for idx, row in used_df.iterrows():
            ax.plot(x, [row["Pre_MVC_Nm"], row["Post_MVC_Nm"]], color="gray", alpha=0.5, linewidth=1.4)
            ax.scatter(x, [row["Pre_MVC_Nm"], row["Post_MVC_Nm"]], color="gray", alpha=0.7, s=25)

        ax.plot(x, [mean_pre, mean_post], color="crimson", linewidth=3, marker="o", markersize=9, label="Group mean")
        ax.set_xticks(x)
        ax.set_xticklabels(["PRE (best ref MVC)", "POST (last MVC)"])
        ax.set_ylabel("MVC (Nm)")
        ax.set_title("Paired MVC dynamics across subjects")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    return {
        "all_pairs_df": pairs_df,
        "used_pairs_df": used_df,
        "line_labels": line_labels,
        "n_pairs": int(len(pre)),
        "mean_pre": mean_pre,
        "mean_post": mean_post,
        "mean_diff_post_minus_pre": mean_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def compute_and_plot_basic_torque_stats(
    df_source: pd.DataFrame,
    time_col: str = "time",
    torque_col: str = "torque_Nm",
) -> Dict[str, float]:
    """Compute basic stats and display torque with mean +/- one std."""
    mean_torque = float(df_source[torque_col].mean())
    std_torque = float(df_source[torque_col].std())
    skewness_torque = float(df_source[torque_col].skew())
    kurtosis_torque = float(df_source[torque_col].kurtosis())

    print(f"Mean Torque: {mean_torque} Nm")
    print(f"Standard Deviation of Torque: {std_torque} Nm")
    print(f"Skewness of Torque: {skewness_torque}")
    print(f"Kurtosis of Torque: {kurtosis_torque}")

    plt.plot(df_source[time_col], df_source[torque_col])
    plt.axhline(mean_torque, color="red", linestyle="--", label="Mean Torque")
    plt.fill_between(
        df_source[time_col],
        mean_torque - std_torque,
        mean_torque + std_torque,
        color="red",
        alpha=0.2,
        label="+-1 Std Dev",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Torque vs Time with Mean and Standard Deviation")
    plt.legend()
    plt.grid()
    plt.show()

    return {
        "mean_torque": mean_torque,
        "std_torque": std_torque,
        "skewness_torque": skewness_torque,
        "kurtosis_torque": kurtosis_torque,
    }


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_group(
    ax_raw,
    ax_norm,
    title: str,
    results: Dict[str, Dict[str, Any]],
    key30: str,
    key50: str,
    force_max: float,
    intensities=(30, 50),
    line_color="steelblue",
    p30_color="blue",
    p50_color="red",
):
    """Plot raw and normalized force for one mode (concentric or eccentric)."""
    mean_vals = [results[key30]["mean"], results[key50]["mean"]]
    std_vals = [results[key30]["std"], results[key50]["std"]]
    vals30 = results[key30]["forces"]
    vals50 = results[key50]["forces"]

    norm_scale = 100.0 / force_max if np.isfinite(force_max) and force_max > 0 else np.nan

    for ax, ylabel, scale, suffix in [
        (ax_raw, "Force (Nm)", 1.0, "raw data"),
        (ax_norm, "Force (% MVC)", norm_scale, "normalized data"),
    ]:
        mv = [v * scale for v in mean_vals]
        sv = [v * scale for v in std_vals]
        v30 = [v * scale for v in vals30]
        v50 = [v * scale for v in vals50]

        marker = "o" if title.upper() == "CONCENTRIC" else "s"
        ax.plot(intensities, mv, marker=marker, markersize=10, linewidth=2.5, color=line_color, label="Mean force")
        ax.errorbar(intensities, mv, yerr=sv, fmt="none", ecolor=line_color, capsize=8, capthick=2, alpha=0.6)
        ax.scatter([30] * len(v30), v30, alpha=0.4, s=80, color=p30_color, label="Series")
        ax.scatter([50] * len(v50), v50, alpha=0.4, s=80, color=p50_color)
        ax.set_xlabel("Intensity", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.set_title(f"{title} - Force vs intensity\n({suffix})", fontsize=14, fontweight="bold")
        ax.set_xticks(list(intensities))
        ax.set_xticklabels(["30%", "50%"])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")


def plot_peak_series(
    ax,
    df_source: pd.DataFrame,
    events_source: pd.DataFrame,
    fs_source: float,
    label: str,
    threshold: float,
    direction: str,
    color: str,
    window: int = DEFAULT_WINDOW,
    onset_offset: float = 0.0,
    peak_min_interval: float = DEFAULT_PEAK_MIN_INTERVAL,
):
    """Plot signal and detected peaks for each set of one condition."""
    for serie_idx, (_, row) in enumerate(events_source.iterrows(), 1):
        t0 = row["onset"] + onset_offset
        seg_df = df_source[(df_source["time"] >= t0) & (df_source["time"] <= t0 + window)]
        seg_data = seg_df["torque_Nm"].reset_index(drop=True)
        seg_time = seg_df["time"].reset_index(drop=True)

        peak_values, peak_indices = detect_peaks_in_segment(
            seg_data,
            sampling_freq=fs_source,
            height_threshold=threshold,
            direction=direction,
            peak_min_interval=peak_min_interval,
        )

        ax[serie_idx - 1].plot(seg_time, seg_data, linewidth=1.5, color=color, label="Signal")

        if len(peak_values) > 0:
            peak_times = seg_time.iloc[peak_indices]
            mean_peak = float(np.mean(peak_values))
            ax[serie_idx - 1].scatter(
                peak_times,
                peak_values,
                color="red",
                s=100,
                zorder=5,
                marker="x",
                linewidths=3,
                label=f"Peaks ({len(peak_values)})",
            )
            ax[serie_idx - 1].axhline(
                mean_peak,
                color="darkgreen",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"Mean: {mean_peak:.3f} Nm",
            )

        ax[serie_idx - 1].set_xlabel("Time (s)", fontsize=11)
        ax[serie_idx - 1].set_ylabel("Force (Nm)", fontsize=11)
        ax[serie_idx - 1].set_title(
            f"{label} #{serie_idx} - {len(peak_values)} peaks (min spacing {peak_min_interval}s)",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
        ax[serie_idx - 1].legend(loc="upper right")
        ax[serie_idx - 1].grid(True, alpha=0.3)


# -----------------------------------------------------------------------------
# Subject-level analysis
# -----------------------------------------------------------------------------
def compare_subject_torque_modes(
    subject_id: str,
    base_root: str = DEFAULT_BASE_ROOT,
    concentric_session: Optional[str] = None,
    eccentric_session: Optional[str] = None,
    window: int = DEFAULT_WINDOW,
    peak_min_interval: float = DEFAULT_PEAK_MIN_INTERVAL,
    make_plot: bool = True,
):
    """Compare Con30/Con50 and Ecc30/Ecc50 for one subject."""
    subject_id = _normalize_subject_id(subject_id)

    if concentric_session is None:
        concentric_session = _find_session_for_task(subject_id, "concentric", base_root)
    if eccentric_session is None:
        eccentric_session = _find_session_for_task(subject_id, "eccentric", base_root)
    if concentric_session is None:
        raise FileNotFoundError(f"No concentric session found for subject {subject_id}.")
    if eccentric_session is None:
        raise FileNotFoundError(f"No eccentric session found for subject {subject_id}.")

    con_session = load_session_data(subject_id, concentric_session, task_choice="concentric", base_root=base_root)
    ecc_session = load_session_data(subject_id, eccentric_session, task_choice="eccentric", base_root=base_root)

    df_con, events_con, fs_con = con_session["df"], con_session["events"], con_session["fs"]
    df_ecc, events_ecc, fs_ecc = ecc_session["df"], ecc_session["events"], ecc_session["fs"]

    con30_events = events_con[events_con["trial_type"] == "Con30"]
    con50_events = events_con[events_con["trial_type"] == "Con50"]
    ecc30_events = events_ecc[events_ecc["trial_type"] == "Ecc30"]
    ecc50_events = events_ecc[events_ecc["trial_type"] == "Ecc50"]

    mvc_reference_con = compute_average_mvc(df_con, events_con)
    mvc_reference_ecc = compute_average_mvc(df_ecc, events_ecc)
    force_max_con = mvc_reference_con
    force_max_ecc = mvc_reference_ecc

    def _ratio_to_threshold(mvc_value: float, ratio: float, detection_factor: float = 0.7) -> float:
        # Use a slightly lower threshold than target intensity to keep peak detection robust.
        effective_ratio = ratio * detection_factor
        if np.isfinite(mvc_value) and mvc_value > 0:
            return float(mvc_value * effective_ratio)
        return float(effective_ratio)

    intensites = [30, 50]

    series_map = {
        "Con30": (df_con, con30_events, fs_con, _ratio_to_threshold(mvc_reference_con, 0.3), "positive", 0),
        "Con50": (df_con, con50_events, fs_con, _ratio_to_threshold(mvc_reference_con, 0.5), "positive", 0),
        "Ecc30": (df_ecc, ecc30_events, fs_ecc, _ratio_to_threshold(mvc_reference_ecc, 0.3), "absolute", 0),
        "Ecc50": (df_ecc, ecc50_events, fs_ecc, _ratio_to_threshold(mvc_reference_ecc, 0.5), "absolute", 0),
    }

    results: Dict[str, Dict[str, Any]] = {}
    for title, keys in [("CONCENTRIC", ["Con30", "Con50"]), ("ECCENTRIC", ["Ecc30", "Ecc50"])]:
        print("\n" + "=" * 70)
        print(f"PROCESSING {title} DATA")
        print("=" * 70)

        for key in keys:
            df_source, events_source, fs_source, threshold, direction, onset_offset = series_map[key]
            if len(events_source) == 0:
                print(f"No {key} data available")
                forces, all_peaks = [], []
            else:
                forces, all_peaks = process_series(
                    df_source,
                    events_source,
                    key,
                    threshold,
                    fs_source,
                    direction,
                    window=window,
                    onset_offset=onset_offset,
                    peak_min_interval=peak_min_interval,
                )
            mean_force, std_force = mean_std(forces)
            results[key] = {"forces": forces, "all_peaks": all_peaks, "mean": mean_force, "std": std_force}

    if make_plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        plot_group(ax1, ax2, "CONCENTRIC", results, "Con30", "Con50", force_max_con, intensites, "steelblue", "blue", "red")
        plot_group(ax3, ax4, "ECCENTRIC", results, "Ecc30", "Ecc50", force_max_ecc, intensites, "coral", "darkred", "orange")
        plt.tight_layout()
        plt.show()

    return {
        "con_session": con_session,
        "ecc_session": ecc_session,
        "df": df_con,
        "events_con": events_con,
        "fs": fs_con,
        "df_ecc": df_ecc,
        "events_ecc": events_ecc,
        "fs_ecc": fs_ecc,
        "con30_events": con30_events,
        "con50_events": con50_events,
        "ecc30_events": ecc30_events,
        "ecc50_events": ecc50_events,
        "average_mvc_max": mvc_reference_con,
        "average_mvc_max_con": mvc_reference_con,
        "average_mvc_max_ecc": mvc_reference_ecc,
        "force_max": force_max_con,
        "force_max_ecc": force_max_ecc,
        "intensites": intensites,
        "results": results,
    }


def unpack_comparison_legacy_variables(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Convert comparison output into notebook legacy variable names."""
    results = comparison["results"]
    return {
        "con_session": comparison["con_session"],
        "ecc_session": comparison["ecc_session"],
        "df": comparison["df"],
        "events_con": comparison["events_con"],
        "fs": comparison["fs"],
        "df_ecc": comparison["df_ecc"],
        "events_ecc": comparison["events_ecc"],
        "fs_ecc": comparison["fs_ecc"],
        "con30_events": comparison["con30_events"],
        "con50_events": comparison["con50_events"],
        "ecc30_events": comparison["ecc30_events"],
        "ecc50_events": comparison["ecc50_events"],
        "average_mvc_max": comparison["average_mvc_max"],
        "force_max": comparison["force_max"],
        "intensites": comparison["intensites"],
        "results": results,
        "forces_con30": results["Con30"]["forces"],
        "all_peaks_con30": results["Con30"]["all_peaks"],
        "forces_con50": results["Con50"]["forces"],
        "all_peaks_con50": results["Con50"]["all_peaks"],
        "forces_ecc30": results["Ecc30"]["forces"],
        "all_peaks_ecc30": results["Ecc30"]["all_peaks"],
        "forces_ecc50": results["Ecc50"]["forces"],
        "all_peaks_ecc50": results["Ecc50"]["all_peaks"],
        "mean_force_con30": results["Con30"]["mean"],
        "std_force_con30": results["Con30"]["std"],
        "mean_force_con50": results["Con50"]["mean"],
        "std_force_con50": results["Con50"]["std"],
        "mean_force_ecc30": results["Ecc30"]["mean"],
        "std_force_ecc30": results["Ecc30"]["std"],
        "mean_force_ecc50": results["Ecc50"]["mean"],
        "std_force_ecc50": results["Ecc50"]["std"],
    }


def plot_peak_detection_for_subject(
    subject_id: str,
    base_root: str = DEFAULT_BASE_ROOT,
    concentric_session: Optional[str] = None,
    eccentric_session: Optional[str] = None,
    window: int = DEFAULT_WINDOW,
    peak_min_interval: float = DEFAULT_PEAK_MIN_INTERVAL,
    figure_width: int = 15,
):
    """Plot peak detection for all Con/Ecc 30/50 sets of one chosen subject."""
    comparison = compare_subject_torque_modes(
        subject_id=subject_id,
        base_root=base_root,
        concentric_session=concentric_session,
        eccentric_session=eccentric_session,
        window=window,
        peak_min_interval=peak_min_interval,
        make_plot=False,
    )
    legacy = unpack_comparison_legacy_variables(comparison)

    con_mvc = comparison.get("average_mvc_max_con", comparison.get("average_mvc_max", np.nan))
    ecc_mvc = comparison.get("average_mvc_max_ecc", np.nan)

    def _ratio_to_threshold(mvc_value: float, ratio: float, detection_factor: float = 0.7) -> float:
        # Keep the same detection strategy as compare_subject_torque_modes.
        effective_ratio = ratio * detection_factor
        if np.isfinite(mvc_value) and mvc_value > 0:
            return float(mvc_value * effective_ratio)
        return float(effective_ratio)

    peak_series = {
        "CONCENTRIC Con30": (
            legacy["df"],
            legacy["con30_events"],
            legacy["fs"],
            _ratio_to_threshold(con_mvc, 0.3),
            "positive",
            0,
            "steelblue",
        ),
        "CONCENTRIC Con50": (
            legacy["df"],
            legacy["con50_events"],
            legacy["fs"],
            _ratio_to_threshold(con_mvc, 0.5),
            "positive",
            0,
            "darkblue",
        ),
        "ECCENTRIC Ecc30": (
            legacy["df_ecc"],
            legacy["ecc30_events"],
            legacy["fs_ecc"],
            _ratio_to_threshold(ecc_mvc, 0.3),
            "absolute",
            0,
            "coral",
        ),
        "ECCENTRIC Ecc50": (
            legacy["df_ecc"],
            legacy["ecc50_events"],
            legacy["fs_ecc"],
            _ratio_to_threshold(ecc_mvc, 0.5),
            "absolute",
            0,
            "darkorange",
        ),
    }

    total_plots = sum(len(ev) for _, ev, *_ in peak_series.values())
    if total_plots == 0:
        print(f"No events to visualize for subject {subject_id}.")
        return {"subject_id": subject_id, "comparison": comparison, "peak_series": peak_series, "total_plots": 0}

    fig, axes = plt.subplots(total_plots, 1, figsize=(figure_width, 4 * total_plots))
    if total_plots == 1:
        axes = [axes]

    offset = 0
    for label, (df_src, events_src, fs_src, threshold, direction, onset_offset, color) in peak_series.items():
        n = len(events_src)
        if n == 0:
            continue
        plot_peak_series(
            axes[offset:offset + n],
            df_src,
            events_src,
            fs_src,
            label,
            threshold,
            direction,
            color=color,
            window=window,
            onset_offset=onset_offset,
            peak_min_interval=peak_min_interval,
        )
        offset += n

    plt.tight_layout()
    plt.show()
    return {"subject_id": subject_id, "comparison": comparison, "peak_series": peak_series, "total_plots": total_plots}


# -----------------------------------------------------------------------------
# Group-level analysis and export
# -----------------------------------------------------------------------------
def compare_group_torque_modes(
    subject_ids: Optional[Iterable[str]] = None,
    base_root: str = DEFAULT_BASE_ROOT,
    concentric_session: Optional[str] = None,
    eccentric_session: Optional[str] = None,
    window: int = DEFAULT_WINDOW,
    peak_min_interval: float = DEFAULT_PEAK_MIN_INTERVAL,
    enforce_mvc_eligibility: bool = True,
    mvc_window: int = 6,
    mvc_tolerance_pct: float = 0.15,
    make_plot: bool = True,
    print_stats: bool = True,
):
    """Compute and optionally plot group means for Con30/Con50/Ecc30/Ecc50."""
    if subject_ids is None:
        subject_ids_clean = _list_subject_ids(base_root)
    else:
        subject_ids_clean = [_normalize_subject_id(s) for s in subject_ids]

    all_subject: Dict[str, Dict[str, Any]] = {}
    values_nm = {"Con30": [], "Con50": [], "Ecc30": [], "Ecc50": []}
    values_pct = {"Con30": [], "Con50": [], "Ecc30": [], "Ecc50": []}

    for subject_id in subject_ids_clean:
        try:
            comparison = compare_subject_torque_modes(
                subject_id=subject_id,
                base_root=base_root,
                concentric_session=concentric_session,
                eccentric_session=eccentric_session,
                window=window,
                peak_min_interval=peak_min_interval,
                make_plot=False,
            )

            if enforce_mvc_eligibility:
                con_ses = comparison["con_session"]["session_id"]
                ecc_ses = comparison["ecc_session"]["session_id"]

                con_qc = evaluate_subject_mvc_eligibility(
                    subject_id=subject_id,
                    session_id=con_ses,
                    base_root=base_root,
                    task_choice="concentric",
                    mvc_window=mvc_window,
                    tolerance_pct=mvc_tolerance_pct,
                )
                ecc_qc = evaluate_subject_mvc_eligibility(
                    subject_id=subject_id,
                    session_id=ecc_ses,
                    base_root=base_root,
                    task_choice="eccentric",
                    mvc_window=mvc_window,
                    tolerance_pct=mvc_tolerance_pct,
                )

                if not (con_qc["is_subject_eligible"] and ecc_qc["is_subject_eligible"]):
                    if print_stats:
                        print(
                            f"Skipping subject {subject_id}: MVC QC failed "
                            f"(Con={con_qc['is_subject_eligible']}, Ecc={ecc_qc['is_subject_eligible']})."
                        )
                    continue

            all_subject[subject_id] = comparison

            mvc_ref_con = comparison.get("force_max", np.nan)
            mvc_ref_ecc = comparison.get("force_max_ecc", np.nan)
            for key in ["Con30", "Con50", "Ecc30", "Ecc50"]:
                mean_val = comparison["results"][key]["mean"]
                if np.isfinite(mean_val):
                    values_nm[key].append(float(mean_val))
                    mvc_ref = mvc_ref_ecc if key.startswith("Ecc") else mvc_ref_con
                    if np.isfinite(mvc_ref) and mvc_ref > 0:
                        values_pct[key].append(float(mean_val / mvc_ref * 100.0))
        except Exception as e:
            if print_stats:
                print(f"Skipping subject {subject_id}: {e}")

    if not all_subject:
        raise RuntimeError("No subject could be processed for group comparison.")

    def _mean_std_n(arr: Iterable[float]) -> Tuple[float, float, int]:
        vals = np.asarray(list(arr), dtype=float)
        if vals.size == 0:
            return np.nan, np.nan, 0
        return float(np.mean(vals)), float(np.std(vals)), int(vals.size)

    summary: Dict[str, Dict[str, float]] = {}
    for key in ["Con30", "Con50", "Ecc30", "Ecc50"]:
        mean_nm, std_nm, n_nm = _mean_std_n(values_nm[key])
        mean_pct, std_pct, n_pct = _mean_std_n(values_pct[key])
        summary[key] = {
            "mean_nm": mean_nm,
            "std_nm": std_nm,
            "n_nm": n_nm,
            "mean_pct": mean_pct,
            "std_pct": std_pct,
            "n_pct": n_pct,
        }

    if make_plot:
        x = [30, 50]
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        def _plot_points(ax, x_center, values, color):
            if not values:
                return
            jitter = np.linspace(-0.8, 0.8, num=len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(values), x_center, dtype=float) + jitter,
                values,
                color=color,
                alpha=0.6,
                s=75,
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
            )

        ax1.errorbar(x, [summary["Con30"]["mean_nm"], summary["Con50"]["mean_nm"]], yerr=[summary["Con30"]["std_nm"], summary["Con50"]["std_nm"]], fmt="-o", color="steelblue", markersize=9, linewidth=2.5, capsize=7)
        _plot_points(ax1, 30, values_nm["Con30"], "steelblue")
        _plot_points(ax1, 50, values_nm["Con50"], "steelblue")
        ax1.set_title("CONCENTRIC - Group mean (raw)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Force (Nm)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["30%", "50%"])
        ax1.grid(True, alpha=0.3, linestyle="--")

        ax2.errorbar(x, [summary["Con30"]["mean_pct"], summary["Con50"]["mean_pct"]], yerr=[summary["Con30"]["std_pct"], summary["Con50"]["std_pct"]], fmt="-o", color="royalblue", markersize=9, linewidth=2.5, capsize=7)
        _plot_points(ax2, 30, values_pct["Con30"], "royalblue")
        _plot_points(ax2, 50, values_pct["Con50"], "royalblue")
        ax2.set_title("CONCENTRIC - Group mean (normalized)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Force (% MVC)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["30%", "50%"])
        ax2.grid(True, alpha=0.3, linestyle="--")

        ax3.errorbar(x, [summary["Ecc30"]["mean_nm"], summary["Ecc50"]["mean_nm"]], yerr=[summary["Ecc30"]["std_nm"], summary["Ecc50"]["std_nm"]], fmt="-s", color="coral", markersize=9, linewidth=2.5, capsize=7)
        _plot_points(ax3, 30, values_nm["Ecc30"], "coral")
        _plot_points(ax3, 50, values_nm["Ecc50"], "coral")
        ax3.set_title("ECCENTRIC - Group mean (raw)", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Intensity")
        ax3.set_ylabel("Force (Nm)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(["30%", "50%"])
        ax3.grid(True, alpha=0.3, linestyle="--")

        ax4.errorbar(x, [summary["Ecc30"]["mean_pct"], summary["Ecc50"]["mean_pct"]], yerr=[summary["Ecc30"]["std_pct"], summary["Ecc50"]["std_pct"]], fmt="-s", color="darkorange", markersize=9, linewidth=2.5, capsize=7)
        _plot_points(ax4, 30, values_pct["Ecc30"], "darkorange")
        _plot_points(ax4, 50, values_pct["Ecc50"], "darkorange")
        ax4.set_title("ECCENTRIC - Group mean (normalized)", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Intensity")
        ax4.set_ylabel("Force (% MVC)")
        ax4.set_xticks(x)
        ax4.set_xticklabels(["30%", "50%"])
        ax4.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.show()

    if print_stats:
        print(f"Subjects used: {sorted(all_subject.keys())}")
        print("\n" + "=" * 70)
        print("GROUP TORQUE SUMMARY")
        print("=" * 70)
        for key in ["Con30", "Con50", "Ecc30", "Ecc50"]:
            item = summary[key]
            print(
                f"{key}: {item['mean_nm']:.3f} +/- {item['std_nm']:.3f} Nm (n={item['n_nm']}), "
                f"{item['mean_pct']:.2f} +/- {item['std_pct']:.2f} %MVC (n={item['n_pct']})"
            )
        print("=" * 70)

    return {"subject_ids_used": sorted(all_subject.keys()), "all_subject": all_subject, "summary": summary}


def compute_mode_forces(
    df_source: pd.DataFrame,
    events_source: pd.DataFrame,
    fs_source: float,
    mode_name: str,
    height_threshold: float,
    task_type: str,
    window_contraction: int = DEFAULT_WINDOW,
    peak_min_interval: float = DEFAULT_PEAK_MIN_INTERVAL,
):
    """Compute mean force per set for one mode (Con30/Con50/Ecc30/Ecc50)."""
    direction = "positive" if task_type == "concentric" else "absolute"
    mode_events = events_source[events_source["trial_type"] == mode_name]
    forces_mode = []

    for _, row in mode_events.iterrows():
        start = row["onset"]
        end = start + window_contraction
        segment = df_source[(df_source["time"] >= start) & (df_source["time"] <= end)]["torque_Nm"]
        peaks, _ = detect_peaks_in_segment(
            segment,
            sampling_freq=fs_source,
            height_threshold=height_threshold,
            direction=direction,
            peak_min_interval=peak_min_interval,
        )
        if len(peaks) > 0:
            forces_mode.append(float(np.mean(peaks)))

    return forces_mode


def export_torque_summary_csv(
    base_dir: str,
    output_file: str,
    task_modes: Optional[Dict[str, Dict[str, float]]] = None,
    window_contraction: int = DEFAULT_WINDOW,
    peak_min_interval: float = DEFAULT_PEAK_MIN_INTERVAL,
    enforce_mvc_eligibility: bool = True,
    mvc_window: int = 6,
    mvc_tolerance_pct: float = 0.15,
    print_progress: bool = True,
) -> pd.DataFrame:
    """Run torque analysis over all subjects/sessions and export CSV summary."""
    if task_modes is None:
        task_modes = {
            "concentric": {"Con30": 0.3, "Con50": 0.5},
            "eccentric": {"Ecc30": 0.3, "Ecc50": 0.5},
        }

    rows = []
    subjects = _list_subject_ids(base_dir)

    for subject_id in subjects:
        sub_folder = os.path.join(base_dir, f"sub-{subject_id}")
        sessions = sorted(d for d in os.listdir(sub_folder) if d.startswith("ses-"))

        for ses_id in sessions:
            ses_clean = ses_id.replace("ses-", "")
            for task_type, modes in task_modes.items():
                paths = build_beh_file_paths(subject_id, ses_clean, task_type, base_root=base_dir)
                if not all(os.path.exists(p) for p in paths.values()):
                    continue

                if print_progress:
                    print(f"\nProcessing sub-{subject_id} - {ses_id} - {task_type}...")

                try:
                    if enforce_mvc_eligibility:
                        qc = evaluate_subject_mvc_eligibility(
                            subject_id=subject_id,
                            session_id=ses_clean,
                            base_root=base_dir,
                            task_choice=task_type,
                            mvc_window=mvc_window,
                            tolerance_pct=mvc_tolerance_pct,
                        )
                        if not qc["is_subject_eligible"]:
                            if print_progress:
                                print(
                                    f"  Skipped by MVC QC: last={qc['last_mvc']:.2f} Nm, "
                                    f"range=[{qc['lower_bound']:.2f}, {qc['upper_bound']:.2f}] Nm"
                                )
                            continue

                    df_source, events_source, fs_source = load_task_data(paths["torque"], paths["events"], paths["json"])
                    mvc_reference = compute_average_mvc(df_source, events_source)

                    for mode_name, height_threshold in modes.items():
                        threshold_value = float(height_threshold)
                        if np.isfinite(mvc_reference) and mvc_reference > 0 and threshold_value <= 1.0:
                            threshold_value = float(mvc_reference * threshold_value)

                        forces_mode = compute_mode_forces(
                            df_source,
                            events_source,
                            fs_source,
                            mode_name,
                            threshold_value,
                            task_type,
                            window_contraction=window_contraction,
                            peak_min_interval=peak_min_interval,
                        )
                        if not forces_mode:
                            continue

                        mean_force, std_force = mean_std(forces_mode)
                        if np.isfinite(mvc_reference) and mvc_reference > 0:
                            mean_force_normalized = (mean_force / mvc_reference) * 100
                            std_force_normalized = (std_force / mvc_reference) * 100
                        else:
                            mean_force_normalized = np.nan
                            std_force_normalized = np.nan

                        rows.append(
                            {
                                "Subject": subject_id,
                                "Session": ses_clean,
                                "Task": task_type,
                                "Intensity": int(mode_name[-2:]),
                                "Mean_Force_Nm": mean_force,
                                "Std_Force_Nm": std_force,
                                "Mean_Force_Percent_MVC": mean_force_normalized,
                                "Std_Force_Percent_MVC": std_force_normalized,
                                "MVC_Nm": mvc_reference,
                                "Number_of_Series": len(forces_mode),
                            }
                        )

                        if print_progress:
                            print(
                                f"  {mode_name}: Mean={mean_force:.2f} Nm "
                                f"({mean_force_normalized:.1f}% MVC), n={len(forces_mode)}"
                            )
                except Exception as e:
                    if print_progress:
                        print(f"  Error processing sub-{subject_id} - {ses_id} - {task_type}: {e}")

    df_results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_results.to_csv(output_file, index=False, sep=";", decimal=",")

    if print_progress:
        print(f"\n{'=' * 70}")
        print("Analysis complete")
        print(f"Results saved to: {output_file}")
        print(f"Total rows: {len(df_results)}")
        print(f"{'=' * 70}")

    return df_results

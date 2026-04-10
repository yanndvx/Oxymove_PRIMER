import os

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

"""# NIRS Oxymove Utilities

Fonctions classees par usage pour faciliter la lecture et la maintenance.

## 1) Helpers internes (infrastructure)
- _empty_device_stats
- _find_first_session_for_task
- _style_epoch_axis
- _safe_legend
- _extract_task_from_events_filename
- _iter_subject_session_task_events
- _default_output_dir_from_base_root
- _collect_metric_values

## 2) Chargement et pre-traitement des signaux
- load_filtered_hb
- get_hbr_channels
- epoch_signal

## 3) Extraction de metriques HbR
- extract_hbr_means
- compute_condition_hbr_average
- mean_std
- extract_temporal_features

## 4) Visualisation
- plot_device
- print_block
- plot_temporal_comparison
- _plot_epochs_one_by_one
- _plot_intensity_per_row
- plot_envelope_and_auc_per_set
- display_sets_one_by_one

## 5) Logique de mapping task/session
- infer_task_from_session
- build_events_file

## 6) Analyses comparatives et exports
- process_device
- compare_devices_by_task
- compare_devices_all_subjects
- build_hbr_summary_table
- export_hbr_summary_csv
- compare_devices_group_mean
- print_average_hbr_mvc
"""


DEFAULT_BASE_ROOT = r'C:\Program Files\DigiMove\DigiMove\DataAnalysisProject\data\MOXY-bids'
DEFAULT_WINDOW_DURATION = 30
DEVICE_COLORS = {'semaxone': '#D63DDB', 'portalite': '#D8C730'}
CHROMOPHORE_COLORS = {
    'HbR': 'red',
    'HbO': 'blue',
    'HbDiff': 'yellow',
    'HbTot': 'green',
}


# -----------------------------------------------------------------------------
# 1) Helpers internes (infrastructure)
# -----------------------------------------------------------------------------
def _empty_device_stats():
    """Return a standardized empty stats dict for unavailable device/task."""
    return {
        'available': False,
        'means30': [],
        'means50': [],
        'mean30': np.nan,
        'std30': np.nan,
        'mean50': np.nan,
        'std50': np.nan,
    }


def _find_first_session_for_task(base_root, subject_id, task_name):
    """Find first session containing events for a given subject/task."""
    sub_dir = os.path.join(base_root, f'sub-{subject_id}')
    if not os.path.isdir(sub_dir):
        return None

    for ses_name in sorted(os.listdir(sub_dir)):
        if not ses_name.startswith('ses-'):
            continue
        ses_id = ses_name.replace('ses-', '')
        events_path = os.path.join(
            sub_dir,
            ses_name,
            'nirs',
            f'sub-{subject_id}_ses-{ses_id}_task-{task_name}_events.tsv',
        )
        if os.path.exists(events_path):
            return ses_id, events_path
    return None


def _style_epoch_axis(ax, title=None, xlabel='Temps (s)', ylabel='ΔHbR (uM)'):
    """Apply consistent styling for time-series epoch plots."""
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)


def _safe_legend(ax):
    """Draw legend only when plot handles are available."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def _integrate_auc(values, times):
    """Integrate a curve with NumPy's trapezoidal rule, with version fallback."""
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(values, times))
    return float(np.trapz(values, times))


# -----------------------------------------------------------------------------
# 2) Chargement et pre-traitement des signaux
# -----------------------------------------------------------------------------
def load_filtered_hb(subject, session, task, recording, base_root=None):
    """Load SNIRF from BIDS, convert to Hb, then apply band-pass filtering."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT

    raw_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        recording=recording,
        suffix='nirs',
        extension='.snirf',
        datatype='nirs',
        root=base_root,
    )
    raw_in = read_raw_bids(raw_path)
    raw_od = mne.preprocessing.nirs.optical_density(raw_in)
    raw_hb = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=4)
    return raw_hb.copy().filter(
        l_freq=0.01,
        h_freq=4,
        method='iir', 
        iir_params=dict(order=4, ftype='butter'),
    )


def get_hbr_channels(raw_obj):
    """Return channel names corresponding to HbR signal."""
    return [ch for ch in raw_obj.ch_names if 'hbr' in ch.lower()]


def get_hbo_channels(raw_obj):
    """Return channel names corresponding to HbO signal."""
    return [ch for ch in raw_obj.ch_names if 'hbo' in ch.lower()]


def _get_chromophore_channels(raw_obj, chromophore='HbR'):
    """Return channel names for a given chromophore label."""
    chromophore_key = str(chromophore).strip().lower()
    if chromophore_key == 'hbr':
        return get_hbr_channels(raw_obj)
    if chromophore_key == 'hbo':
        return get_hbo_channels(raw_obj)
    if chromophore_key in {'hbtot', 'hbdiff'}:
        return []
    raise ValueError("chromophore must be 'HbR', 'HbO', 'HbTot', or 'HbDiff'.")


# -----------------------------------------------------------------------------
# 3) Extraction de metriques HbR
# -----------------------------------------------------------------------------
def extract_hbr_means(
    raw_obj,
    events_df,
    trial_type,
    hbr_channels,
    label,
    window_duration=None,
    verbose=True,
):
    """Compute mean HbR (uM) per event for one trial type."""
    if window_duration is None:
        window_duration = DEFAULT_WINDOW_DURATION

    hbr_idx = [raw_obj.ch_names.index(ch) for ch in hbr_channels if ch in raw_obj.ch_names]
    if not hbr_idx:
        return []

    means = []
    trial_events = events_df.loc[events_df['trial_type'] == trial_type, 'onset']
    if verbose:
        print(f"\nProcessing {trial_type} series ({label})...")

    for onset in trial_events:
        segment = raw_obj.copy().crop(tmin=onset, tmax=onset + window_duration)
        data_array = segment.get_data()
        mean_hbr = np.mean(data_array[hbr_idx, :]) * 1e6
        means.append(mean_hbr)
        if verbose:
            print(f"  Series {len(means)}: Mean HbR = {mean_hbr:.3f} µM")

    return means


def compute_condition_hbr_average(raw_obj, events_df, trial_type, window_duration=None):
    """Compute mean HbR concentration for one condition.

    Formula:
    1) For each series/event, compute mean across HbR channels and samples in the effort window.
    2) Condition average = arithmetic mean of all series means.
    """
    hbr_channels = get_hbr_channels(raw_obj)
    series_means = extract_hbr_means(
        raw_obj,
        events_df,
        trial_type,
        hbr_channels,
        label='formula-check',
        window_duration=window_duration,
    )
    if len(series_means) == 0:
        return np.nan, []
    return float(np.mean(series_means)), series_means


def mean_std(values):
    """Return mean/std as floats; NaN/NaN for empty input."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr))


# -----------------------------------------------------------------------------
# 4) Visualisation
# -----------------------------------------------------------------------------
def plot_device(ax, x, means_pair, std_pair, series_pair, color_line, color_30, color_50, marker, label):
    """Plot 30/50% means, error bars, and individual series for one device."""
    if np.isnan(means_pair[0]) or np.isnan(means_pair[1]):
        return
    ax.plot(x, means_pair, marker=marker, markersize=10, linewidth=2.5, color=color_line, label=label)
    ax.errorbar(x, means_pair, yerr=std_pair, fmt='none', ecolor=color_line, capsize=8, capthick=2, alpha=0.6)
    ax.scatter([30] * len(series_pair[0]), series_pair[0], alpha=0.3, s=60, color=color_30)
    ax.scatter([50] * len(series_pair[1]), series_pair[1], alpha=0.3, s=60, color=color_50)


def print_block(title, m30, s30, n30, m50, s50, n50):
    """Print formatted stats block for one condition/device pair."""
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"30%: {m30:.3f} ± {s30:.3f} µM ({n30} series)")
    print(f"50%: {m50:.3f} ± {s50:.3f} µM ({n50} series)")
    print(f"Δ HbR (50%-30%): {m50 - m30:.3f} µM")


# -----------------------------------------------------------------------------
# 6) Analyses comparatives et exports
# -----------------------------------------------------------------------------
def process_device(
    subject,
    session,
    task,
    recording,
    events_df,
    trial30,
    trial50,
    title,
    raw_obj=None,
    channels=None,
    base_root=None,
):
    """Process one device and return 30/50% HbR means/std with availability flag."""
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)

    try:
        raw_proc = (
            raw_obj
            if raw_obj is not None
            else load_filtered_hb(subject, session, task, recording, base_root=base_root)
        )
        hbr_ch = channels if channels is not None else get_hbr_channels(raw_proc)
        means_30 = extract_hbr_means(raw_proc, events_df, trial30, hbr_ch, recording)
        means_50 = extract_hbr_means(raw_proc, events_df, trial50, hbr_ch, recording)
        mean_30, std_30 = mean_std(means_30)
        mean_50, std_50 = mean_std(means_50)
        return {
            'available': True,
            'means30': means_30,
            'means50': means_50,
            'mean30': mean_30,
            'std30': std_30,
            'mean50': mean_50,
            'std50': std_50,
        }
    except Exception as e:
        print(f"  ⚠ {title} data not available: {str(e)}")
        return _empty_device_stats()
    


def epoch_signal(raw_hb_filt, trial_type, t_pre=5, t_post=25, events_df=None, chromophore='HbR'):
    """Build baseline-corrected epochs around trial onsets for HbR, HbO, HbTot, or HbDiff."""
    sfreq = raw_hb_filt.info['sfreq']
    chromophore_key = str(chromophore).strip().lower()
    if chromophore_key in {'hbtot', 'hbdiff'}:
        hbo_chs = get_hbo_channels(raw_hb_filt)
        hbr_chs = get_hbr_channels(raw_hb_filt)
        if not hbo_chs or not hbr_chs:
            return np.array([]), np.array([])
        hbo_data_full = raw_hb_filt.get_data(picks=hbo_chs) * 1e6
        hbr_data_full = raw_hb_filt.get_data(picks=hbr_chs) * 1e6
        n_samples = min(hbo_data_full.shape[1], hbr_data_full.shape[1])
    else:
        signal_chs = _get_chromophore_channels(raw_hb_filt, chromophore=chromophore)
        if not signal_chs:
            return np.array([]), np.array([])
        data_full = raw_hb_filt.get_data(picks=signal_chs) * 1e6
        n_samples = data_full.shape[1]
    n_pre = int(t_pre * sfreq)
    n_post    = int(t_post * sfreq)
    epochs    = []

    if events_df is not None:
        trial_events = events_df[events_df['trial_type'] == trial_type]
        onset_samples = [int(row['onset'] * sfreq) for _, row in trial_events.iterrows()]
    else:
        events, event_id = mne.events_from_annotations(raw_hb_filt, verbose=False)
        if trial_type not in event_id:
            return np.array([]), np.array([])
        target_id = event_id[trial_type]
        trial_events = events[events[:, 2] == target_id]
        onset_samples = [event[0] - raw_hb_filt.first_samp for event in trial_events]

    for onset_sample in onset_samples:
        start = onset_sample - n_pre
        stop = onset_sample + n_post
        if start < 0 or stop > n_samples:
            continue

        if chromophore_key == 'hbtot':
            segment = np.mean(hbo_data_full[:, start:stop], axis=0) + np.mean(hbr_data_full[:, start:stop], axis=0)
        elif chromophore_key == 'hbdiff':
            segment = np.mean(hbo_data_full[:, start:stop], axis=0) - np.mean(hbr_data_full[:, start:stop], axis=0)
        else:
            segment = np.mean(data_full[:, start:stop], axis=0)

        n_baseline = int(2 * sfreq)
        baseline = np.median(segment[n_pre - n_baseline:n_pre])
        segment -= baseline

        epochs.append(segment)

    if not epochs:
        return np.array([]), np.array([])

    min_len = min(e.shape[0] for e in epochs)
    epochs  = np.array([e[:min_len] for e in epochs])
    times   = np.linspace(-t_pre, t_post, min_len)
    return epochs, times


def plot_hb_envelope_and_auc_per_set(
    subject_choice='001',
    session_choice='002',
    recording_choice='semaxone',
    task_choice=None,
    t_pre=5,
    t_post=40,
    auc_window=(0, None),
    envelope_mode='minmax',
    use_abs_auc=False,
    chromophores=('HbR', 'HbO'),
    plot_mode='by_device',
    base_root=None,
):
    """Plot mean envelope and AUC for HbR, HbO, HbTot, and HbDiff, by trial type and device."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT

    if isinstance(recording_choice, str):
        recordings = [recording_choice]
    else:
        recordings = list(recording_choice)

    if not recordings:
        raise ValueError('recording_choice must contain at least one device.')

    if envelope_mode not in {'minmax', 'std'}:
        raise ValueError("envelope_mode must be 'minmax' or 'std'.")

    if plot_mode not in {'by_device', 'combined'}:
        raise ValueError("plot_mode must be 'by_device' or 'combined'.")

    if not isinstance(auc_window, (tuple, list)) or len(auc_window) != 2:
        raise ValueError('auc_window must be a tuple/list: (t_start, t_end).')
    auc_tmin, auc_tmax = auc_window

    if isinstance(chromophores, str):
        chromophore_list = [chromophores]
    else:
        chromophore_list = list(chromophores)

    if not chromophore_list:
        raise ValueError('chromophores must contain at least one value.')

    normalized_chromophores = []
    for chromophore in chromophore_list:
        chromophore_key = str(chromophore).strip().lower()
        if chromophore_key not in {'hbr', 'hbo', 'hbtot', 'hbdiff'}:
            raise ValueError("chromophores must contain only 'HbR', 'HbO', 'HbTot', and/or 'HbDiff'.")
        if chromophore_key == 'hbr':
            normalized_chromophores.append('HbR')
        elif chromophore_key == 'hbo':
            normalized_chromophores.append('HbO')
        elif chromophore_key == 'hbtot':
            normalized_chromophores.append('HbTot')
        else:
            normalized_chromophores.append('HbDiff')

    task_used = infer_task_from_session(
        session_choice,
        task_choice=task_choice,
        subject_choice=subject_choice,
        base_root=base_root,
    )
    events_file = build_events_file(base_root, subject_choice, session_choice, task_used)
    if not os.path.exists(events_file):
        raise FileNotFoundError(f'Events file not found: {events_file}')

    events_df_selected = pd.read_csv(events_file, sep='\t')
    print(f'Using events file: {events_file}')

    raw_by_device = {}
    for dev in recordings:
        try:
            raw_by_device[dev] = load_filtered_hb(
                subject_choice,
                session_choice,
                task_used,
                dev,
                base_root=base_root,
            )
        except Exception as e:
            print(f"Warning: unable to load {dev}: {e}")

    if not raw_by_device:
        raise RuntimeError('No device could be loaded.')

    trial_types = [
        t for t in sorted(events_df_selected['trial_type'].dropna().unique())
        if _is_displayable_trial_type(t)
    ]

    if not trial_types:
        print('No displayable trial_type found.')
        return {}

    summary = {}

    if plot_mode == 'combined':
        fig, axes = plt.subplots(
            len(trial_types),
            len(normalized_chromophores),
            figsize=(5.2 * len(normalized_chromophores), 3.8 * len(trial_types)),
            squeeze=False,
        )

        for row_idx, trial_type in enumerate(trial_types):
            summary[trial_type] = {}

            for col_idx, chromophore in enumerate(normalized_chromophores):
                ax = axes[row_idx, col_idx]
                summary[trial_type][chromophore] = {}
                has_data = False

                for dev, raw_selected in raw_by_device.items():
                    epochs, times = epoch_signal(
                        raw_selected,
                        trial_type,
                        t_pre=t_pre,
                        t_post=t_post,
                        events_df=events_df_selected,
                        chromophore=chromophore,
                    )
                    if len(epochs) == 0:
                        continue

                    has_data = True
                    mean_curve = np.mean(epochs, axis=0)

                    if envelope_mode == 'std':
                        std_curve = np.std(epochs, axis=0)
                        lower = mean_curve - std_curve
                        upper = mean_curve + std_curve
                    else:
                        lower = np.min(epochs, axis=0)
                        upper = np.max(epochs, axis=0)

                    auc_mask = times >= auc_tmin
                    if auc_tmax is not None:
                        auc_mask = auc_mask & (times <= auc_tmax)

                    if np.any(auc_mask):
                        auc_signal = np.abs(mean_curve[auc_mask]) if use_abs_auc else mean_curve[auc_mask]
                        auc_value = _integrate_auc(auc_signal, times[auc_mask])
                    else:
                        auc_value = np.nan

                    color = DEVICE_COLORS.get(dev, None)
                    ax.fill_between(times, lower, upper, color=color, alpha=0.18)
                    ax.plot(times, mean_curve, color=color, linewidth=2.0, label=f"{dev} (AUC={auc_value:.2f} uM*s)")

                    summary[trial_type][chromophore][dev] = {
                        'auc_uM_s': auc_value,
                        'n_sets': int(epochs.shape[0]),
                    }

                _style_epoch_axis(
                    ax,
                    title=f"sub-{subject_choice} ses-{session_choice} | {trial_type} | {chromophore}",
                    ylabel=f'Δ{chromophore} (uM)',
                )

                if has_data:
                    _safe_legend(ax)
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', alpha=0.6)

        fig.suptitle('Envelope and area under the curve for Hb signals', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        for dev, raw_selected in raw_by_device.items():
            fig, axes = plt.subplots(
                len(trial_types),
                1,
                figsize=(9, 3.8 * len(trial_types)),
                squeeze=False,
            )

            for row_idx, trial_type in enumerate(trial_types):
                ax = axes[row_idx, 0]
                has_data = False
                if trial_type not in summary:
                    summary[trial_type] = {}

                for chromophore in normalized_chromophores:
                    if chromophore not in summary[trial_type]:
                        summary[trial_type][chromophore] = {}

                    epochs, times = epoch_signal(
                        raw_selected,
                        trial_type,
                        t_pre=t_pre,
                        t_post=t_post,
                        events_df=events_df_selected,
                        chromophore=chromophore,
                    )
                    if len(epochs) == 0:
                        continue

                    has_data = True
                    mean_curve = np.mean(epochs, axis=0)

                    if envelope_mode == 'std':
                        std_curve = np.std(epochs, axis=0)
                        lower = mean_curve - std_curve
                        upper = mean_curve + std_curve
                    else:
                        lower = np.min(epochs, axis=0)
                        upper = np.max(epochs, axis=0)

                    auc_mask = times >= auc_tmin
                    if auc_tmax is not None:
                        auc_mask = auc_mask & (times <= auc_tmax)

                    if np.any(auc_mask):
                        auc_signal = np.abs(mean_curve[auc_mask]) if use_abs_auc else mean_curve[auc_mask]
                        auc_value = _integrate_auc(auc_signal, times[auc_mask])
                    else:
                        auc_value = np.nan

                    color = CHROMOPHORE_COLORS.get(chromophore, 'black')
                    ax.fill_between(times, lower, upper, color=color, alpha=0.18)
                    ax.plot(
                        times,
                        mean_curve,
                        color=color,
                        linewidth=2.0,
                        label=f"{chromophore} (AUC={auc_value:.2f} uM*s)",
                    )

                    summary[trial_type][chromophore][dev] = {
                        'auc_uM_s': auc_value,
                        'n_sets': int(epochs.shape[0]),
                    }

                _style_epoch_axis(
                    ax,
                    title=f"sub-{subject_choice} ses-{session_choice} | {dev} | {trial_type}",
                    ylabel='ΔHb (uM)',
                )

                if has_data:
                    _safe_legend(ax)
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', alpha=0.6)

            fig.suptitle(f'Envelope and area under the curve for {dev}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()

    return summary

def plot_temporal_comparison(results):
    """Plot each temporal series separately for mode/intensity/device comparisons."""
    modes = [('Con', 'Concentrique'), ('Ecc', 'Excentrique')]

    for mode, mode_label in modes:
        for intensity in ['30', '50']:
            key = f'{mode}{intensity}'

            # Vérifie qu'au moins un appareil est disponible
            data = {
                dev: results[f'{key}_{dev}']
                for dev in ['semaxone', 'portalite']
                if f'{key}_{dev}' in results
            }
            if not data:
                continue

            # Nombre de séries = max entre les appareils
            n = max(len(d['epochs']) for d in data.values())

            # Affiche chaque série dans une figure séparée (une par une)
            for i in range(n):
                fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
                for dev, d in data.items():
                    if i >= len(d['epochs']):
                        continue
                    ax.plot(d['times'], d['epochs'][i],
                            color=DEVICE_COLORS[dev], linewidth=1.5, label=dev)

                _style_epoch_axis(
                    ax,
                    title=f'{mode_label} — {intensity}% — Série {i + 1}',
                    ylabel='ΔHbR (µM)',
                )
                _safe_legend(ax)

                plt.tight_layout()
                plt.show()

def extract_temporal_features(epochs, times, t_effort_end=30):
    """Extract onset delay, peak, and half-recovery metrics from each epoch."""
    rows = []
    for epoch in epochs:
        baseline_std = epoch[times < 0].std()
        threshold    = 2 * baseline_std

        mask_post   = times >= 0
        above       = mask_post & (np.abs(epoch) > threshold)
        onset_delay = times[np.argmax(above)] if above.any() else np.nan

        mask_effort = (times >= 0) & (times <= t_effort_end)
        sub         = np.abs(epoch) * mask_effort
        peak_idx    = np.argmax(sub)
        peak_amp    = epoch[peak_idx]
        peak_time   = times[peak_idx]

        mask_rec = times > t_effort_end
        t_half   = np.nan
        if mask_rec.any() and not np.isnan(peak_amp):
            rec_sig   = epoch[mask_rec]
            rec_times = times[mask_rec]
            crossed   = np.where(np.abs(rec_sig) <= np.abs(peak_amp) / 2)[0]
            if len(crossed):
                t_half = rec_times[crossed[0]] - t_effort_end

        rows.append({
            'onset_delay_s':     onset_delay,
            'peak_amplitude_uM': peak_amp,
            'peak_time_s':       peak_time,
            't_half_recovery_s': t_half,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 5) Logique de mapping task/session
# -----------------------------------------------------------------------------
def infer_task_from_session(
    session_choice,
    task_choice=None,
    subject_choice=None,
    base_root=None,
):
    """Infer task from explicit input, events files, or default session mapping."""
    if task_choice is not None:
        return task_choice

    # Prefer explicit discovery from existing events files for this subject/session.
    if subject_choice is not None and base_root is not None:
        nirs_dir = os.path.join(
            base_root,
            f'sub-{subject_choice}',
            f'ses-{session_choice}',
            'nirs',
        )
        if os.path.isdir(nirs_dir):
            prefix = f'sub-{subject_choice}_ses-{session_choice}_task-'
            suffix = '_events.tsv'
            tasks = []
            for fname in os.listdir(nirs_dir):
                if fname.startswith(prefix) and fname.endswith(suffix):
                    tasks.append(fname[len(prefix):-len(suffix)])

            unique_tasks = sorted(set(tasks))
            if len(unique_tasks) == 1:
                return unique_tasks[0]
            if len(unique_tasks) > 1:
                raise ValueError(
                    'Multiple tasks found for this subject/session: '
                    f"{unique_tasks}. Set task_choice explicitly."
                )

    session_task_map = {'001': 'concentric', '002': 'eccentric'}
    task_used = session_task_map.get(session_choice)
    if task_used is None:
        raise ValueError(
            f"Cannot infer task from session '{session_choice}'. Set task_choice explicitly."
        )
    return task_used


def build_events_file(base_root, subject_choice, session_choice, task_used):
    """Build path to NIRS events TSV for selected subject/session/task."""
    return os.path.join(
        base_root,
        f'sub-{subject_choice}',
        f'ses-{session_choice}',
        'nirs',
        f'sub-{subject_choice}_ses-{session_choice}_task-{task_used}_events.tsv',
    )


def _plot_epochs_one_by_one(
    epochs_by_device,
    times_by_device,
    subject_choice,
    session_choice,
    trial_type,
):
    """Plot one figure per series overlaying all available devices."""
    max_series = max(len(epochs_by_device[dev]) for dev in epochs_by_device)

    for idx in range(max_series):
        fig, ax = plt.subplots(figsize=(5, 4))
        for dev in epochs_by_device:
            if idx >= len(epochs_by_device[dev]):
                continue
            ax.plot(
                times_by_device[dev],
                epochs_by_device[dev][idx],
                color=DEVICE_COLORS.get(dev, None),
                linewidth=1.5,
                label=dev,
            )

        _style_epoch_axis(
            ax,
            title=f"sub-{subject_choice} ses-{session_choice} | {trial_type} | Serie {idx + 1}",
        )
        _safe_legend(ax)
        plt.tight_layout()
        plt.show()


def _plot_intensity_per_row(trial_data, subject_choice, session_choice):
    """Plot one row per trial intensity and one column per series index."""
    if not trial_data:
        return

    trial_types = list(trial_data.keys())
    n_rows = len(trial_types)

    max_series = 1
    for trial_type in trial_types:
        epochs_by_device = trial_data[trial_type]['epochs_by_device']
        for dev in epochs_by_device:
            max_series = max(max_series, len(epochs_by_device[dev]))

    fig, axes = plt.subplots(
        n_rows,
        max_series,
        figsize=(4.5 * max_series, 3.6 * n_rows),
        squeeze=False,
    )

    for row_idx, trial_type in enumerate(trial_types):
        epochs_by_device = trial_data[trial_type]['epochs_by_device']
        times_by_device = trial_data[trial_type]['times_by_device']

        for col_idx in range(max_series):
            ax = axes[row_idx, col_idx]
            has_data = False

            for dev in epochs_by_device:
                if col_idx >= len(epochs_by_device[dev]):
                    continue
                has_data = True
                ax.plot(
                    times_by_device[dev],
                    epochs_by_device[dev][col_idx],
                    color=DEVICE_COLORS.get(dev, None),
                    linewidth=1.5,
                    label=dev,
                )

            _style_epoch_axis(ax)

            if row_idx == 0:
                ax.set_title(f'Serie {col_idx + 1}', fontsize=10)

            if col_idx == 0:
                ax.set_ylabel(f'{trial_type}\nΔHbR (uM)')

            if has_data and row_idx == 0 and col_idx == 0:
                _safe_legend(ax)

            if not has_data:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', alpha=0.6)

    fig.suptitle(f"sub-{subject_choice} ses-{session_choice} | One intensity per row", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_envelope_and_auc_per_set(
    subject_choice='001',
    session_choice='002',
    recording_choice='semaxone',
    task_choice=None,
    t_pre=5,
    t_post=40,
    auc_window=(0, None),
    envelope_mode='minmax',
    use_abs_auc=False,
    base_root=None,
):
    """Plot mean envelope and AUC for each trial_type set and device.

    The envelope can be either min/max across series (default) or mean +/- std.
    AUC is computed on the mean epoch within auc_window.
    """
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT

    if isinstance(recording_choice, str):
        recordings = [recording_choice]
    else:
        recordings = list(recording_choice)

    if not recordings:
        raise ValueError('recording_choice must contain at least one device.')

    if envelope_mode not in {'minmax', 'std'}:
        raise ValueError("envelope_mode must be 'minmax' or 'std'.")

    if not isinstance(auc_window, (tuple, list)) or len(auc_window) != 2:
        raise ValueError('auc_window must be a tuple/list: (t_start, t_end).')
    auc_tmin, auc_tmax = auc_window

    task_used = infer_task_from_session(
        session_choice,
        task_choice=task_choice,
        subject_choice=subject_choice,
        base_root=base_root,
    )
    events_file = build_events_file(base_root, subject_choice, session_choice, task_used)
    if not os.path.exists(events_file):
        raise FileNotFoundError(f'Events file not found: {events_file}')

    events_df_selected = pd.read_csv(events_file, sep='\t')
    print(f'Using events file: {events_file}')

    raw_by_device = {}
    for dev in recordings:
        try:
            raw_by_device[dev] = load_filtered_hb(
                subject_choice,
                session_choice,
                task_used,
                dev,
                base_root=base_root,
            )
        except Exception as e:
            print(f"Warning: unable to load {dev}: {e}")

    if not raw_by_device:
        raise RuntimeError('No device could be loaded.')

    trial_types = [
        t for t in sorted(events_df_selected['trial_type'].dropna().unique())
        if _is_displayable_trial_type(t)
    ]

    if not trial_types:
        print('No displayable trial_type found.')
        return {}

    fig, axes = plt.subplots(len(trial_types), 1, figsize=(9, 3.8 * len(trial_types)), squeeze=False)
    summary = {}

    for row_idx, trial_type in enumerate(trial_types):
        ax = axes[row_idx, 0]
        summary[trial_type] = {}
        has_data = False

        for dev, raw_selected in raw_by_device.items():
            epochs, times = epoch_signal(
                raw_selected,
                trial_type,
                t_pre=t_pre,
                t_post=t_post,
                events_df=events_df_selected,
            )
            if len(epochs) == 0:
                continue

            has_data = True
            mean_curve = np.mean(epochs, axis=0)

            if envelope_mode == 'std':
                std_curve = np.std(epochs, axis=0)
                lower = mean_curve - std_curve
                upper = mean_curve + std_curve
            else:
                lower = np.min(epochs, axis=0)
                upper = np.max(epochs, axis=0)

            auc_mask = times >= auc_tmin
            if auc_tmax is not None:
                auc_mask = auc_mask & (times <= auc_tmax)

            if np.any(auc_mask):
                auc_signal = np.abs(mean_curve[auc_mask]) if use_abs_auc else mean_curve[auc_mask]
                auc_value = _integrate_auc(auc_signal, times[auc_mask])
            else:
                auc_value = np.nan

            color = DEVICE_COLORS.get(dev, None)
            ax.fill_between(times, lower, upper, color=color, alpha=0.18)
            ax.plot(times, mean_curve, color=color, linewidth=2.0, label=f"{dev} (AUC={auc_value:.2f} uM*s)")

            summary[trial_type][dev] = {
                'auc_uM_s': auc_value,
                'n_sets': int(epochs.shape[0]),
            }

        _style_epoch_axis(
            ax,
            title=f"sub-{subject_choice} ses-{session_choice} | {trial_type}",
            ylabel='ΔHbR (uM)',
        )

        if has_data:
            _safe_legend(ax)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', alpha=0.6)

    fig.suptitle('Envelope and area under the curve for each set', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return summary


def display_sets_one_by_one(
    subject_choice='001',
    session_choice='002',
    recording_choice='semaxone',
    task_choice=None,
    t_pre=5,
    t_post=40,
    include_occlusion_events=False,
    base_root=None,
):
    """Load selected recordings and display epoch series grouped by trial type."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT

    if isinstance(recording_choice, str):
        recordings = [recording_choice]
    else:
        recordings = list(recording_choice)

    if not recordings:
        raise ValueError('recording_choice must contain at least one device.')

    task_used = infer_task_from_session(
        session_choice,
        task_choice=task_choice,
        subject_choice=subject_choice,
        base_root=base_root,
    )
    events_file = build_events_file(base_root, subject_choice, session_choice, task_used)
    if not os.path.exists(events_file):
        raise FileNotFoundError(f'Events file not found: {events_file}')

    events_df_selected = pd.read_csv(events_file, sep='\t')
    print(f'Using events file: {events_file}')

    raw_by_device = {}
    for dev in recordings:
        try:
            raw_by_device[dev] = load_filtered_hb(
                subject_choice,
                session_choice,
                task_used,
                dev,
                base_root=base_root,
            )
        except Exception as e:
            print(f"Warning: unable to load {dev}: {e}")

    if not raw_by_device:
        raise RuntimeError('No device could be loaded.')

    trial_types = [
        t for t in sorted(events_df_selected['trial_type'].dropna().unique())
        if include_occlusion_events or _is_displayable_trial_type(t)
    ]

    if not trial_types:
        print('No displayable trial_type found.')
        return

    print(f'Trial types found: {trial_types}')
    trial_data = {}

    for trial_type in trial_types:
        epochs_by_device = {}
        times_by_device = {}

        for dev, raw_selected in raw_by_device.items():
            epochs, times = epoch_signal(
                raw_selected,
                trial_type,
                t_pre=t_pre,
                t_post=t_post,
                events_df=events_df_selected,
            )
            if len(epochs) == 0:
                continue
            epochs_by_device[dev] = epochs
            times_by_device[dev] = times

        if not epochs_by_device:
            print(f'No valid epoch for {trial_type} on selected devices')
            continue

        trial_data[trial_type] = {
            'epochs_by_device': epochs_by_device,
            'times_by_device': times_by_device,
        }

    _plot_intensity_per_row(trial_data, subject_choice, session_choice)


def compare_devices_by_task(
    subject_id='001',
    base_root=None,
    intensities=None,
    make_plot=True,
    print_stats=True,
):
    """Compare Semaxone vs Portalite within concentric/eccentric tasks for one subject."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT
    if intensities is None:
        intensities = [30, 50]

    trial_map = {
        'concentric': {'trial30': 'Con30', 'trial50': 'Con50'},
        'eccentric': {'trial30': 'Ecc30', 'trial50': 'Ecc50'},
    }

    task_events = {}
    for task_name in ['concentric', 'eccentric']:
        found = _find_first_session_for_task(base_root, subject_id, task_name)
        if found is None:
            print(f"Warning: no {task_name} events found for subject {subject_id}")
            continue
        session_id, events_path = found
        task_events[task_name] = {
            'session': session_id,
            'events': pd.read_csv(events_path, sep='\t'),
        }

    def _process_task_for_devices(task_name):
        if task_name not in task_events:
            return _empty_device_stats(), _empty_device_stats()

        session = task_events[task_name]['session']
        events_df = task_events[task_name]['events']
        sem = process_device(
            subject_id,
            session,
            task_name,
            'semaxone',
            events_df,
            trial_map[task_name]['trial30'],
            trial_map[task_name]['trial50'],
            f'PROCESSING {task_name.upper()} DATA - SEMAXONE (ses-{session})',
            base_root=base_root,
        )
        port = process_device(
            subject_id,
            session,
            task_name,
            'portalite',
            events_df,
            trial_map[task_name]['trial30'],
            trial_map[task_name]['trial50'],
            f'PROCESSING {task_name.upper()} DATA - PORTALITE (ses-{session})',
            base_root=base_root,
        )
        return sem, port

    con_sem, con_port = _process_task_for_devices('concentric')
    ecc_sem, ecc_port = _process_task_for_devices('eccentric')

    # Keep legacy variable names for downstream notebook cells.
    hbr_con30_sem_means, hbr_con50_sem_means = con_sem['means30'], con_sem['means50']
    mean_hbr_con30_sem, std_hbr_con30_sem = con_sem['mean30'], con_sem['std30']
    mean_hbr_con50_sem, std_hbr_con50_sem = con_sem['mean50'], con_sem['std50']

    hbr_con30_port_means, hbr_con50_port_means = con_port['means30'], con_port['means50']
    mean_hbr_con30_port, std_hbr_con30_port = con_port['mean30'], con_port['std30']
    mean_hbr_con50_port, std_hbr_con50_port = con_port['mean50'], con_port['std50']
    concentric_portalite_available = con_port['available']

    hbr_ecc30_sem_means, hbr_ecc50_sem_means = ecc_sem['means30'], ecc_sem['means50']
    mean_hbr_ecc30_sem, std_hbr_ecc30_sem = ecc_sem['mean30'], ecc_sem['std30']
    mean_hbr_ecc50_sem, std_hbr_ecc50_sem = ecc_sem['mean50'], ecc_sem['std50']
    eccentric_semaxone_available = ecc_sem['available']

    hbr_ecc30_port_means, hbr_ecc50_port_means = ecc_port['means30'], ecc_port['means50']
    mean_hbr_ecc30_port, std_hbr_ecc30_port = ecc_port['mean30'], ecc_port['std30']
    mean_hbr_ecc50_port, std_hbr_ecc50_port = ecc_port['mean50'], ecc_port['std50']
    eccentric_portalite_available = ecc_port['available']

    # Plot
    if make_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        plot_device(
            ax1,
            intensities,
            [mean_hbr_con30_sem, mean_hbr_con50_sem],
            [std_hbr_con30_sem, std_hbr_con50_sem],
            [hbr_con30_sem_means, hbr_con50_sem_means],
            'blue',
            'darkblue',
            'lightblue',
            'o',
            'Semaxone',
        )
        if concentric_portalite_available:
            plot_device(
                ax1,
                intensities,
                [mean_hbr_con30_port, mean_hbr_con50_port],
                [std_hbr_con30_port, std_hbr_con50_port],
                [hbr_con30_port_means, hbr_con50_port_means],
                'red',
                'darkred',
                'lightcoral',
                's',
                'Portalite',
            )

        ax1.set_xlabel('Intensity', fontsize=13, fontweight='bold')
        ax1.set_ylabel('HbR Concentration (uM)', fontsize=13, fontweight='bold')
        ax1.set_title('HbR - Concentric (Semaxone vs Portalite)', fontsize=14, fontweight='bold')
        ax1.set_xticks(intensities)
        ax1.set_xticklabels(['30%', '50%'])
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')

        if eccentric_semaxone_available:
            plot_device(
                ax2,
                intensities,
                [mean_hbr_ecc30_sem, mean_hbr_ecc50_sem],
                [std_hbr_ecc30_sem, std_hbr_ecc50_sem],
                [hbr_ecc30_sem_means, hbr_ecc50_sem_means],
                'blue',
                'darkblue',
                'lightblue',
                'o',
                'Semaxone',
            )
        if eccentric_portalite_available:
            plot_device(
                ax2,
                intensities,
                [mean_hbr_ecc30_port, mean_hbr_ecc50_port],
                [std_hbr_ecc30_port, std_hbr_ecc50_port],
                [hbr_ecc30_port_means, hbr_ecc50_port_means],
                'red',
                'darkred',
                'lightcoral',
                's',
                'Portalite',
            )

        ax2.set_xlabel('Intensity', fontsize=13, fontweight='bold')
        ax2.set_ylabel('HbR Concentration (uM)', fontsize=13, fontweight='bold')
        ax2.set_title('HbR - Eccentric (Semaxone vs Portalite)', fontsize=14, fontweight='bold')
        ax2.set_xticks(intensities)
        ax2.set_xticklabels(['30%', '50%'])
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')

        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()
        y_min = min(y1_min, y2_min)
        y_max = max(y1_max, y2_max)
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()

    # Statistics
    if print_stats:
        print(f"\n{'=' * 70}")
        print('STATISTICS - HbR concentration comparison by device')
        print(f"{'=' * 70}")

        print_block(
            'CONCENTRIC - SEMAXONE',
            mean_hbr_con30_sem,
            std_hbr_con30_sem,
            len(hbr_con30_sem_means),
            mean_hbr_con50_sem,
            std_hbr_con50_sem,
            len(hbr_con50_sem_means),
        )
        if concentric_portalite_available:
            print_block(
                'CONCENTRIC - PORTALITE',
                mean_hbr_con30_port,
                std_hbr_con30_port,
                len(hbr_con30_port_means),
                mean_hbr_con50_port,
                std_hbr_con50_port,
                len(hbr_con50_port_means),
            )
        if eccentric_semaxone_available:
            print_block(
                'ECCENTRIC - SEMAXONE',
                mean_hbr_ecc30_sem,
                std_hbr_ecc30_sem,
                len(hbr_ecc30_sem_means),
                mean_hbr_ecc50_sem,
                std_hbr_ecc50_sem,
                len(hbr_ecc50_sem_means),
            )
        if eccentric_portalite_available:
            print_block(
                'ECCENTRIC - PORTALITE',
                mean_hbr_ecc30_port,
                std_hbr_ecc30_port,
                len(hbr_ecc30_port_means),
                mean_hbr_ecc50_port,
                std_hbr_ecc50_port,
                len(hbr_ecc50_port_means),
            )

        print(f"{'=' * 70}")

    return {
        'con_sem': con_sem,
        'con_port': con_port,
        'ecc_sem': ecc_sem,
        'ecc_port': ecc_port,
        'hbr_con30_sem_means': hbr_con30_sem_means,
        'hbr_con50_sem_means': hbr_con50_sem_means,
        'mean_hbr_con30_sem': mean_hbr_con30_sem,
        'std_hbr_con30_sem': std_hbr_con30_sem,
        'mean_hbr_con50_sem': mean_hbr_con50_sem,
        'std_hbr_con50_sem': std_hbr_con50_sem,
        'hbr_con30_port_means': hbr_con30_port_means,
        'hbr_con50_port_means': hbr_con50_port_means,
        'mean_hbr_con30_port': mean_hbr_con30_port,
        'std_hbr_con30_port': std_hbr_con30_port,
        'mean_hbr_con50_port': mean_hbr_con50_port,
        'std_hbr_con50_port': std_hbr_con50_port,
        'concentric_portalite_available': concentric_portalite_available,
        'hbr_ecc30_sem_means': hbr_ecc30_sem_means,
        'hbr_ecc50_sem_means': hbr_ecc50_sem_means,
        'mean_hbr_ecc30_sem': mean_hbr_ecc30_sem,
        'std_hbr_ecc30_sem': std_hbr_ecc30_sem,
        'mean_hbr_ecc50_sem': mean_hbr_ecc50_sem,
        'std_hbr_ecc50_sem': std_hbr_ecc50_sem,
        'eccentric_semaxone_available': eccentric_semaxone_available,
        'hbr_ecc30_port_means': hbr_ecc30_port_means,
        'hbr_ecc50_port_means': hbr_ecc50_port_means,
        'mean_hbr_ecc30_port': mean_hbr_ecc30_port,
        'std_hbr_ecc30_port': std_hbr_ecc30_port,
        'mean_hbr_ecc50_port': mean_hbr_ecc50_port,
        'std_hbr_ecc50_port': std_hbr_ecc50_port,
        'eccentric_portalite_available': eccentric_portalite_available,
    }


def compare_devices_all_subjects(
    subject_ids=None,
    base_root=None,
    intensities=None,
    make_plot=False,
    print_stats=False,
):
    """Run per-subject device comparison across all available subjects."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT
    if intensities is None:
        intensities = [30, 50]

    if subject_ids is None:
        subject_ids = sorted(
            d.replace('sub-', '')
            for d in os.listdir(base_root)
            if d.startswith('sub-') and os.path.isdir(os.path.join(base_root, d))
        )

    all_stats = {}
    print(f"Running comparison for {len(subject_ids)} subject(s)...")

    for subject_id in subject_ids:
        print(f"\n{'#' * 80}")
        print(f"SUBJECT {subject_id}")
        print(f"{'#' * 80}")

        try:
            all_stats[subject_id] = compare_devices_by_task(
                subject_id=subject_id,
                base_root=base_root,
                intensities=intensities,
                make_plot=make_plot,
                print_stats=print_stats,
            )
        except Exception as e:
            print(f"Skipping subject {subject_id}: {e}")

    print(f"\nFinished. Successful subjects: {len(all_stats)}/{len(subject_ids)}")
    return all_stats


def _extract_task_from_events_filename(events_filename):
    """Parse task name from BIDS-like events filename."""
    marker = '_task-'
    suffix = '_events.tsv'
    if marker not in events_filename or not events_filename.endswith(suffix):
        return None
    return events_filename.split(marker, 1)[1].rsplit(suffix, 1)[0]


def _iter_subject_session_task_events(base_root, subject_ids=None):
    """Iterate over existing subject/session/task events files in base_root."""
    if subject_ids is None:
        subject_ids = sorted(
            d.replace('sub-', '')
            for d in os.listdir(base_root)
            if d.startswith('sub-') and os.path.isdir(os.path.join(base_root, d))
        )

    for subject_id in subject_ids:
        sub_dir = os.path.join(base_root, f'sub-{subject_id}')
        if not os.path.isdir(sub_dir):
            continue

        for ses_name in sorted(os.listdir(sub_dir)):
            if not ses_name.startswith('ses-'):
                continue
            session_id = ses_name.replace('ses-', '')
            nirs_dir = os.path.join(sub_dir, ses_name, 'nirs')
            if not os.path.isdir(nirs_dir):
                continue

            for fname in sorted(os.listdir(nirs_dir)):
                if not fname.endswith('_events.tsv'):
                    continue
                task_name = _extract_task_from_events_filename(fname)
                if task_name is None:
                    continue
                yield subject_id, session_id, task_name, os.path.join(nirs_dir, fname)


def _default_output_dir_from_base_root(base_root):
    """Compute default Analysis_Results folder from BIDS base root."""
    project_root = os.path.dirname(os.path.dirname(base_root))
    return os.path.join(project_root, 'Analysis_Results')


def _is_displayable_trial_type(trial_type):
    """Return True for trial types that should be displayed in plots."""
    trial_text = str(trial_type).strip().lower()
    if not trial_text:
        return False
    if trial_text == 'mvc':
        return False
    if 'occlusion' in trial_text:
        return False
    return True


def build_hbr_summary_table(
    subject_ids=None,
    base_root=None,
    recording='portalite',
    window_duration=None,
    print_progress=True,
):
    """Build a subject/session/task HbR summary table for 30% and 50% intensities."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT

    trial_map = {
        'concentric': {'30': 'Con30', '50': 'Con50', 'mode': 'Concentric'},
        'eccentric': {'30': 'Ecc30', '50': 'Ecc50', 'mode': 'Eccentric'},
    }

    rows = []

    for subject_id, session_id, task_name, events_file in _iter_subject_session_task_events(
        base_root,
        subject_ids=subject_ids,
    ):
        task_key = task_name.lower()
        if task_key not in trial_map:
            if print_progress:
                print(f"Skipping unsupported task '{task_name}' for sub-{subject_id} ses-{session_id}")
            continue

        if print_progress:
            print(f"Processing sub-{subject_id} ses-{session_id} task-{task_name} ({recording})")

        try:
            events_df = pd.read_csv(events_file, sep='\t')
            raw_hb_filt = load_filtered_hb(
                subject=subject_id,
                session=session_id,
                task=task_name,
                recording=recording,
                base_root=base_root,
            )
            hbr_channels = get_hbr_channels(raw_hb_filt)

            if not hbr_channels:
                if print_progress:
                    print('  No HbR channels found, skipping')
                continue

            for intensity in ['30', '50']:
                trial_type = trial_map[task_key][intensity]
                means = extract_hbr_means(
                    raw_hb_filt,
                    events_df,
                    trial_type,
                    hbr_channels,
                    recording,
                    window_duration=window_duration,
                    verbose=False,
                )

                if not means:
                    if print_progress:
                        print(f"  No usable {trial_type} events")
                    continue

                mean_hbr, std_hbr = mean_std(means)
                rows.append(
                    {
                        'Subject': subject_id,
                        'Session': session_id,
                        'Task': task_name,
                        'Contraction_Mode': trial_map[task_key]['mode'],
                        'Intensity': f'{intensity}%',
                        'Mean_HbR_uM': mean_hbr,
                        'Std_HbR_uM': std_hbr,
                        'Number_of_Series': len(means),
                    }
                )

                if print_progress:
                    print(f"  {trial_type}: {mean_hbr:.3f} +/- {std_hbr:.3f} uM ({len(means)} series)")

        except Exception as e:
            if print_progress:
                print(f"  Error for sub-{subject_id} ses-{session_id} task-{task_name}: {e}")

    return pd.DataFrame(rows)


def export_hbr_summary_csv(
    subject_ids=None,
    base_root=None,
    recording='portalite',
    window_duration=None,
    output_filename='hbr_concentration_results.csv',
    output_dir=None,
    print_progress=True,
):
    """Export CSV with HbR summary rows: Subject, Session, Task, mode, intensity, mean, std, n."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT
    if output_dir is None:
        output_dir = _default_output_dir_from_base_root(base_root)

    results_df = build_hbr_summary_table(
        subject_ids=subject_ids,
        base_root=base_root,
        recording=recording,
        window_duration=window_duration,
        print_progress=print_progress,
    )

    if results_df.empty:
        if print_progress:
            print('No data processed; CSV was not created.')
        return {'results_df': results_df, 'output_path': None}

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    try:
        results_df.to_csv(output_path, index=False, sep=';', decimal=',')
    except PermissionError:
        fallback_dir = os.path.join(
            os.path.expanduser('~'),
            'Documents',
            'DigiMove',
            'Analysis_Results',
        )
        os.makedirs(fallback_dir, exist_ok=True)
        output_path = os.path.join(fallback_dir, output_filename)
        results_df.to_csv(output_path, index=False, sep=';', decimal=',')
        if print_progress:
            print('No write permission in target folder. Saved to fallback path.')

    if print_progress:
        print(f"\n{'=' * 70}")
        print('CSV EXPORT COMPLETE')
        print(f"{'=' * 70}")
        print(f'Total rows: {len(results_df)}')
        print(f'Output: {output_path}')
        print(f"{'=' * 70}")

    return {'results_df': results_df, 'output_path': output_path}


def _extract_window_means_by_chromophore(
    raw_obj,
    events_df,
    trial_type,
    window_duration,
    chromophore,
):
    """Compute one mean value per event window for HbR/HbO/HbDiff/HbTot."""
    if window_duration is None:
        window_duration = DEFAULT_WINDOW_DURATION

    onsets = events_df.loc[events_df['trial_type'] == trial_type, 'onset']
    means = []

    hbr_chs = get_hbr_channels(raw_obj)
    hbo_chs = get_hbo_channels(raw_obj)

    for onset in onsets:
        segment = raw_obj.copy().crop(tmin=onset, tmax=onset + window_duration)

        if chromophore == 'HbR':
            if not hbr_chs:
                continue
            data = segment.get_data(picks=hbr_chs) * 1e6
            means.append(float(np.mean(data)))
        elif chromophore == 'HbO':
            if not hbo_chs:
                continue
            data = segment.get_data(picks=hbo_chs) * 1e6
            means.append(float(np.mean(data)))
        elif chromophore in {'HbDiff', 'HbTot'}:
            if not hbo_chs or not hbr_chs:
                continue
            data_hbo = segment.get_data(picks=hbo_chs) * 1e6
            data_hbr = segment.get_data(picks=hbr_chs) * 1e6
            mean_hbo = float(np.mean(data_hbo))
            mean_hbr = float(np.mean(data_hbr))
            if chromophore == 'HbDiff':
                means.append(mean_hbo - mean_hbr)
            else:
                means.append(mean_hbo + mean_hbr)
        else:
            raise ValueError("chromophore must be one of: HbR, HbO, HbDiff, HbTot")

    return means


def build_hb_summary_table(
    subject_ids=None,
    base_root=None,
    recording='portalite',
    window_duration=None,
    include_all_participants=False,
    print_progress=True,
):
    """Build per-intensity summary table with mean HbR/HbO/HbDiff/HbTot for one device."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT

    trial_map = {
        'concentric': {'30': 'Con30', '50': 'Con50', 'mode': 'Concentric'},
        'eccentric': {'30': 'Ecc30', '50': 'Ecc50', 'mode': 'Eccentric'},
    }

    rows = []

    def _append_empty_row(subject_id, session_id, task_key, intensity):
        rows.append(
            {
                'Subject': subject_id,
                'Session': session_id,
                'Contraction_Mode': trial_map[task_key]['mode'],
                'Intensity': f'{intensity}%',
                'Mean_HbR_uM': np.nan,
                'Mean_HbO_uM': np.nan,
                'Mean_HbDiff_uM': np.nan,
                'Mean_HbTot_uM': np.nan,
            }
        )

    for subject_id, session_id, task_name, events_file in _iter_subject_session_task_events(
        base_root,
        subject_ids=subject_ids,
    ):
        task_key = task_name.lower()
        if task_key not in trial_map:
            continue

        if print_progress:
            print(f"Processing sub-{subject_id} ses-{session_id} task-{task_name} ({recording})")

        try:
            raw_path = BIDSPath(
                subject=subject_id,
                session=session_id,
                task=task_name,
                recording=recording,
                suffix='nirs',
                extension='.snirf',
                datatype='nirs',
                root=base_root,
            )

            if not os.path.exists(raw_path.fpath):
                if print_progress:
                    print(f"  Missing raw file: {raw_path.fpath}")
                if include_all_participants:
                    for intensity in ['30', '50']:
                        _append_empty_row(subject_id, session_id, task_key, intensity)
                continue

            events_df = pd.read_csv(events_file, sep='\t')
            raw_hb_filt = load_filtered_hb(
                subject=subject_id,
                session=session_id,
                task=task_name,
                recording=recording,
                base_root=base_root,
            )

            for intensity in ['30', '50']:
                trial_type = trial_map[task_key][intensity]

                hbr_vals = _extract_window_means_by_chromophore(
                    raw_hb_filt, events_df, trial_type, window_duration, 'HbR'
                )
                hbo_vals = _extract_window_means_by_chromophore(
                    raw_hb_filt, events_df, trial_type, window_duration, 'HbO'
                )
                hbdiff_vals = _extract_window_means_by_chromophore(
                    raw_hb_filt, events_df, trial_type, window_duration, 'HbDiff'
                )
                hbtot_vals = _extract_window_means_by_chromophore(
                    raw_hb_filt, events_df, trial_type, window_duration, 'HbTot'
                )

                n_series = min(len(hbr_vals), len(hbo_vals), len(hbdiff_vals), len(hbtot_vals))
                if n_series == 0:
                    if print_progress:
                        print(f"  No usable {trial_type} events")
                    if include_all_participants:
                        _append_empty_row(subject_id, session_id, task_key, intensity)
                    continue

                rows.append(
                    {
                        'Subject': subject_id,
                        'Session': session_id,
                        'Contraction_Mode': trial_map[task_key]['mode'],
                        'Intensity': f'{intensity}%',
                        'Mean_HbR_uM': float(np.mean(hbr_vals[:n_series])),
                        'Mean_HbO_uM': float(np.mean(hbo_vals[:n_series])),
                        'Mean_HbDiff_uM': float(np.mean(hbdiff_vals[:n_series])),
                        'Mean_HbTot_uM': float(np.mean(hbtot_vals[:n_series])),
                    }
                )

        except Exception as e:
            if print_progress:
                print(f"  Error for sub-{subject_id} ses-{session_id} task-{task_name}: {e}")
            if include_all_participants:
                for intensity in ['30', '50']:
                    _append_empty_row(subject_id, session_id, task_key, intensity)

    return pd.DataFrame(rows)


def export_hb_summary_csv_by_device(
    subject_ids=None,
    base_root=None,
    recordings=None,
    window_duration=None,
    output_dir=None,
    filename_prefix='hb_summary',
    include_all_participants=False,
    print_progress=True,
):
    """Export two CSV tables (one per device) with mean HbR/HbO/HbDiff/HbTot."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT
    if output_dir is None:
        output_dir = _default_output_dir_from_base_root(base_root)
    if recordings is None:
        recordings = ['portalite', 'semaxone']

    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for recording in recordings:
        df = build_hb_summary_table(
            subject_ids=subject_ids,
            base_root=base_root,
            recording=recording,
            window_duration=window_duration,
            include_all_participants=include_all_participants,
            print_progress=print_progress,
        )

        output_path = None
        if not df.empty:
            output_path = os.path.join(output_dir, f'{filename_prefix}_{recording}.csv')
            df.to_csv(output_path, index=False, sep=';', decimal=',')

        results[recording] = {
            'results_df': df,
            'output_path': output_path,
        }

    if print_progress:
        print(f"\n{'=' * 70}")
        print('CSV EXPORT COMPLETE (BY DEVICE)')
        print(f"{'=' * 70}")
        for recording in recordings:
            row_count = len(results[recording]['results_df'])
            out = results[recording]['output_path']
            print(f"{recording}: {row_count} rows | {out}")
        print(f"{'=' * 70}")

    return results


def _collect_metric_values(all_subject_stats, metric_name):
    """Collect finite numeric values for one metric across subjects."""
    values = []
    for stats in all_subject_stats.values():
        v = stats.get(metric_name, np.nan)
        if isinstance(v, (float, int, np.floating, np.integer)) and np.isfinite(v):
            values.append(float(v))
    return np.array(values, dtype=float)


def compare_devices_group_mean(
    subject_ids=None,
    base_root=None,
    intensities=None,
    make_plot=True,
    print_stats=True,
):
    """Compute and optionally plot HbR group mean across all available subjects."""
    if intensities is None:
        intensities = [30, 50]

    all_subject_stats = compare_devices_all_subjects(
        subject_ids=subject_ids,
        base_root=base_root,
        intensities=intensities,
        make_plot=False,
        print_stats=False,
    )

    if not all_subject_stats:
        raise RuntimeError('No subject could be processed for group mean.')

    def _mean_std_n(metric_name):
        vals = _collect_metric_values(all_subject_stats, metric_name)
        if vals.size == 0:
            return np.nan, np.nan, 0
        return float(np.mean(vals)), float(np.std(vals)), int(vals.size)

    metric_keys = [
        'mean_hbr_con30_sem',
        'mean_hbr_con50_sem',
        'mean_hbr_con30_port',
        'mean_hbr_con50_port',
        'mean_hbr_ecc30_sem',
        'mean_hbr_ecc50_sem',
        'mean_hbr_ecc30_port',
        'mean_hbr_ecc50_port',
    ]
    metric_stats = {
        key: {
            'mean': stats_tuple[0],
            'std': stats_tuple[1],
            'n': stats_tuple[2],
        }
        for key, stats_tuple in ((k, _mean_std_n(k)) for k in metric_keys)
    }

    mean_hbr_con30_sem = metric_stats['mean_hbr_con30_sem']['mean']
    std_hbr_con30_sem = metric_stats['mean_hbr_con30_sem']['std']
    n_con30_sem = metric_stats['mean_hbr_con30_sem']['n']
    mean_hbr_con50_sem = metric_stats['mean_hbr_con50_sem']['mean']
    std_hbr_con50_sem = metric_stats['mean_hbr_con50_sem']['std']
    n_con50_sem = metric_stats['mean_hbr_con50_sem']['n']
    mean_hbr_con30_port = metric_stats['mean_hbr_con30_port']['mean']
    std_hbr_con30_port = metric_stats['mean_hbr_con30_port']['std']
    n_con30_port = metric_stats['mean_hbr_con30_port']['n']
    mean_hbr_con50_port = metric_stats['mean_hbr_con50_port']['mean']
    std_hbr_con50_port = metric_stats['mean_hbr_con50_port']['std']
    n_con50_port = metric_stats['mean_hbr_con50_port']['n']
    mean_hbr_ecc30_sem = metric_stats['mean_hbr_ecc30_sem']['mean']
    std_hbr_ecc30_sem = metric_stats['mean_hbr_ecc30_sem']['std']
    n_ecc30_sem = metric_stats['mean_hbr_ecc30_sem']['n']
    mean_hbr_ecc50_sem = metric_stats['mean_hbr_ecc50_sem']['mean']
    std_hbr_ecc50_sem = metric_stats['mean_hbr_ecc50_sem']['std']
    n_ecc50_sem = metric_stats['mean_hbr_ecc50_sem']['n']
    mean_hbr_ecc30_port = metric_stats['mean_hbr_ecc30_port']['mean']
    std_hbr_ecc30_port = metric_stats['mean_hbr_ecc30_port']['std']
    n_ecc30_port = metric_stats['mean_hbr_ecc30_port']['n']
    mean_hbr_ecc50_port = metric_stats['mean_hbr_ecc50_port']['mean']
    std_hbr_ecc50_port = metric_stats['mean_hbr_ecc50_port']['std']
    n_ecc50_port = metric_stats['mean_hbr_ecc50_port']['n']

    if make_plot:
        x = intensities
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.errorbar(
            x,
            [mean_hbr_con30_sem, mean_hbr_con50_sem],
            yerr=[std_hbr_con30_sem, std_hbr_con50_sem],
            fmt='-o',
            marker='o',
            markersize=9,
            linewidth=2.5,
            color='pink',
            label='Semaxone',
            capsize=7,
        )
        ax1.errorbar(
            x,
            [mean_hbr_con30_port, mean_hbr_con50_port],
            yerr=[std_hbr_con30_port, std_hbr_con50_port],
            fmt='-s',
            marker='s',
            markersize=9,
            linewidth=2.5,
            color='yellow',
            label='Portalite',
            capsize=7,
        )
        ax1.set_xlabel('Intensity', fontsize=13, fontweight='bold')
        ax1.set_ylabel('HbR Concentration (uM)', fontsize=13, fontweight='bold')
        ax1.set_title('HbR Group Mean - Concentric (all subjects)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['30%', '50%'])
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax2.errorbar(
            x,
            [mean_hbr_ecc30_sem, mean_hbr_ecc50_sem],
            yerr=[std_hbr_ecc30_sem, std_hbr_ecc50_sem],
            fmt='-o',
            marker='o',
            markersize=9,
            linewidth=2.5,
            color='pink',
            label='Semaxone',
            capsize=7,
        )
        ax2.errorbar(
            x,
            [mean_hbr_ecc30_port, mean_hbr_ecc50_port],
            yerr=[std_hbr_ecc30_port, std_hbr_ecc50_port],
            fmt='-s',
            marker='s',
            markersize=9,
            linewidth=2.5,
            color='yellow',
            label='Portalite',
            capsize=7,
        )
        ax2.set_xlabel('Intensity', fontsize=13, fontweight='bold')
        ax2.set_ylabel('HbR Concentration (uM)', fontsize=13, fontweight='bold')
        ax2.set_title('HbR Group Mean - Eccentric (all subjects)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['30%', '50%'])
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')

        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()
        y_min = min(y1_min, y2_min)
        y_max = max(y1_max, y2_max)
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()

    if print_stats:
        subject_ids_used = sorted(all_subject_stats.keys())
        print(f"Subjects used for group mean: {subject_ids_used}")
        print(f"\n{'=' * 70}")
        print('GROUP STATISTICS - HbR mean across subjects')
        print(f"{'=' * 70}")
        print(
            f"CONCENTRIC - SEMAXONE: 30%={mean_hbr_con30_sem:.3f}+-{std_hbr_con30_sem:.3f} "
            f"(n={n_con30_sem}), 50%={mean_hbr_con50_sem:.3f}+-{std_hbr_con50_sem:.3f} (n={n_con50_sem})"
        )
        print(
            f"CONCENTRIC - PORTALITE: 30%={mean_hbr_con30_port:.3f}+-{std_hbr_con30_port:.3f} "
            f"(n={n_con30_port}), 50%={mean_hbr_con50_port:.3f}+-{std_hbr_con50_port:.3f} (n={n_con50_port})"
        )
        print(
            f"ECCENTRIC - SEMAXONE: 30%={mean_hbr_ecc30_sem:.3f}+-{std_hbr_ecc30_sem:.3f} "
            f"(n={n_ecc30_sem}), 50%={mean_hbr_ecc50_sem:.3f}+-{std_hbr_ecc50_sem:.3f} (n={n_ecc50_sem})"
        )
        print(
            f"ECCENTRIC - PORTALITE: 30%={mean_hbr_ecc30_port:.3f}+-{std_hbr_ecc30_port:.3f} "
            f"(n={n_ecc30_port}), 50%={mean_hbr_ecc50_port:.3f}+-{std_hbr_ecc50_port:.3f} (n={n_ecc50_port})"
        )
        print(f"{'=' * 70}")

    return {
        'all_subject_stats': all_subject_stats,
        'subject_ids_used': sorted(all_subject_stats.keys()),
        'mean_hbr_con30_sem': mean_hbr_con30_sem,
        'std_hbr_con30_sem': std_hbr_con30_sem,
        'n_con30_sem': n_con30_sem,
        'mean_hbr_con50_sem': mean_hbr_con50_sem,
        'std_hbr_con50_sem': std_hbr_con50_sem,
        'n_con50_sem': n_con50_sem,
        'mean_hbr_con30_port': mean_hbr_con30_port,
        'std_hbr_con30_port': std_hbr_con30_port,
        'n_con30_port': n_con30_port,
        'mean_hbr_con50_port': mean_hbr_con50_port,
        'std_hbr_con50_port': std_hbr_con50_port,
        'n_con50_port': n_con50_port,
        'mean_hbr_ecc30_sem': mean_hbr_ecc30_sem,
        'std_hbr_ecc30_sem': std_hbr_ecc30_sem,
        'n_ecc30_sem': n_ecc30_sem,
        'mean_hbr_ecc50_sem': mean_hbr_ecc50_sem,
        'std_hbr_ecc50_sem': std_hbr_ecc50_sem,
        'n_ecc50_sem': n_ecc50_sem,
        'mean_hbr_ecc30_port': mean_hbr_ecc30_port,
        'std_hbr_ecc30_port': std_hbr_ecc30_port,
        'n_ecc30_port': n_ecc30_port,
        'mean_hbr_ecc50_port': mean_hbr_ecc50_port,
        'std_hbr_ecc50_port': std_hbr_ecc50_port,
        'n_ecc50_port': n_ecc50_port,
    }


def print_average_hbr_mvc(
    subject_id='001',
    base_root=None,
    devices=None,
    window_duration=None,
):
    """Print average HbR at 30% and 50% MVC for concentric and eccentric.

    Printed values are computed as:
    average_HbR(condition) = mean(series_mean_1, ..., series_mean_n)
    where each series_mean is the mean HbR over selected HbR channels and the full effort window.
    """
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT
    if devices is None:
        devices = ['semaxone', 'portalite']

    trial_map = {
        'concentric': {'30': 'Con30', '50': 'Con50'},
        'eccentric': {'30': 'Ecc30', '50': 'Ecc50'},
    }

    print(f"\n{'=' * 80}")
    print(f"AVERAGE HbR AT 30% AND 50% MVC - SUBJECT {subject_id}")
    print(f"{'=' * 80}")

    summary = {}
    for task_name in ['concentric', 'eccentric']:
        found = _find_first_session_for_task(base_root, subject_id, task_name)
        if found is None:
            print(f"\n{task_name.upper()}: no events file found")
            continue

        session_id, events_path = found
        events_df = pd.read_csv(events_path, sep='\t')
        summary[task_name] = {}

        print(f"\n{task_name.upper()} (ses-{session_id})")
        for dev in devices:
            try:
                raw_hb = load_filtered_hb(
                    subject=subject_id,
                    session=session_id,
                    task=task_name,
                    recording=dev,
                    base_root=base_root,
                )

                avg_30, series_30 = compute_condition_hbr_average(
                    raw_hb,
                    events_df,
                    trial_map[task_name]['30'],
                    window_duration=window_duration,
                )
                avg_50, series_50 = compute_condition_hbr_average(
                    raw_hb,
                    events_df,
                    trial_map[task_name]['50'],
                    window_duration=window_duration,
                )

                summary[task_name][dev] = {
                    'session': session_id,
                    'avg_30': avg_30,
                    'avg_50': avg_50,
                    'n_30': len(series_30),
                    'n_50': len(series_50),
                }

                print(
                    f"  {dev}: 30% = {avg_30:.3f} uM (n={len(series_30)}), "
                    f"50% = {avg_50:.3f} uM (n={len(series_50)})"
                )
            except Exception as e:
                print(f"  {dev}: unavailable ({e})")

    print(f"{'=' * 80}")
    return summary

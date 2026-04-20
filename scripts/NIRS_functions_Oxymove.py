import os
import h5py
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
from scipy.signal import hilbert

# Install the external dependencies with pip if they are not already available:
# pip install mne matplotlib numpy pandas mne-bids scipy h5py

"""# NIRS Oxymove Utilities

Functions grouped by purpose to make the module easier to read and maintain.

## 1) Internal helpers
- _empty_device_stats
- _find_first_session_for_task
- _style_epoch_axis
- _safe_legend
- _extract_task_from_events_filename
- _iter_subject_session_task_events
- _default_output_dir_from_base_root
- _collect_metric_values

## 2) Signal loading and preprocessing
- load_filtered_hb
- get_hbr_channels
- epoch_signal

## 3) HbR metric extraction
- extract_hbr_means
- compute_condition_hbr_average
- mean_std
- extract_temporal_features

## 4) Visualization
- plot_device
- print_block
- plot_temporal_comparison
- _plot_epochs_one_by_one
- _plot_intensity_per_row
- plot_envelope_and_auc_per_set
- plot_hbr_hbo_mean_and_auc_by_device
- display_sets_one_by_one

## 5) Task/session mapping logic
- infer_task_from_session
- build_events_file

## 6) Comparative analyses and exports
- process_device
- compare_devices_by_task
- compare_devices_all_subjects
- build_hbr_summary_table
- export_hbr_summary_csv
- compare_devices_group_mean
- print_average_hbr_mvc
"""


DEFAULT_BASE_ROOT = r'C:\Program Files\DigiMove\DigiMove\DataAnalysisProject\data\MOXY-bids' # Default root directory for BIDS data
DEFAULT_ANALYSIS_RESULTS_DIR = r'C:\Program Files\DigiMove\DigiMove\DataAnalysisProject\Analysis_Results'
DEFAULT_WINDOW_DURATION = 30 # Default duration (in seconds) of the analysis window after trial onset
DEVICE_COLORS = {'semaxone': "#4E0451", 'portalite': "#4F8A02"} 
CHROMOPHORE_COLORS = {
    'HbR': 'red',
    'HbO': 'blue',

    'HbDiff': 'yellow',
    'HbTot': 'green',
}


# -----------------------------------------------------------------------------
# 1) Structure
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


def _style_epoch_axis(ax, title=None, xlabel='Time (s)', ylabel='ΔHbR (uM)'):
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


def _decode_snirf_text(value):
    """Decode SNIRF string values to plain Python text."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def _get_snirf_sensor_pairs(snirf_path):
    """Read source-detector pairs and distances from a SNIRF file."""
    with h5py.File(snirf_path, 'r') as snirf_file:
        probe = snirf_file['nirs']['probe']
        source_positions = np.asarray(probe['sourcePos3D'][...], dtype=float)
        detector_positions = np.asarray(probe['detectorPos3D'][...], dtype=float)
        source_labels = [_decode_snirf_text(value) for value in probe['sourceLabels'][...]]
        detector_labels = [_decode_snirf_text(value) for value in probe['detectorLabels'][...]]

    pairs = []
    for source_idx, source_position in enumerate(source_positions, start=1):
        for detector_idx, detector_position in enumerate(detector_positions, start=1):
            distance_mm = float(np.linalg.norm(source_position - detector_position) * 1000.0)
            pairs.append({
                'source_idx': source_idx,
                'detector_idx': detector_idx,
                'source_label': source_labels[source_idx - 1],
                'detector_label': detector_labels[detector_idx - 1],
                'distance_mm': distance_mm,
            })

    return pairs


def _select_sensor_pairs_for_recording(snirf_path, recording):
    """Return the sensor pair or pairs to keep for a given device recording."""
    recording_key = str(recording).strip().lower()
    sensor_pairs = _get_snirf_sensor_pairs(snirf_path)

    def _select_requested_pairs(requested_pairs):
        selected_pairs = []
        for requested_pair in requested_pairs:
            match = next(
                (
                    pair
                    for pair in sensor_pairs
                    if (pair['source_idx'], pair['detector_idx']) == requested_pair
                ),
                None,
            )
            if match is None:
                raise RuntimeError(
                    f"Requested sensor pair S{requested_pair[0]}_D{requested_pair[1]} "
                    f"not found in {os.path.basename(snirf_path)}."
                )
            selected_pairs.append(match)
        return selected_pairs

    if recording_key == 'portalite':
        # Portalite: keep only S2_D1.
        return _select_requested_pairs([(2, 1)])

    if recording_key == 'semaxone':
        # Semaxone: keep S2_D1, S2_D3, S1_D2, and S1_D4.
        requested_pairs = [(2, 1), (2, 3), (1, 2), (1, 4)]
        return _select_requested_pairs(requested_pairs)

    raise ValueError(f"Unsupported recording '{recording}'. Expected 'portalite' or 'semaxone'.")


def _pick_device_sensor_channels(raw_obj, recording):
    """Keep only the sensor channels requested for a specific device."""
    if not raw_obj.filenames:
        raise RuntimeError('Cannot resolve the SNIRF file for channel selection.')

    sensor_pairs = _select_sensor_pairs_for_recording(raw_obj.filenames[0], recording)
    selected_channels = []
    for sensor_pair in sensor_pairs:
        channel_prefix = f"S{sensor_pair['source_idx']}_D{sensor_pair['detector_idx']} "
        selected_channels.extend(ch for ch in raw_obj.ch_names if ch.startswith(channel_prefix))

    if not selected_channels:
        raise RuntimeError(
            f"No channels found for {recording} selected sensor pair(s)."
        )

    selected_channels = list(dict.fromkeys(selected_channels))
    return raw_obj.copy().pick(selected_channels)


# -----------------------------------------------------------------------------
# 2) Signal loading and preprocessing
# -----------------------------------------------------------------------------
def load_filtered_hb(subject, session, task, recording, base_root=None):
    """Load SNIRF from BIDS, keep the device-specific sensor, convert to Hb, then filter."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT

    # Build the BIDS path that points to the requested SNIRF file.
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
    # Read the raw optical signal, convert it to concentration, and filter it.
    raw_in = read_raw_bids(raw_path)
    raw_in = _pick_device_sensor_channels(raw_in, recording)
    raw_od = mne.preprocessing.nirs.optical_density(raw_in)
    raw_hb = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=4)
    # Keep only the frequency band that is used by the analysis pipeline.
    return raw_hb.copy().filter(
        l_freq=0.01,
        h_freq=4,
        method='iir', 
        iir_params=dict(order=4, ftype='butter'),
    )


def get_hbr_channels(raw_obj):
    """Return channel names corresponding to HbR signal."""
    # Channel names usually encode the chromophore type in their label.
    return [ch for ch in raw_obj.ch_names if 'hbr' in ch.lower()]


def get_hbo_channels(raw_obj):
    """Return channel names corresponding to HbO signal."""
    # Reuse the same naming convention to isolate HbO channels.
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
# 3) hbR metrics
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

    # Convert selected channel names to integer indices in the raw data array.
    hbr_idx = [raw_obj.ch_names.index(ch) for ch in hbr_channels if ch in raw_obj.ch_names]
    if not hbr_idx:
        return []

    means = []
    # Select only the requested trial type and use its onset times.
    trial_events = events_df.loc[events_df['trial_type'] == trial_type, 'onset']
    if verbose:
        print(f"\nProcessing {trial_type} series ({label})...")

    for onset in trial_events:
        # Crop one fixed analysis window starting at the trial onset.
        segment = raw_obj.copy().crop(tmin=onset, tmax=onset + window_duration)
        data_array = segment.get_data()
        # Average all HbR samples in the window and convert to micromolar.
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
    # Reuse the lower-level extractor so the definition stays identical everywhere.
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
# 4) Visualization
# -----------------------------------------------------------------------------
def plot_device(ax, x, means_pair, std_pair, series_pair, color_line, color_30, color_50, marker, label):
    """Plot 30/50% means, error bars, and individual series for one device."""
    if np.isnan(means_pair[0]) or np.isnan(means_pair[1]):
        return
    # Show mean markers only (no connecting segment between intensities).
    ax.plot(x, means_pair, marker=marker, markersize=10, linestyle='None', color=color_line, label=label)
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
# 6) Comparative analyses and exports
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
    # Sampling frequency defines how many samples correspond to the requested time window.
    sfreq = raw_hb_filt.info['sfreq']
    chromophore_key = str(chromophore).strip().lower()
    if chromophore_key in {'hbtot', 'hbdiff'}:
        # HbTot and HbDiff are derived from both HbO and HbR channels.
        hbo_chs = get_hbo_channels(raw_hb_filt)
        hbr_chs = get_hbr_channels(raw_hb_filt)
        if not hbo_chs or not hbr_chs:
            return np.array([]), np.array([])
        hbo_data_full = raw_hb_filt.get_data(picks=hbo_chs) * 1e6
        hbr_data_full = raw_hb_filt.get_data(picks=hbr_chs) * 1e6
        n_samples = min(hbo_data_full.shape[1], hbr_data_full.shape[1])
    else:
        # For a single chromophore, only keep the matching channels.
        signal_chs = _get_chromophore_channels(raw_hb_filt, chromophore=chromophore)
        if not signal_chs:
            return np.array([]), np.array([])
        data_full = raw_hb_filt.get_data(picks=signal_chs) * 1e6
        n_samples = data_full.shape[1]
    # Convert time windows into sample counts.
    n_pre = int(t_pre * sfreq)
    n_post    = int(t_post * sfreq)
    epochs    = []

    if events_df is not None:
        # Prefer the events table when it is available because it already contains trial onsets.
        trial_events = events_df[events_df['trial_type'] == trial_type]
        onset_samples = [int(row['onset'] * sfreq) for _, row in trial_events.iterrows()]
    else:
        # Fall back to annotations if the events TSV is not available.
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
            # Skip epochs that would extend outside the available recording.
            continue

        if chromophore_key == 'hbtot':
            segment = np.mean(hbo_data_full[:, start:stop], axis=0) + np.mean(hbr_data_full[:, start:stop], axis=0)
        elif chromophore_key == 'hbdiff':
            segment = np.mean(hbo_data_full[:, start:stop], axis=0) - np.mean(hbr_data_full[:, start:stop], axis=0)
        else:
            segment = np.mean(data_full[:, start:stop], axis=0)

        # Remove the baseline so all epochs are aligned to the pre-trial reference level.
        n_baseline = int(2 * sfreq)
        baseline = np.median(segment[n_pre - n_baseline:n_pre])
        segment -= baseline

        epochs.append(segment)

    if not epochs:
        return np.array([]), np.array([])

    # Trim to the shortest epoch so the output array has a consistent length.
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

    # Normalize the recording input to a list so the loop below can treat one or many devices the same way.
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

    # Resolve the task name before touching the filesystem.
    task_used = infer_task_from_session(
        session_choice,
        task_choice=task_choice,
        subject_choice=subject_choice,
        base_root=base_root,
    )
    # Events drive the set of displayable trial types and the epoch extraction step.
    events_file = build_events_file(base_root, subject_choice, session_choice, task_used)
    if not os.path.exists(events_file):
        raise FileNotFoundError(f'Events file not found: {events_file}')

    events_df_selected = pd.read_csv(events_file, sep='\t')
    print(f'Using events file: {events_file}')

    # Load every requested device up front so plotting and summary computation can reuse the same raw objects.
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

    # Only keep trial labels that are meant to be shown in analysis plots.
    trial_types = [
        t for t in sorted(events_df_selected['trial_type'].dropna().unique())
        if _is_displayable_trial_type(t)
    ]

    if not trial_types:
        print('No displayable trial_type found.')
        return {}

    summary = {}

    if plot_mode == 'combined':
        # In combined mode, each row is a trial type and each column is a chromophore.
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
                    # Build all epochs for this trial type, chromophore, and device.
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

                    # Use either standard deviation or min/max to build the visible envelope.
                    if envelope_mode == 'std':
                        std_curve = np.std(epochs, axis=0)
                        lower = mean_curve - std_curve
                        upper = mean_curve + std_curve
                    else:
                        lower = np.min(epochs, axis=0)
                        upper = np.max(epochs, axis=0)

                    # Integrate only the portion of the curve that falls inside the requested AUC window.
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
        # In by-device mode, each figure is dedicated to one device.
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

                    # Reuse the same epoch extraction logic for each chromophore.
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

                    # The envelope shows dispersion across repeated sets.
                    if envelope_mode == 'std':
                        std_curve = np.std(epochs, axis=0)
                        lower = mean_curve - std_curve
                        upper = mean_curve + std_curve
                    else:
                        lower = np.min(epochs, axis=0)
                        upper = np.max(epochs, axis=0)

                    # Compute one AUC per plotted mean curve so the legend carries the summary value.
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


def plot_hbr_hbo_mean_and_auc_by_device(
    subject_choice='001',
    session_choice='002',
    recording_choice=('semaxone', 'portalite'),
    task_choice=None,
    t_pre=5,
    t_post=40,
    auc_window=(0, None),
    use_abs_envelope=False,
    use_abs_auc=False,
    auc_plot_style='bar',
    base_root=None,
):
    """Plot HbR/HbO upper envelopes and AUC in separate figures, one set per device.

    Figure 1 (per device): HbR and HbO analytic envelopes on the same axes.
    If use_abs_envelope=True, the analytic envelope is computed from |signal|.
    Figure 2 (per device): AUC as bars ('bar') or shaded area under each curve ('area').
    """
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT

    if isinstance(recording_choice, str):
        recordings = [recording_choice]
    else:
        recordings = list(recording_choice)

    if not recordings:
        raise ValueError('recording_choice must contain at least one device.')

    if not isinstance(auc_window, (tuple, list)) or len(auc_window) != 2:
        raise ValueError('auc_window must be a tuple/list: (t_start, t_end).')
    auc_tmin, auc_tmax = auc_window

    if auc_plot_style not in {'bar', 'area'}:
        raise ValueError("auc_plot_style must be 'bar' or 'area'.")

    chromophores = ['HbR', 'HbO']

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
        raise RuntimeError('No displayable trial_type found.')

    envelope_payload = {}
    auc_payload = {}

    # Compute curves and AUC once, then reuse for all plots.
    for trial_type in trial_types:
        envelope_payload[trial_type] = {}
        auc_payload[trial_type] = {}
        for chromophore in chromophores:
            envelope_payload[trial_type][chromophore] = {}
            auc_payload[trial_type][chromophore] = {}
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

                mean_curve = np.mean(epochs, axis=0)
                signal_for_envelope = np.abs(mean_curve) if use_abs_envelope else mean_curve
                analytic_signal = hilbert(signal_for_envelope)
                amplitude_envelope = np.abs(analytic_signal)
                auc_mask = times >= auc_tmin
                if auc_tmax is not None:
                    auc_mask = auc_mask & (times <= auc_tmax)

                if np.any(auc_mask):
                    auc_signal = np.abs(mean_curve[auc_mask]) if use_abs_auc else mean_curve[auc_mask]
                    auc_value = _integrate_auc(auc_signal, times[auc_mask])
                else:
                    auc_value = np.nan

                envelope_payload[trial_type][chromophore][dev] = {
                    'times': times,
                    'envelope': amplitude_envelope,
                    'mean_for_auc': mean_curve,
                }
                auc_payload[trial_type][chromophore][dev] = auc_value

    chromo_color = {'HbR': 'red', 'HbO': 'blue'}

    # Figure 1: one analytic-envelope figure per device.
    for dev in raw_by_device:
        fig, axes = plt.subplots(
            len(trial_types),
            1,
            figsize=(9.2, 3.8 * len(trial_types)),
            squeeze=False,
        )

        for row_idx, trial_type in enumerate(trial_types):
            ax = axes[row_idx, 0]
            has_data = False

            for chromophore in chromophores:
                payload = envelope_payload[trial_type][chromophore].get(dev, None)
                if payload is None:
                    continue
                has_data = True
                color = chromo_color.get(chromophore, None)
                ax.plot(
                    payload['times'],
                    payload['envelope'],
                    color=color,
                    linewidth=2.0,
                    label=f"{chromophore} envelope",
                )

            _style_epoch_axis(
                ax,
                title=f"sub-{subject_choice} ses-{session_choice} | {dev} | {trial_type}",
                ylabel='|ΔHb| (uM)' if use_abs_envelope else 'ΔHb (uM)',
            )
            if has_data:
                _safe_legend(ax)
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', alpha=0.6)

        fig.suptitle(f'HbR/HbO analytic envelopes - {dev}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # Figure 2: one AUC figure per device.
    for dev in raw_by_device:
        fig, axes = plt.subplots(
            len(trial_types),
            1,
            figsize=(8.8, 3.8 * len(trial_types)),
            squeeze=False,
        )

        for row_idx, trial_type in enumerate(trial_types):
            ax = axes[row_idx, 0]
            has_data = False

            if auc_plot_style == 'bar':
                x = np.arange(len(chromophores))
                vals = [auc_payload[trial_type][ch].get(dev, np.nan) for ch in chromophores]

                ax.bar(
                    x,
                    vals,
                    width=0.55,
                    color=[chromo_color.get(ch, 'gray') for ch in chromophores],
                    alpha=0.85,
                )
                ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(chromophores)
                ax.set_ylabel('AUC (uM*s)')
                ax.set_title(f"sub-{subject_choice} ses-{session_choice} | {dev} | {trial_type} | AUC")
                ax.grid(True, axis='y', alpha=0.2)

                if np.all(np.isnan(vals)):
                    ax.text(0.5, 0.5, 'No AUC data', transform=ax.transAxes, ha='center', va='center', alpha=0.6)
            else:
                for chromophore in chromophores:
                    payload = envelope_payload[trial_type][chromophore].get(dev, None)
                    if payload is None:
                        continue

                    times = payload['times']
                    mean_curve = payload['mean_for_auc']
                    auc_mask = times >= auc_tmin
                    if auc_tmax is not None:
                        auc_mask = auc_mask & (times <= auc_tmax)
                    if not np.any(auc_mask):
                        continue

                    has_data = True
                    auc_curve = np.abs(mean_curve) if use_abs_auc else mean_curve
                    auc_value = auc_payload[trial_type][chromophore].get(dev, np.nan)
                    color = chromo_color.get(chromophore, 'gray')

                    ax.plot(times, mean_curve, color=color, linewidth=1.8, label=f"{chromophore} mean")
                    ax.fill_between(
                        times[auc_mask],
                        0,
                        auc_curve[auc_mask],
                        color=color,
                        alpha=0.25,
                        label=f"{chromophore} AUC={auc_value:.2f} uM*s",
                    )

                _style_epoch_axis(
                    ax,
                    title=f"sub-{subject_choice} ses-{session_choice} | {dev} | {trial_type} | AUC area",
                    ylabel='ΔHb (uM)',
                )
                if has_data:
                    _safe_legend(ax)
                else:
                    ax.text(0.5, 0.5, 'No AUC data', transform=ax.transAxes, ha='center', va='center', alpha=0.6)

        fig.suptitle(f'HbR/HbO AUC - {dev} ({auc_plot_style})', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return auc_payload


def plot_concentric_vs_eccentric_envelopes(
    subject_choice='001',
    recording_choice=('semaxone', 'portalite'),
    t_pre=5,
    t_post=40,
    chromophore='HbR',
    use_abs_envelope=True,
    base_root=None,
):
    """Compare concentric vs eccentric envelopes for 30% and 50% for one subject.

    One figure is produced per device. Each figure contains 2 subplots:
    - intensity 30%: Con30 vs Ecc30
    - intensity 50%: Con50 vs Ecc50
    """
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT

    if isinstance(recording_choice, str):
        recordings = [recording_choice]
    else:
        recordings = list(recording_choice)

    if not recordings:
        raise ValueError('recording_choice must contain at least one device.')

    chromo_key = str(chromophore).strip().lower()
    if chromo_key == 'hbr':
        chromophore = 'HbR'
    elif chromo_key == 'hbo':
        chromophore = 'HbO'
    else:
        raise ValueError("chromophore must be 'HbR' or 'HbO'.")

    task_to_session_events = {}
    for task_name in ['concentric', 'eccentric']:
        found = _find_first_session_for_task(base_root, subject_choice, task_name)
        if found is None:
            raise FileNotFoundError(
                f"No events file found for task '{task_name}' and subject {subject_choice}."
            )
        session_id, events_file = found
        task_to_session_events[task_name] = {
            'session': session_id,
            'events_df': pd.read_csv(events_file, sep='\t'),
        }

    trial_map = {
        30: {'concentric': 'Con30', 'eccentric': 'Ecc30'},
        50: {'concentric': 'Con50', 'eccentric': 'Ecc50'},
    }
    # Use task-specific colors that do not overlap with HbR/HbO red/blue convention.
    task_colors = {'concentric': '#2E8B57', 'eccentric': '#E67E22'}
    summary = {}

    for dev in recordings:
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), squeeze=False)
        fig_axes = axes[0]
        summary[dev] = {}

        for col_idx, intensity in enumerate([30, 50]):
            ax = fig_axes[col_idx]
            summary[dev][intensity] = {}
            has_data = False

            for task_name in ['concentric', 'eccentric']:
                session_id = task_to_session_events[task_name]['session']
                events_df = task_to_session_events[task_name]['events_df']
                trial_type = trial_map[intensity][task_name]

                try:
                    raw_selected = load_filtered_hb(
                        subject=subject_choice,
                        session=session_id,
                        task=task_name,
                        recording=dev,
                        base_root=base_root,
                    )
                except Exception as e:
                    print(
                        f"Warning: unable to load {dev} for {task_name} "
                        f"(sub-{subject_choice} ses-{session_id}): {e}"
                    )
                    continue

                epochs, times = epoch_signal(
                    raw_selected,
                    trial_type,
                    t_pre=t_pre,
                    t_post=t_post,
                    events_df=events_df,
                    chromophore=chromophore,
                )
                if len(epochs) == 0:
                    continue

                has_data = True
                mean_curve = np.mean(epochs, axis=0)
                # Hilbert envelope is the analytic signal amplitude (always non-negative).
                signal_for_env = np.abs(mean_curve) if use_abs_envelope else mean_curve
                envelope = np.abs(hilbert(signal_for_env))

                ax.plot(
                    times,
                    envelope,
                    color=task_colors[task_name],
                    linewidth=2.0,
                    label=f"{task_name.capitalize()} ({trial_type}, n={len(epochs)})",
                )

                summary[dev][intensity][task_name] = {
                    'session': session_id,
                    'trial_type': trial_type,
                    'n_sets': int(len(epochs)),
                }

            _style_epoch_axis(
                ax,
                title=f"sub-{subject_choice} | {dev} | {intensity}%",
                ylabel='ΔHHb (uM)' if use_abs_envelope else 'ΔHHb (uM)',
            )

            if has_data:
                _safe_legend(ax)
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', alpha=0.6)

        fig.suptitle(
            f"{chromophore} envelope comparison: Concentric vs Eccentric - {dev}",
            fontsize=12,
            fontweight='bold',
        )
        plt.tight_layout()
        plt.show()

    return summary

def plot_temporal_comparison(results):
    """Plot each temporal series separately for concentric and eccentric comparisons."""
    modes = [('Con', 'Concentric'), ('Ecc', 'Eccentric')]

    for mode, mode_label in modes:
        for intensity in ['30', '50']:
            key = f'{mode}{intensity}'

            # Keep only devices that are present in the results dictionary.
            data = {
                dev: results[f'{key}_{dev}']
                for dev in ['semaxone', 'portalite']
                if f'{key}_{dev}' in results
            }
            if not data:
                continue

            # Use the largest series count so every available trace gets a figure.
            n = max(len(d['epochs']) for d in data.values())

            # Draw one figure per series so devices can be compared trace by trace.
            for i in range(n):
                fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
                for dev, d in data.items():
                    if i >= len(d['epochs']):
                        continue
                    ax.plot(d['times'], d['epochs'][i],
                            color=DEVICE_COLORS[dev], linewidth=1.5, label=dev)

                _style_epoch_axis(
                    ax,
                    title=f'{mode_label} — {intensity}% — Series {i + 1}',
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
# 5) Task/session mapping logic
# -----------------------------------------------------------------------------
def infer_task_from_session(
    session_choice,
    task_choice=None,
    subject_choice=None,
    base_root=None,
):
    """Infer task from explicit input, events files, or default session mapping."""
    if task_choice is not None:
        # Explicit user input always wins.
        return task_choice

    # Prefer explicit discovery from existing events files for this subject/session.
    # This avoids relying on the default session-to-task mapping when the data already says otherwise.
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
                    # Extract the task name from each matching events filename.
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
    # Fall back to the legacy session-to-task convention when no file-based match exists.
    task_used = session_task_map.get(session_choice)
    if task_used is None:
        raise ValueError(
            f"Cannot infer task from session '{session_choice}'. Set task_choice explicitly."
        )
    return task_used


def build_events_file(base_root, subject_choice, session_choice, task_used):
    """Build path to NIRS events TSV for selected subject/session/task."""
    # This is the canonical BIDS-style path used throughout the module.
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
            title=f"sub-{subject_choice} ses-{session_choice} | {trial_type} | Series {idx + 1}",
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
                ax.set_title(f'Series {col_idx + 1}', fontsize=10)

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

    # Normalize the recording argument so downstream code can loop over a simple list.
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

    # Resolve the task first so the events file and the raw SNIRF file stay aligned.
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

    # Load all requested recordings once and reuse the raw objects for every trial type.
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

    # Ignore non-displayable trial labels such as MVC and occlusion markers.
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
            # Extract epochs once per device so the mean curve and envelope use the same data.
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

            # Choose how to visualize dispersion across repeated trials.
            if envelope_mode == 'std':
                std_curve = np.std(epochs, axis=0)
                lower = mean_curve - std_curve
                upper = mean_curve + std_curve
            else:
                lower = np.min(epochs, axis=0)
                upper = np.max(epochs, axis=0)

            # Restrict the integration to the requested time range before computing AUC.
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

    # Keep legacy variable names because downstream notebook cells may already reference them.
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

    # Plot the per-device comparison when requested.
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

    # Print the per-device summary statistics when requested.
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
    """Return the fixed Analysis_Results folder used by this project."""
    return DEFAULT_ANALYSIS_RESULTS_DIR


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

    # Walk through every available subject/session/task events file.
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
            # Read the events table and load the corresponding filtered Hb data.
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
                # Compute one value per intensity so the summary table stays compact.
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
    # Always write exports to the project Analysis_Results folder.
    output_dir = _default_output_dir_from_base_root(base_root)

    # Build the table first so we can skip file creation when there is no data.
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
        # Use a semicolon-separated CSV because the downstream workflow expects that format.
        results_df.to_csv(output_path, index=False, sep=';', decimal=',')
    except PermissionError:
        raise PermissionError(
            f"Cannot write CSV to {output_path}. "
            "Close the file if it is open and check write permissions for Analysis_Results."
        )

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

    # Cache the chromophore-specific channel lists once per call.
    hbr_chs = get_hbr_channels(raw_obj)
    hbo_chs = get_hbo_channels(raw_obj)

    for onset in onsets:
        # Crop the window once and then compute each derived metric from the same segment.
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
        # Keep the table rectangular even when data is missing for one condition.
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
            # Load the raw file directly so we can skip sessions where the SNIRF file is absent.
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
                    # Write placeholder rows so downstream stats can still align subjects.
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
                # Derive all chromophore summaries from the same set of event windows.
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


def _extract_intensity_from_trial_type(trial_type):
    """Extract intensity label from trial type, e.g., Con30 -> 30%."""
    trial_text = str(trial_type).strip()
    digits = ''.join(ch for ch in trial_text if ch.isdigit())
    if not digits:
        return 'NA'
    return f'{digits}%'


def _compute_trial_chromophore_metrics(
    raw_obj,
    onset,
    window_duration,
    chromophore,
    use_abs_envelope=True,
    use_abs_auc=True,
):
    """Compute mean, envelope metrics, and AUC for one trial window and one chromophore."""
    if window_duration is None:
        window_duration = DEFAULT_WINDOW_DURATION

    hbr_chs = get_hbr_channels(raw_obj)
    hbo_chs = get_hbo_channels(raw_obj)
    segment = raw_obj.copy().crop(tmin=onset, tmax=onset + window_duration)

    if chromophore == 'HbR':
        if not hbr_chs:
            return None
        data = segment.get_data(picks=hbr_chs) * 1e6
        mean_curve = np.mean(data, axis=0)
    elif chromophore == 'HbO':
        if not hbo_chs:
            return None
        data = segment.get_data(picks=hbo_chs) * 1e6
        mean_curve = np.mean(data, axis=0)
    elif chromophore in {'HbDiff', 'HbTot'}:
        if not hbo_chs or not hbr_chs:
            return None
        data_hbo = segment.get_data(picks=hbo_chs) * 1e6
        data_hbr = segment.get_data(picks=hbr_chs) * 1e6
        mean_hbo_curve = np.mean(data_hbo, axis=0)
        mean_hbr_curve = np.mean(data_hbr, axis=0)
        if chromophore == 'HbDiff':
            mean_curve = mean_hbo_curve - mean_hbr_curve
        else:
            mean_curve = mean_hbo_curve + mean_hbr_curve
    else:
        raise ValueError("chromophore must be one of: HbR, HbO, HbDiff, HbTot")

    if mean_curve.size == 0:
        return None

    envelope_signal = np.abs(mean_curve) if use_abs_envelope else mean_curve
    envelope = np.abs(hilbert(envelope_signal))

    auc_signal = np.abs(mean_curve) if use_abs_auc else mean_curve
    times = segment.times
    if auc_signal.size >= 2 and times.size >= 2:
        auc_value = _integrate_auc(auc_signal, times)
    else:
        auc_value = np.nan

    return {
        'mean_uM': float(np.mean(mean_curve)),
        'envelope_peak_uM': float(np.max(envelope)),
        'envelope_mean_uM': float(np.mean(envelope)),
        'auc_uM_s': float(auc_value) if np.isfinite(auc_value) else np.nan,
    }


def build_hb_trial_table(
    subject_ids=None,
    base_root=None,
    recording='portalite',
    window_duration=None,
    include_all_participants=False,
    use_abs_envelope=True,
    use_abs_auc=True,
    print_progress=True,
):
    """Build one row per trial with Hb means, envelope, and AUC for one device."""
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT

    rows = []

    for subject_id, session_id, task_name, events_file in _iter_subject_session_task_events(
        base_root,
        subject_ids=subject_ids,
    ):
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
                continue

            events_df = pd.read_csv(events_file, sep='\t')
            raw_hb_filt = load_filtered_hb(
                subject=subject_id,
                session=session_id,
                task=task_name,
                recording=recording,
                base_root=base_root,
            )

            event_rows = events_df.copy()
            event_rows = event_rows[event_rows['trial_type'].map(_is_displayable_trial_type)]

            if event_rows.empty:
                if include_all_participants:
                    rows.append(
                        {
                            'Subject': subject_id,
                            'Session': session_id,
                            'Task': task_name,
                            'Contraction_Mode': task_name.capitalize(),
                            'Recording': recording,
                            'Trial_Type': 'NA',
                            'Trial_Index': np.nan,
                            'Onset_s': np.nan,
                            'Intensity': 'NA',
                            'Mean_HbR_uM': np.nan,
                            'Mean_HbO_uM': np.nan,
                            'Mean_HbDiff_uM': np.nan,
                            'Mean_HbTot_uM': np.nan,
                            'EnvelopePeak_HbR_uM': np.nan,
                            'EnvelopePeak_HbO_uM': np.nan,
                            'EnvelopePeak_HbDiff_uM': np.nan,
                            'EnvelopePeak_HbTot_uM': np.nan,
                            'EnvelopeMean_HbR_uM': np.nan,
                            'EnvelopeMean_HbO_uM': np.nan,
                            'EnvelopeMean_HbDiff_uM': np.nan,
                            'EnvelopeMean_HbTot_uM': np.nan,
                            'AUC_HbR_uM_s': np.nan,
                            'AUC_HbO_uM_s': np.nan,
                            'AUC_HbDiff_uM_s': np.nan,
                            'AUC_HbTot_uM_s': np.nan,
                        }
                    )
                continue

            trial_counters = {}
            for _, event_row in event_rows.iterrows():
                trial_type = str(event_row['trial_type']).strip()
                onset = float(event_row['onset'])
                trial_index = trial_counters.get(trial_type, 0) + 1
                trial_counters[trial_type] = trial_index

                hbr_metrics = _compute_trial_chromophore_metrics(
                    raw_hb_filt,
                    onset,
                    window_duration,
                    'HbR',
                    use_abs_envelope=use_abs_envelope,
                    use_abs_auc=use_abs_auc,
                )
                hbo_metrics = _compute_trial_chromophore_metrics(
                    raw_hb_filt,
                    onset,
                    window_duration,
                    'HbO',
                    use_abs_envelope=use_abs_envelope,
                    use_abs_auc=use_abs_auc,
                )
                hbdiff_metrics = _compute_trial_chromophore_metrics(
                    raw_hb_filt,
                    onset,
                    window_duration,
                    'HbDiff',
                    use_abs_envelope=use_abs_envelope,
                    use_abs_auc=use_abs_auc,
                )
                hbtot_metrics = _compute_trial_chromophore_metrics(
                    raw_hb_filt,
                    onset,
                    window_duration,
                    'HbTot',
                    use_abs_envelope=use_abs_envelope,
                    use_abs_auc=use_abs_auc,
                )

                rows.append(
                    {
                        'Subject': subject_id,
                        'Session': session_id,
                        'Task': task_name,
                        'Contraction_Mode': task_name.capitalize(),
                        'Recording': recording,
                        'Trial_Type': trial_type,
                        'Trial_Index': trial_index,
                        'Onset_s': onset,
                        'Intensity': _extract_intensity_from_trial_type(trial_type),
                        'Mean_HbR_uM': np.nan if hbr_metrics is None else hbr_metrics['mean_uM'],
                        'Mean_HbO_uM': np.nan if hbo_metrics is None else hbo_metrics['mean_uM'],
                        'Mean_HbDiff_uM': np.nan if hbdiff_metrics is None else hbdiff_metrics['mean_uM'],
                        'Mean_HbTot_uM': np.nan if hbtot_metrics is None else hbtot_metrics['mean_uM'],
                        'EnvelopePeak_HbR_uM': np.nan if hbr_metrics is None else hbr_metrics['envelope_peak_uM'],
                        'EnvelopePeak_HbO_uM': np.nan if hbo_metrics is None else hbo_metrics['envelope_peak_uM'],
                        'EnvelopePeak_HbDiff_uM': np.nan if hbdiff_metrics is None else hbdiff_metrics['envelope_peak_uM'],
                        'EnvelopePeak_HbTot_uM': np.nan if hbtot_metrics is None else hbtot_metrics['envelope_peak_uM'],
                        'EnvelopeMean_HbR_uM': np.nan if hbr_metrics is None else hbr_metrics['envelope_mean_uM'],
                        'EnvelopeMean_HbO_uM': np.nan if hbo_metrics is None else hbo_metrics['envelope_mean_uM'],
                        'EnvelopeMean_HbDiff_uM': np.nan if hbdiff_metrics is None else hbdiff_metrics['envelope_mean_uM'],
                        'EnvelopeMean_HbTot_uM': np.nan if hbtot_metrics is None else hbtot_metrics['envelope_mean_uM'],
                        'AUC_HbR_uM_s': np.nan if hbr_metrics is None else hbr_metrics['auc_uM_s'],
                        'AUC_HbO_uM_s': np.nan if hbo_metrics is None else hbo_metrics['auc_uM_s'],
                        'AUC_HbDiff_uM_s': np.nan if hbdiff_metrics is None else hbdiff_metrics['auc_uM_s'],
                        'AUC_HbTot_uM_s': np.nan if hbtot_metrics is None else hbtot_metrics['auc_uM_s'],
                    }
                )

        except Exception as e:
            if print_progress:
                print(f"  Error for sub-{subject_id} ses-{session_id} task-{task_name}: {e}")

    return pd.DataFrame(rows)


def export_hb_summary_csv_by_device(
    subject_ids=None,
    base_root=None,
    recordings=None,
    window_duration=None,
    output_dir=None,
    filename_prefix='hb_summary',
    include_all_participants=False,
    table_granularity='trial',
    use_abs_envelope=True,
    use_abs_auc=True,
    print_progress=True,
):
    """Export one CSV per device in summary or per-trial granularity.

    table_granularity:
        - 'trial' (default): one row per trial with mean, envelope, and AUC metrics.
        - 'summary': one row per intensity with mean chromophore values.
    """
    if base_root is None:
        base_root = DEFAULT_BASE_ROOT
    # Always write exports to the project Analysis_Results folder.
    output_dir = _default_output_dir_from_base_root(base_root)
    if recordings is None:
        recordings = ['portalite', 'semaxone']

    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for recording in recordings:
        # Build one table per device so device-specific trends remain separated.
        if table_granularity == 'trial':
            df = build_hb_trial_table(
                subject_ids=subject_ids,
                base_root=base_root,
                recording=recording,
                window_duration=window_duration,
                include_all_participants=include_all_participants,
                use_abs_envelope=use_abs_envelope,
                use_abs_auc=use_abs_auc,
                print_progress=print_progress,
            )
        elif table_granularity == 'summary':
            df = build_hb_summary_table(
                subject_ids=subject_ids,
                base_root=base_root,
                recording=recording,
                window_duration=window_duration,
                include_all_participants=include_all_participants,
                print_progress=print_progress,
            )
        else:
            raise ValueError("table_granularity must be 'trial' or 'summary'.")

        output_path = None
        if not df.empty:
            output_path = os.path.join(output_dir, f'{filename_prefix}_{recording}.csv')
            try:
                df.to_csv(output_path, index=False, sep=';', decimal=',')
            except PermissionError:
                raise PermissionError(
                    f"Cannot write CSV to {output_path}. "
                    "Close the file if it is open and check write permissions for Analysis_Results."
                )

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
            fmt='o',
            marker='o',
            markersize=9,
            linestyle='None',
            color='purple',
            label='Semaxone',
            elinewidth=2,
            capsize=7,
        )
        ax1.errorbar(
            x,
            [mean_hbr_con30_port, mean_hbr_con50_port],
            yerr=[std_hbr_con30_port, std_hbr_con50_port],
            fmt='s',
            marker='s',
            markersize=9,
            linestyle='None',
            color='green',
            label='Portalite',
            elinewidth=2,
            capsize=7,
        )
        ax1.set_xlabel('Intensity', fontsize=13, fontweight='bold')
        ax1.set_ylabel('HHb Concentration (uM)', fontsize=13, fontweight='bold')
        ax1.set_title('HHb Group Mean - Concentric (all subjects)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['30%', '50%'])
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax2.errorbar(
            x,
            [mean_hbr_ecc30_sem, mean_hbr_ecc50_sem],
            yerr=[std_hbr_ecc30_sem, std_hbr_ecc50_sem],
            fmt='o',
            marker='o',
            markersize=9,
            linestyle='None',
            color='purple',
            label='Semaxone',
            elinewidth=2,
            capsize=7,
        )
        ax2.errorbar(
            x,
            [mean_hbr_ecc30_port, mean_hbr_ecc50_port],
            yerr=[std_hbr_ecc30_port, std_hbr_ecc50_port],
            fmt='s',
            marker='s',
            markersize=9,
            linestyle='None',
            color='green',
            label='Portalite',
            elinewidth=2,
            capsize=7,
        )
        ax2.set_xlabel('Intensity', fontsize=13, fontweight='bold')
        ax2.set_ylabel('HHb Concentration (uM)', fontsize=13, fontweight='bold')
        ax2.set_title('HHb Group Mean - Eccentric (all subjects)', fontsize=14, fontweight='bold')
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
        print('DESCRIPTIVE GROUP STATISTICS - HHb mean across subjects')
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

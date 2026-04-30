import polars as pl
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from typing import Iterable, Tuple, Sequence
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def pipeline_spectra_GUI(df, sigma=None, multiplier=None, freq_col='freq', cols=None):
    df = pl.from_pandas(df)
    df_clean, detection_limits = apply_detection_limits(df, sigma_list=sigma, detection_mult=multiplier)
    peak_dict = detect_peaks(df_clean) #gets freq of each peak above noise
    peak_array = peaks_dict_to_arrays(peak_dict) # N arrays of [freq, int] pairs
    all_peaks = combine_unique_freqs(peak_dict)
    df_peaks = get_int_at_peaks_AIopt(all_peaks, df_clean, return_df=True) #using df_signals so freq is 0 if peak is bloew signal
    df_peaks = unique_by_freq_keep_max3(df_peaks, "freq", cols, tol=0.05)
    df_peaks = df_peaks.sort("freq")

    df = df.to_pandas()
    df_clean = df_clean.to_pandas()
    df_peaks = df_peaks.to_pandas()


    return df, df_clean, df_peaks, detection_limits, peak_array 


def pipeline_sigma_GUI(df):
    df = pl.from_pandas(df)
    df = set_baseline_at_zero(df) #computes median and sets base line (median) at 0 (median in noise is very very similar)
    noise = only_noise(df, 1) #noise region over 5x mean (mean always positive, 0 and negative gives errors)
    sigma_list = compute_sigma(noise) #computes sigma (std) from noise region

    return sigma_list

def apply_detection_limits(df: pl.DataFrame, sigma_list: list = None, detection_mult=3) -> pl.DataFrame:
    """
    Zero out values below detection_mult*sigma for all columns except the first one.
    
    Parameters:
        df: Polars DataFrame, first column is freq, remaining are intensities.
        sigma_dict: optional dictionary of column-specific sigma values.
                    If None, uses median of each column.
    
    Returns:
        Polars DataFrame with noise removed.
    """
    detection_limits = []
    new_cols = []
    for i, col in enumerate(df.columns[1:]):
        # Determine sigma
        if sigma_list:

            sigma = sigma_list[i]
            detection_limits.append(detection_mult*sigma)

        else:
            print("No sigma values found, computing standard deviation...")
            sigma = df[col].std()
            detection_limits.append(detection_mult*sigma)

        #print(f"{col} detection limits = {detection_mult*sigma}")
        # Apply noise removal
        new_col = pl.when(pl.col(col) < detection_mult*sigma).then(0).otherwise(pl.col(col)).alias(col)
        new_cols.append(new_col)

    # Return new DataFrame with first column unchanged and list of sigmas for each col
    return df.select([df.columns[0]] + new_cols), detection_limits


def detect_peaks(df: pl.DataFrame, freq_col: str = "freq", prominence: float = 0.0):

    # ensure Polars
    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df)

    freq = df.get_column(freq_col).to_numpy()

    peak_dict = {}

    for col in df.columns:
        if col == freq_col:
            continue

        signal = df.get_column(col).to_numpy()

        peaks, _ = find_peaks(signal, prominence=prominence)

        peak_dict[col] = {
            "peak_freq": freq[peaks],
            "peak_int": signal[peaks]
        }

    return peak_dict

def peaks_dict_to_arrays(peak_dict):
    """
    Convert a nested dictionary of peaks into 2D NumPy arrays (freq, intensity) per dataset.

    Parameters:
        peak_dict: dict
            Format: {
                'dataset_name': {'peak_freq': np.array, 'peak_int': np.array},
                ...
            }

    Returns:
        dict of np.ndarray:
            Each value is a 2D array with columns [freq, intensity].
    """
    arrays = {}
    for key, val in peak_dict.items():
        arrays[key] = np.stack([val['peak_freq'], val['peak_int']], axis=1)
    return arrays

def set_baseline_at_zero(df: pl.DataFrame) -> pl.DataFrame:
    cols = df.columns[1:]
    df = df.with_columns([(pl.col(col) - pl.col(col).median()).alias(col) for col in cols])
    return df

def only_noise(df, mult):
    #Removes datapoint above mult times mean -> only noise region 
    cols = df.columns[1:]
    return df.with_columns([pl.when(abs(pl.col(col)) < (pl.col(col).mean() * mult)).then(pl.col(col)).otherwise(None).alias(col) for col in cols])

def compute_sigma(df: pl.DataFrame) -> list:
    # Always skip the first column
    cols = df.columns[1:]
    # Vectorized computation: filter peaks and compute std for each column
    sigma_df = df.select([
        pl.col(col).std()
        .alias(f"std_{col}")
        for col in cols
    ])
    # Return the std values as a Python list
    return list(sigma_df.row(0))

def concat_cols_on_freq(dfs: Iterable[pl.DataFrame], names) -> pl.DataFrame:
 
    new_dfs = []
    for i, (df, col_name) in enumerate(zip(dfs, names)):
        if i == 0:
            new_df = df.rename({df.columns[0]: 'freq',
                                df.columns[1]: col_name})
            new_dfs.append(new_df)
            continue
        new_df = df.drop(df.columns[0]).rename({df.columns[1]: col_name})
        new_dfs.append(new_df)

    df_concat = pl.concat(new_dfs, how="horizontal")

    return df_concat

def unique_by_freq_keep_max3(df, freq_col, cols, tol):

    i1 = cols[0]
    i2 = cols[1]

    df_binned = df.with_columns(
        ((pl.col(freq_col) / tol).round(0) * tol).alias("f_bin")
    )

    df_scored = df_binned.with_columns(
        pl.max_horizontal(i1, i2).alias("imax")
    )

    df_unique = (
        df_scored
        .sort("imax", descending=True)
        .unique(subset="f_bin", keep="first")
        .drop(["f_bin", "imax"])
    )

    return df_unique

    
def combine_unique_freqs(peak_dict):
    """
    Combine all peak frequencies from multiple datasets into a single array without duplicates.

    Parameters:
        peak_dict: dict
            Format: {'dataset_name': {'peak_freq': np.array, 'peak_int': np.array}, ...}

    Returns:
        np.ndarray: sorted unique frequencies
    """
    all_freqs = []
    for vals in peak_dict.values():
        all_freqs.append(vals['peak_freq'])
    
    # Concatenate all arrays
    combined = np.concatenate(all_freqs)
    
    # Keep only unique frequencies and sort
    unique_freqs = np.unique(combined)
    
    return unique_freqs



def get_int_at_peaks_AIopt(peak_freqs, df, return_df = False):
    """
    Given an array of peak frequencies and a Polars dataframe with
    the first column as frequencies and the rest as intensity columns,
    return a dictionary of intensities at the peak frequencies. Df return if preferred.

    Assumes the frequency column is sorted ascendingly.

    Notes: this  may not get the exact maximums, since peaks could be displaced by one grid point.
            Does not matter in terms of ratio for ML
    """
    freqs = df.select(df.columns[0]).to_numpy().ravel()
    intensities = {}

    # Precompute array of all intensities
    cols = df.columns[1:]
    data = df.select(cols).to_numpy()  # shape: (N, num_cols)

    # For each peak, find the nearest index once
    idxs = np.searchsorted(freqs, peak_freqs)
    idxs = np.clip(idxs, 0, len(freqs)-1)  # stay within bounds

    # Check left/right neighbor to get truly nearest
    left = np.maximum(idxs-1, 0)
    right = np.minimum(idxs, len(freqs)-1)
    nearest = np.where(
        np.abs(freqs[left] - peak_freqs) <= np.abs(freqs[right] - peak_freqs),
        left,
        right
    )

    # Extract intensities for each column
    for i, col in enumerate(cols):
        intensities[col] = data[nearest, i]
    
    if return_df is True:
        freq_col = pl.DataFrame({"freq": peak_freqs})
        intensity_df = pl.DataFrame(intensities)
        df_peaks = pl.concat([freq_col, intensity_df], how="horizontal")
        return df_peaks

    return intensities
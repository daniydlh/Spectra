import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path
from typing import Iterable, Tuple, Sequence
import plotly.express as px
import plotly.graph_objects as go

#######################

#####################
def make_ratio_ranges(
    rmin: float,
    rmax: float,
    step: float = 0.5,
    ):
    """
    Generate contiguous ratio ranges.

    Example
    -------
    (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), ...

    Returns
    -------
    ranges : list of (low, high)
    labels : list of str
    """

    edges = np.arange(rmin, rmax + step, step)

    ranges = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    labels = [f"{edges[i]:.1f} ≤ r < {edges[i+1]:.1f}" for i in range(len(edges) - 1)]

    return ranges, labels
#########################

def plot_xy_by_ratio_ranges(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    ratio_col: str,
    ratio_ranges: Sequence[Tuple[float | None, float | None]],
    range_labels: Sequence[str] | None = None,
    bins: int | None = None,
):
    """
    Plot x vs y, colored by ranges of a ratio column.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    x_col, y_col : str
        Columns for x and y axes.
    ratio_col : str
        Column used to define the ranges.
    ratio_ranges : sequence of (low, high)
        Range boundaries. Use None for open-ended.
        Example: [(None, 0.5), (0.5, 1.0), (1.0, None)]
    range_labels : sequence of str, optional
        Labels for legend. Must match ratio_ranges length.
    bins : int or None
        None → scatter plot
        int → 2D histogram with nbins = bins

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    # --- validation ---
    for col in (x_col, y_col, ratio_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    if range_labels is not None and len(range_labels) != len(ratio_ranges):
        raise ValueError("range_labels must match ratio_ranges length")

    # Bright, distinct colors for each range
    colors = ["#e41a1c",  # bright red
              "#377eb8",  # bright blue
              "#4daf4a",  # bright green
              "#ff7f00"]  # bright orange

    fig = go.Figure()

    # --- plotting ---
    for i, (low, high) in enumerate(ratio_ranges):
        if low is None:
            mask = df[ratio_col] < high
            label = f"ratio < {high}"
        elif high is None:
            mask = df[ratio_col] >= low
            label = f"ratio ≥ {low}"
        else:
            mask = (df[ratio_col] >= low) & (df[ratio_col] < high)
            label = f"{low} ≤ ratio < {high}"

        if range_labels is not None:
            label = range_labels[i]

        sub = df.filter(mask)
        if sub.height == 0:
            continue

        # only first three ranges get legend
        show_legend = True if i < 4 else False

        if bins is None:
            fig.add_trace(
                go.Scatter(
                    x=sub[x_col],
                    y=sub[y_col],
                    mode="markers",
                    name=label,
                    showlegend=show_legend,
                    marker=dict(size=5, opacity=0.85, color=colors[i]),
                )
            )
        else:
            fig.add_trace(
                go.Histogram2d(
                    x=sub[x_col],
                    y=sub[y_col],
                    nbinsx=bins,
                    nbinsy=bins,
                    name=label,
                    opacity=0.8,
                )
            )

    fig.update_layout(
        title=f"{x_col} vs {y_col} colored by {ratio_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    # Disable legend
    fig.update_layout(showlegend=True)
    fig.show()

    return fig

###########################

def plot_histogram_array(
    data: Iterable,
    nbins: int = 40,
    histnorm: str | None = None,
    title: str | None = None,
    xlabel: str = "Value",
    ):
    """
    Create a Plotly histogram from a 1-D array-like input.

    Parameters
    ----------
    data : array-like
        Input data (list, tuple, NumPy array).
    nbins : int, optional
        Number of histogram bins.
    histnorm : str or None, optional
        Normalization: None, 'percent', 'probability',
        'density', or 'probability density'.
    title : str or None, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Histogram figure.
    """

    data = np.asarray(data).ravel()

    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")

    # remove NaN and inf
    data = data[np.isfinite(data)]
    total = len(data)

    fig = go.Figure(
        go.Histogram(
            x=data,
            nbinsx=nbins,
            xbins=dict(
                start=0.0,
                size=0.5
            ),
            histnorm=histnorm,
            hovertemplate=(
                f"{xlabel}: %{{x}}<br>"
                "Count: %{y}<br>"
                "Percentage: %{percent:.2f}%<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or "Histogram",
        xaxis_title=xlabel,
        yaxis_title=histnorm if histnorm else "Count",
        bargap=0.05,
    )

    fig.show()
    return fig
########################

def how_much_decr_ref(
    df: pl.DataFrame,
    ref_col1: str,
    ref_col2: str,
    decr_percentage_to: float,
    tol_percentage: float
    ):

    #target = decr_percentage_to * pl.col(ref_col)
    #tol = tol_percentage * pl.col(ref_col).abs()

    df_filtered = df.filter(
        (((1 / pl.col(ref_col1)) - decr_percentage_to).abs() <= tol_percentage)
        &
        (((1 / pl.col(ref_col2)) - decr_percentage_to).abs() <= tol_percentage)
    )
        
    return df_filtered

#########################
def increase_or_decrease(df, tol_percentage):

    i1 = "int_so2"
    i2 = "int_water"
    i3 = "int_deu"

    col1 = "+H2O"
    col2 = "+D2O"

    df_new = df.with_columns([
        pl.when(pl.col("i1ispeak") & pl.col("i2ispeak"))
        .then(
            pl.when((((pl.col(i1) - pl.col(i2)).abs())/pl.col(i1)) <= tol_percentage ).then(1)
            .when(pl.col(i1) > pl.col(i2)).then(-2)
            .otherwise(+2)
        )
        .otherwise(0)
        .alias(col1),

        pl.when(pl.col("i1ispeak") & pl.col("i3ispeak"))
        .then(
            pl.when((((pl.col(i1) - pl.col(i2)).abs())/pl.col(i1)) <= tol_percentage ).then(1)
            .when(pl.col(i1) > pl.col(i3)).then(-2)
            .otherwise(+2)
        )
        .otherwise(0)
        .alias(col2),
    ])

    group_counts = (
    df_new.group_by([col1, col2])  # group by your comparison columns
      .agg([
          pl.count().alias("count")  # count how many rows in each group
      ])
      .sort([col1, col2])  # optional: sort for readability
    )

    return df_new, group_counts


#########################

def int_is_peak(df, dict_array, tol=1e-3):
    freq = df.columns[0]
    i1 = df.columns[1]
    i2 = df.columns[2]
    i3 = df.columns[3]

    # Collect reference frequencies
    ref_freqs = np.concatenate(
        [arr[:, 0] for arr in dict_array.values()]
    )

    # Frequency is a peak (FORCED Boolean)
    freq_is_peak = (
        pl.col(freq)
        .map_elements(
            lambda f: bool(np.any(np.isclose(f, ref_freqs, atol=tol)))
        )
        .cast(pl.Boolean)
    )

    df_ispeak = df.with_columns([
        (freq_is_peak & (pl.col(i1) != 0)).alias("i1ispeak"),
        (freq_is_peak & (pl.col(i2) != 0)).alias("i2ispeak"),
        (freq_is_peak & (pl.col(i3) != 0)).alias("i3ispeak"),
    ])

    total = df.height
    print(f"Total number of rows {total}")

    df_count = (
        df_ispeak
        .group_by(["i1ispeak", "i2ispeak", "i3ispeak"])
        .agg(pl.len().alias("n"))
        .with_columns(
            (pl.col("n") / total * 100).round(2).alias("%")
        )
        .sort("n", descending=True)
    )

    return df_ispeak, df_count

########################

 # Helper expression: frequency ∈ ref_freqs (within tol)
def freq_in_ref():
    return (
        pl.col(freq_col)
        .map_elements(
            lambda f: np.any(np.isclose(f, ref_freqs, atol=tol))
        )
    )
##############################

def groups_iszero(df):
    return {
        "TTT": df.filter(
            pl.col("i1es0") & pl.col("i2es0") & pl.col("i3es0")
        ),
        "FFF": df.filter(
            ~pl.col("i1es0") & ~pl.col("i2es0") & ~pl.col("i3es0")
        ),
        "TFF": df.filter(
            pl.col("i1es0") & ~pl.col("i2es0") & ~pl.col("i3es0")
        ),
        "FTF": df.filter(
            ~pl.col("i1es0") & pl.col("i2es0") & ~pl.col("i3es0")
        ),
        "FFT": df.filter(
            ~pl.col("i1es0") & ~pl.col("i2es0") & pl.col("i3es0")
        ),
        "TTF": df.filter(
            pl.col("i1es0") & pl.col("i2es0") & ~pl.col("i3es0")
        ),
        "TFT": df.filter(
            pl.col("i1es0") & ~pl.col("i2es0") & pl.col("i3es0")
        ),
        "FTT": df.filter(
            ~pl.col("i1es0") & pl.col("i2es0") & pl.col("i3es0")
        ),
    }

############################

def unique_by_freq_keep_max3(df, freq_col, i1, i2, i3, tol):

    
    df_binned = df.with_columns(
        ((pl.col(freq_col) / tol).round(0) * tol).alias("f_bin")
    )

    df_scored = df_binned.with_columns(
        pl.max_horizontal(i1, i2, i3).alias("imax")
    )

    df_unique = (
        df_scored
        .sort("imax", descending=True)
        .unique(subset="f_bin", keep="first")
        .drop(["f_bin", "imax"])
    )

    return df_unique

#############################

def groups_incr_decr(df: pl.DataFrame):
    h = pl.col("+H2O")
    d = pl.col("+D2O")

    return {
        "++": df.filter((h == 2) & (d == 2)),
        "+-": df.filter((h == 2) & (d == -2)),
        "+=": df.filter((h == 2) & (d == 1)),
        "+0": df.filter((h == 2) & (d == 0)),

        "-+": df.filter((h == -2) & (d == 2)),
        "--": df.filter((h == -2) & (d == -2)),
        "-=": df.filter((h == -2) & (d == 1)),
        "-0": df.filter((h == -2) & (d == 0)),

        "=+": df.filter((h == 1) & (d == 2)),
        "=-": df.filter((h == 1) & (d == -2)),
        "==": df.filter((h == 1) & (d == 1)),
        "=0": df.filter((h == 1) & (d == 0)),

        "0+": df.filter((h == 0) & (d == 2)),
        "0-": df.filter((h == 0) & (d == -2)),
        "0=": df.filter((h == 0) & (d == 1)),
        "00": df.filter((h == 0) & (d == 0)),
    }
#############################

def groups_ispeak(df):
    return {
        "TTT": df.filter(
            pl.col("i1ispeak") & pl.col("i2ispeak") & pl.col("i3ispeak")
        ),
        "FFF": df.filter(
            ~pl.col("i1ispeak") & ~pl.col("i2ispeak") & ~pl.col("i3ispeak")
        ),
        "TFF": df.filter(
            pl.col("i1ispeak") & ~pl.col("i2ispeak") & ~pl.col("i3ispeak")
        ),
        "FTF": df.filter(
            ~pl.col("i1ispeak") & pl.col("i2ispeak") & ~pl.col("i3ispeak")
        ),
        "FFT": df.filter(
            ~pl.col("i1ispeak") & ~pl.col("i2ispeak") & pl.col("i3ispeak")
        ),
        "TTF": df.filter(
            pl.col("i1ispeak") & pl.col("i2ispeak") & ~pl.col("i3ispeak")
        ),
        "TFT": df.filter(
            pl.col("i1ispeak") & ~pl.col("i2ispeak") & pl.col("i3ispeak")
        ),
        "FTT": df.filter(
            ~pl.col("i1ispeak") & pl.col("i2ispeak") & pl.col("i3ispeak")
        ),
    }

#############################

def count_nans(df, cols):
    for col in cols:
        nan_count = df.filter(pl.col(col).is_nan()).height
        print(f"{col}: {nan_count} NaNs")
    return

##############################

def int_is_0(df):
    i1 = df.columns[1]
    i2 = df.columns[2]
    i3 = df.columns[3]

    df_is0 = df.with_columns([
        (pl.col(i1) == 0).alias("i1es0"),
        (pl.col(i2) == 0).alias("i2es0"),
        (pl.col(i3) == 0).alias("i3es0"),
    ])

    total = df.height
    print(f"Total number of peaks {total}")

    df_count0 = (
        df_is0
        .group_by(["i1es0", "i2es0", "i3es0"])
        .agg(pl.len().alias("n"))
        .with_columns(
            (pl.col("n") / total * 100).round(2).alias("%")
        )
        .sort("n", descending=True)
    )

    return df_is0, df_count0

################################
def plot_base_peaks(output, x_base, y_base, x_peak, y_peak, sigma):

    fig = go.Figure()
    # Base spectrum (line)
    fig.add_trace(go.Scatter(
        x=x_base,
        y=y_base,
        mode="lines",
        name="base",
        opacity=1.0,
        line=dict(color="blue", width=1),
    ))

    fig.update_layout(
        xaxis_title='Frequency',
        yaxis_title='Intensity',
        shapes=[
            dict(
                type="line",
                x0=min(x_base),
                x1=max(x_base),
                y0=3*sigma,
                y1=3*sigma,
                line=dict(color="black", width=2, dash="dash"),
            )
        ]
    )

    # SO2 peaks
    fig.add_trace(go.Scatter(
        x=x_peak,
        y=y_peak,
        mode="markers",
        name="peaks",
        opacity=1.,
        marker=dict(
            color="red",
            symbol="circle",
            size=4,
        ),
    ))

    fig.write_html(output, include_plotlyjs="cdn")  # archivo interactivo
    fig.show()
    return

################################

def plot_2d_ratio_int(output, x, y):

    fig = go.Figure()
    # Base spectrum (line)
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        opacity=0.5,
        line=dict(color="blue", width=1),
    ))

    fig.update_layout(
        xaxis_title='Intensity 1',
        yaxis_title='Ratio 1/2',
    )

    fig.write_html(output, include_plotlyjs="cdn")  # archivo interactivo
    fig.show()
    return

################################

def plot_2d_int(output, x, y):

    fig = go.Figure()
    # Base spectrum (line)
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        opacity=0.5,
        line=dict(color="blue", width=1),
    ))

    fig.update_layout(
        xaxis_title='Intensity 1',
        yaxis_title='Intensity 2',
    )

    fig.write_html(output, include_plotlyjs="cdn")  # archivo interactivo
    fig.show()
    return

################################

def plot_3d(output, x, y, z, min=None, max=None):

    fig = px.scatter_3d(
    x=x.to_numpy(),
    y=y.to_numpy(),
    z=z.to_numpy(),
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Intensity 1",
            yaxis_title="Intensity 2",
            zaxis_title="Intensity 3",
        )
    )
    if min != None and max != None:
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[min, max]),
                yaxis=dict(range=[min, max]),
                zaxis=dict(range=[min, max]),
            )
        )

    fig.update_traces(
        marker=dict(
            size=3,        # point size
            opacity=0.8,   # transparency (0–1)
        )
    )

    fig.show()
    fig.write_html(
        output,
        include_plotlyjs="cdn",  # or "embed" for fully offline
        full_html=True
    )
    return

################################

def ratio_arc_cols(df, ratio=False, arctan2=False):

    i1 = df.columns[1]
    i2 = df.columns[2]
    i3 = df.columns[3]

    if ratio:
        df = df.with_columns([
            (pl.col(i1) / pl.col(i2)).alias(f"{i1}/{i2}"),
            (pl.col(i1) / pl.col(i3)).alias(f"{i1}/{i3}"),
            (pl.col(i2) / pl.col(i3)).alias(f"{i2}/{i3}")
        ])
    if arctan2:
        df = df.with_columns([
            (np.arctan2(pl.col(i1), pl.col(i2)).alias(f"arctan2_{i1}/{i2}")),
            (np.arctan2(pl.col(i1), pl.col(i3)).alias(f"arctan2_{i1}/{i3}")),
            (np.arctan2(pl.col(i2), pl.col(i3)).alias(f"arctan2_{i2}/{i3}"))
        ])

    return df

################################


def get_int_at_peaks_AIopt(peak_freqs, df, return_df = False):
    """
    Given an array of peak frequencies and a Polars dataframe with
    the first column as frequencies and the rest as intensity columns,
    return a dictionary of intensities at the peak frequencies. Df return if preferred.

    Assumes the frequency column is sorted ascending.

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
    
    if return_df == True:
        freq_col = pl.DataFrame({"freq": peak_freqs})
        intensity_df = pl.DataFrame(intensities)
        df_peaks = pl.concat([freq_col, intensity_df], how="horizontal")
        return df_peaks

    return intensities


##############################

def get_int_at_peaks(peak_freqs, df, return_df = False):

    """
    Notes: this  may not get the exact maximums, since peaks could be displaced by one grid point.
            Does not matter in terms of ratio for ML
    """
    intensities = {}
    freqs = df.select(df.columns[0]).to_numpy().ravel()

    for col in df.columns[1:]:
        
        col_array = df[col].to_numpy().ravel()
        values = []

        for pf in peak_freqs:
            diff = np.abs(freqs - pf)
            idx = int(np.argmin(diff))
            values.append(col_array[idx])
        
        intensities[col] = np.array(values)

        if return_df == True:
            freq_col = pl.DataFrame({"freq": peak_freqs})
            intensity_df = pl.DataFrame(intensities)
            df_peaks = pl.concat([freq_col, intensity_df], how="horizontal")
            return df_peaks

    return intensities

#################################

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


#################################
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

##################################

def detect_peaks(df, prominence=0.0):

    freq = df[:,0].to_numpy()
    peak_dict = {}

    for col in df.columns[1:]:
        int = df[col].to_numpy()
        peaks, props = find_peaks(int, prominence=prominence)
        peak_freq = freq[peaks]
        peak_int = int[peaks]

        peak_dict[col] = {
            "peak_freq": peak_freq,
            "peak_int": peak_int
        }
    return peak_dict


####################################

def noise_rm_all(df: pl.DataFrame, sigma_list: list = None) -> pl.DataFrame:
    """
    Zero out values below 3*sigma for all columns except the first one.
    
    Parameters:
        df: Polars DataFrame, first column is freq, remaining are intensities.
        sigma_dict: optional dictionary of column-specific sigma values.
                    If None, uses median of each column.
    
    Returns:
        Polars DataFrame with noise removed.
    """
    sigma3 = []
    new_cols = []
    for i, col in enumerate(df.columns[1:]):
        # Determine sigma
        if sigma_list:
            sigma = sigma_list[i]
            sigma3.append(3*sigma_list[i])

        else:
            sigma = df[col].median()
            sigma3.append(3*sigma)
        
        print(f"{col} 3sigma = {3*sigma}")
        
        # Apply noise removal
        new_col = pl.when(pl.col(col) < 3*sigma).then(0).otherwise(pl.col(col)).alias(col)
        new_cols.append(new_col)
    
    # Return new DataFrame with first column unchanged and list of sigmas for each col
    return df.select([df.columns[0]] + new_cols), sigma3

####################################

def fft_df(file_path, spectra: pl.DataFrame, sep=",", decimals=6):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    extra_lines = [
        "!Params",
        "!Apodization=Kaiser",
        "!Apodization Beta=6.0",
        "!Band (MHz)=2000.0,8000.0",
        "!Frequency Start (MHz)=2000.0",
        "!Frequency Stop (MHz)=8000.0",
        "!Gate Length (us)=20.0",
        "!Gate Start (us)=0.0",
        "!Normalization Reference Frequency (MHz)=2000.0",
        "!Normalization Scale=1.0",
        "!Normalization Slope (1/MHz)=0.000137",
        "!Zero Pad=2",
        "!Data",
        "!FFT",
    ]

    with open(file_path, "w") as f:
        # write header
        for line in extra_lines:
            f.write(line + "\n")

        # write dataframe with controlled float precision
        spectra.write_csv(
            f,
            separator=sep,
            float_precision=decimals
        )
########################

def fft_arr(
    file_path,
    x: np.ndarray,
    y: np.ndarray,
    sep: str = ",",
    x_name: str = "frequency",
    y_name: str = "intensity",
):
    """
    Write FFT data to file from two NumPy arrays.

    Parameters
    ----------
    file_path : str or Path
        Output file path.
    x, y : np.ndarray
        1D NumPy arrays of equal length.
    sep : str
        Column separator for CSV output.
    x_name, y_name : str
        Column names.
    """

    if x.shape != y.shape:
        raise ValueError("x and y arrays must have the same shape")

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    spectra = pl.DataFrame({
        x_name: x,
        y_name: y,
    })

    extra_lines = [
        "!Params",
        "!Apodization=Kaiser",
        "!Apodization Beta=6.0",
        "!Band (MHz)=2000.0,8000.0",
        "!Frequency Start (MHz)=2000.0",
        "!Frequency Stop (MHz)=8000.0",
        "!Gate Length (us)=20.0",
        "!Gate Start (us)=0.0",
        "!Normalization Reference Frequency (MHz)=2000.0",
        "!Normalization Scale=1.0",
        "!Normalization Slope (1/MHz)=0.000137",
        "!Zero Pad=2",
        "!Data",
        "!FFT",
    ]

    with open(file_path, "w") as f:
        for line in extra_lines:
            f.write(line + "\n")
        spectra.write_csv(f, separator=sep)

#################################################


def get_noise(spectra, noise_regions):

    std_regions = np.zeros((len(noise_regions)))

    for i, (x_min, x_max) in enumerate(noise_regions):
        noise_region = spectra.filter(pl.col("freq").is_between(x_min, x_max)).drop("freq").to_numpy()
        std_regions[i] = noise_region.std(ddof=1)

    return std_regions.mean()

#############################

def matching_peaks_pair(freq1, int1, freq2, int2, tol):

    print("Threhshold of coincidence: ",tol)
    matched_freqs = []
    matched_ints_1 = []
    matched_ints_2 = []

    for f1, i1 in zip(freq1, int1):
        # Buscar picos en spectrum deu cercanos a f1 
        diffs = np.abs(freq2 - f1)
        idx = np.argmin(diffs)
        if diffs[idx] <= tol:
            matched_freqs.append(f1)
            matched_ints_1.append(i1)
            matched_ints_2.append(int2[idx])

    # Convertir a arrays
    matched_freqs = np.array(matched_freqs)
    matched_ints1 = np.array(matched_ints_1)
    matched_ints2 = np.array(matched_ints_2)

    return matched_freqs, matched_ints1, matched_ints2

########################################

def matching_peaks_three_arrays(freq1, int1, freq2, int2, freq3, int3, tol):

    print("Threhshold of coincidence: ",tol)
    matched_freqs = []
    matched_ints_1 = []
    matched_ints_2 = []
    matched_ints_3 = []


    for f1, i1 in zip(freq1, int1):
        # Buscar picos en spectrum deu cercanos a f1 
        diffs2 = np.abs(freq2 - f1)
        diffs3 = np.abs(freq3 - f1)
        idx2 = np.argmin(diffs2)
        idx3 = np.argmin(diffs3)
        if diffs2[idx2] <= tol and diffs3[idx3] <= tol:
            matched_freqs.append(f1)
            matched_ints_1.append(i1)
            matched_ints_2.append(int2[idx2])
            matched_ints_3.append(int3[idx3])
           
    # Convertir a arrays
    matched_freqs = np.array(matched_freqs)
    matched_ints1 = np.array(matched_ints_1)
    matched_ints2 = np.array(matched_ints_2)
    matched_ints3 = np.array(matched_ints_3)

    return matched_freqs, matched_ints1, matched_ints2, matched_ints3

######################################

def multi_gaussian(x, *params):
    """
    Sum of N Gaussians.
    params: [A1, mu1, sigma1, A2, mu2, sigma2, ..., AN, muN, sigmaN]
    """
    n_gauss = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n_gauss):
        A = params[3*i]
        mu = params[3*i+1]
        sigma = params[3*i+2]
        y += A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y

#################################################################

def auto_fit_multi_gaussian(x, y, height=None, distance=None, width_guess=0.5):
    """
    Automatically fit multiple Gaussians to a spectrum.

    Parameters
    ----------
    x : array-like
        Independent variable (wavelength, pixels).
    y : array-like
        Spectrum flux.
    height : float or None
        Minimum height for peak detection (passed to find_peaks).
    distance : float or None
        Minimum distance between peaks (passed to find_peaks).
    width_guess : float
        Initial guess for sigma for all peaks.

    Returns
    -------
    popt : array
        Optimized Gaussian parameters [A1, mu1, sigma1, ...].
    pcov : 2D array
        Covariance matrix of fit parameters.
    y_fit : array
        Reconstructed fitted spectrum.
    n_peaks : int
        Number of detected peaks.
    """
    # 2a. Detect peaks
    peaks, props = find_peaks(y, height=height, distance=distance)
    n_peaks = len(peaks)

    if n_peaks == 0:
        raise ValueError("No peaks detected. Adjust 'height' or 'distance' parameters.")

    # 2b. Initial guesses: amplitude = peak height, mu = peak position
    A_guess = y[peaks]
    mu_guess = x[peaks]
    sigma_guess = np.full(n_peaks, width_guess)

    # Flatten initial guesses for curve_fit
    p0 = []
    for A, mu, sigma in zip(A_guess, mu_guess, sigma_guess):
        p0 += [A, mu, sigma]

    # Optional: set bounds (sigma>0)
    lower_bounds = []
    upper_bounds = []
    for _ in range(n_peaks):
        lower_bounds += [0, x.min(), 0]  # A>0, mu>=min(x), sigma>0
        upper_bounds += [np.inf, x.max(), np.inf]

    # 2c. Fit
    popt, pcov = curve_fit(multi_gaussian, x, y, p0=p0, bounds=(lower_bounds, upper_bounds))

    # 2d. Reconstruct fitted spectrum
    y_fit = multi_gaussian(x, *popt)

    return popt, pcov, y_fit, n_peaks

################################

def symlog(x, linthresh=1e-3):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log10(1 + np.abs(x) / linthresh)

################################

def concat_cols_on_freq(dfs: Iterable[pl.DataFrame], names: Iterable[str]) -> pl.DataFrame:
    dfs = list(dfs)
    names = list(names)
    new_dfs = []
    for i, (df, name) in enumerate(zip(dfs, names)):
        if i == 0:
            new_df = df.rename({df.columns[1]: f"int_{name}"})
            new_dfs.append(new_df)
            continue
        new_df = df.drop(df.columns[0]).rename({df.columns[1]: f"int_{name}"})
        new_dfs.append(new_df)

    df_concat = pl.concat(new_dfs, how="horizontal")

    return df_concat

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path


def fft_df(file_path, spectra, sep):

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    extra_lines = [
        '!Params',
        '!Apodization=Kaiser',
        '!Apodization Beta=6.0',
        '!Band (MHz)=2000.0,8000.0',
        '!Frequency Start (MHz)=2000.0',
        '!Frequency Stop (MHz)=8000.0',
        '!Gate Length (us)=20.0',
        '!Gate Start (us)=0.0',
        '!Normalization Reference Frequency (MHz)=2000.0',
        '!Normalization Scale=1.0',
        '!Normalization Slope (1/MHz)=0.000137',
        '!Zero Pad=2',
        '!Data',
        '!FFT',
        ]

    with open(file_path, "w") as f:
        for line in extra_lines:
            f.write(line + "\n")
        spectra.write_csv(f, separator=sep)

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

def matching_peaks_three(freq1, int1, freq2, int2, freq3, int3, tol):

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
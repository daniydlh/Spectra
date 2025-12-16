import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from utils import get_noise, fft_file, auto_fit_multi_gaussian, multi_gaussian
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


spectra_water = pl.read_csv("2025-10-16-SO2-W_2200k.fft", 
                        has_header=True, 
                        skip_rows=14)

spectra_deu = pl.read_csv("2025-10-16-SO2-D-W_2000k.fft", 
                        has_header=True, 
                        skip_rows=14)


only_noise_regions = [[2210,2250],[2660,2672],[2814,2830]] #input for function
sigma_water = get_noise(spectra_water, only_noise_regions)
sigma_deu = get_noise(spectra_deu, only_noise_regions)
sigma = 3e-6


signals_water = spectra_water.with_columns(
                pl.when(pl.col('intensity') < 3*sigma_water)
                .then(0)
                .otherwise(pl.col('intensity'))
                .alias('intensity')
)

signals_deu = spectra_deu.with_columns(
                pl.when(pl.col('intensity') < 3*sigma_water)
                .then(0)
                .otherwise(pl.col('intensity'))
                .alias('intensity')
)

negative_water = spectra_water.filter(pl.col('intensity') < 3 * sigma_water)
negative_deu = spectra_deu.filter(pl.col('intensity') < 3 * sigma_deu)

fft_file('signals_water.fft', signals_water, sep=',')

x = signals_water['freq'].to_numpy()
y = signals_water['intensity'].to_numpy()






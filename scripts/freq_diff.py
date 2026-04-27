from data_analysis import sep
import polars as pl
import numpy as np


molecule = 'so2'
freq = np.loadtxt(f"data/{molecule}/lines_decreased_with_h2o.csv", usecols=0, skiprows=1, delimiter=",")
fitted_so2_dimer = np.loadtxt("lines/so2-dimer/fitted_lines_so2_dimer.csv")
fitted_so2_w = np.loadtxt("lines/so2-w/fitted_lines_so2_w.csv")
len(fitted_so2_w)
mask1 = np.any(np.isclose(freq[:, None], fitted_so2_dimer, atol=0.05), axis=1)
freq1 = freq[~mask1]
len(freq1)
mask2 = np.any(np.isclose(freq1[:, None], fitted_so2_w, atol=0.1), axis=1)
len(mask2)
freq2 = freq1[~mask2]
len(freq2)

diffs_matrix = []
for f in freq1:
    diffs = freq1 - f
    diffs_matrix.append(abs(diffs))

titles = [str(x) for x in freq1]   # convert numbers to strings
df = pl.DataFrame(diffs_matrix)
df.columns = titles


df.write_csv(f'data/{molecule}/freq_diffs_decrease_so2_h2o_nonfitted.csv', include_header=True, separator=";", quote_style="never")

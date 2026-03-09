from data_analysis import sep
import polars as pl
import numpy as np


molecule = 'so2'
freq1 = np.loadtxt(f"data/{molecule}/lines_decreased_with_h2o.csv", usecols=0, skiprows=1, delimiter=",")
diffs_matrix = []
for f in freq1:
    diffs = freq1 - f
    diffs_matrix.append(abs(diffs))

titles = [str(x) for x in freq1]   # convert numbers to strings
df = pl.DataFrame(diffs_matrix)
df.columns = titles

df.write_csv(f'data/{molecule}/freq_diffs_decrease_so2_h2o.csv', include_header=True, separator=";", quote_style="never")

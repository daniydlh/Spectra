import polars as pl
import numpy as np


valores = np.concatenate([np.random.randint(1, 10, 15), [50, 60, 80]])
s = pl.Series("asimetrica", valores)
s
m = s.median()
m
a = s-m
a
a.median()

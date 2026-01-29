# --- Interactive plot ---
set terminal qt enhanced font 'Times,12'

set datafile separator ","
set title ""
set xlabel "Intensity (SO2+H2O)"
set xlabel font "Times,25"
set ylabel "Intensity (SO2+D2O)"
set ylabel font "Times,25"
unset key
unset grid
unset xtics
unset ytics
unset border
set border 3   # 1=bottom, 2=left, 4=top, 8=right
set xr[0:0.06]
set yr[0:0.035]


# Plot
p '../csv/df_signals.csv' using 3:4 w p pt 7 ps 0.8 lc rgb "blue"



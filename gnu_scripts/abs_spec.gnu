# --- Interactive plot ---
set terminal qt enhanced font 'Times,12'

set datafile separator ","
set title ""
set xlabel "Frequency"
set xlabel font "Times,25"
set ylabel "Intensity"
set ylabel font "Times,25"
unset grid
set xrange[7223:7228]
set yrange[0:0.0045]
set key font "Times, 30"
unset xtics
unset ytics
unset border
set border 3   # 1=bottom, 2=left, 4=top, 8=right

# Plot
plot '../data/2025-10-19-SO2_2300k.csv' using 1:2 w l lw 2 linecolor rgb "blue" title "SO2"\
, '../data/2025-10-16-SO2-W_2200k.csv' using 1:2 w l lw 2 linecolor rgb "red" title "SO2 + H2O"\
, '../data/2025-10-16-SO2-D-W_2000k.csv' using 1:2 w l lw 2 linecolor rgb "green" title "SO2 + D2O"

# --- Save to PNG ---
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'slope_WvsD.png'

# Replot the same data
replot

# Close output


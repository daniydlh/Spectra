# --- Interactive plot ---
set terminal qt enhanced font 'Arial,12'

set datafile separator ","
set title "ratio D2O/H2O"
set xlabel "freq"
set ylabel "ratio D/H"
set grid

# Plot
plot 'peaks_water+deu.csv' using 2:4 with points pointtype 7 pointsize 0.5 linecolor rgb "red" title "HvsD"

pause -1  # Wait until you close the interactive window

# --- Save to PNG ---
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'ratio_D/H.png'

# Replot the same data
replot

# Close output
set output
~                

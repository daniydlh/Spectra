# --- Interactive plot ---
set terminal qt enhanced font 'Arial,12'

set datafile separator ","
set title "slope H2O vs D2O"
set xlabel "I water"
set ylabel "I deu"
set grid

# Plot
plot 'peaks_water+deu.csv' using 2:3 with points pointtype 7 pointsize 0.5 linecolor rgb "red" title "HvsD"

pause -1  # Wait until you close the interactive window

# --- Save to PNG ---
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'slope_WvsD.png'

# Replot the same data
replot

# Close output
set output



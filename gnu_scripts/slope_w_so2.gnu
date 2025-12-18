# --- Interactive plot ---
set terminal qt enhanced font 'Arial,12'

set datafile separator ","
set title "slope H2O vs SO2"
set xlabel "I water"
set ylabel "I deu"
set grid

# Plot
plot '../peaks_water+so2.csv' using 2:3 with points pointtype 7 pointsize 0.5 linecolor rgb "red"

pause -1  # Wait until you close the interactive window

# --- Save to PNG ---
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'slope_WvsSO2.png'

# Replot the same data
replot

# Close output
set output



# --- Interactive plot ---
set terminal qt enhanced font 'Arial,12'

set datafile separator ","
set title "Rotational Spectra"
set xlabel "Frequency"
set ylabel "Intensity"
set grid
set key left top

# Plot
plot 'peaks_water+deu.csv' using 1:2 with points pointtype 7 pointsize 1.5 linecolor rgb "red" title "H2O", \
     'peaks_water+deu.csv' using 1:3 with lines linewidth 2 linecolor rgb "blue" title "D2O"

pause -1  # Wait until you close the interactive window

# --- Save to PNG ---
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'spectra_h2o+d2o.png'

# Replot the same data
replot

# Close output
set output


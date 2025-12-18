# --- Interactive plot ---
set terminal qt enhanced font 'Arial,12'

set datafile separator ","
set title "Rotational Spectra"
set xlabel "Frequency"
set ylabel "Intensity"
set grid
#set key left top

# Plot
plot '../Spectra_signals/signals_so2.fft' using 1:2 w l lw 0.5 linecolor rgb "blue", \
     '../Spectra_signals/max_so2.fft' using 1:2 w p ps 0.5 pointtype 7 linecolor rgb "red" title "SO2", \
     '../Spectra_signals/signals_water.fft' using 1:(-$2) w l lw 0.5 linecolor rgb "red", \
     '../Spectra_signals/max_water.fft' using 1:(-$2) with points ps 0.5 pointtype 7 linecolor rgb "orange" title "H2O"

pause -1  # Wait until you close the interactive window

# --- Save to PNG ---
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'spectra_h2o+d2o.png'

# Replot the same data
replot

# Close output
set output


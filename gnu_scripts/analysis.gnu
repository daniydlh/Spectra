# --- Interactive plot ---
set terminal qt enhanced font 'Arial,12'

set datafile separator ","
set title "Rotational Spectra"
set xlabel "Frequency"
set ylabel "Intensity"
set grid
set key left top

# Plot
plot '../2025-10-14-SO2_3000kavg.fft' using 1:2 w l  linecolor rgb "blue" title "SO2", \
     'peaks_water+deu.csv' using 1:3 with lines linewidth 2i linecolor rgb "blue"
     'peaks_water+deu.csv' using 1:3 with lines linewidth 2 linecolor rgb "blue" title "D2O"


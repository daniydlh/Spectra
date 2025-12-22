# --- Interactive plot ---
set terminal qt enhanced font 'Arial,12'

set datafile separator ","
set title "slope H2O vs D2O"
set xlabel "I water"
set ylabel "I deu"
set grid

# Plot
plot 'peaks_water+deu.csv' using 2:3 with points pointtype 7 pointsize 0.5 linecolor rgb "red" title "HvsD"





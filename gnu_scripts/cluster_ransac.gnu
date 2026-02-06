#!/usr/bin/gnuplot

# ===========================================================================
# Script Gnuplot para archivos model_df_X.X_X.X_cluster_N.dat
# Formato:
#   population ranking N
#   slope VALUE
#   intercept VALUE
#   r2 VALUE
#   arctan VALUE
#   n_points VALUE
#   x y
#   datos...
# ===========================================================================

reset

# ---------------------------------------------------------------------------
# CONFIGURACIÓN - AJUSTAR SEGÚN TUS DATOS
# ---------------------------------------------------------------------------

# Prefijo de los archivos
# Definir directorio y nombre por separado
data_dir = "/Users/daniydlh/OneDrive-UVa/PhD-Projects/Spectra/models/RANSAC/model_df_0.0_inf"
model_name = "model_df_0.0_inf"

# Número de clusters
n_clusters = 10  # AJUSTAR según cuántos clusters tengas

# ---------------------------------------------------------------------------
# FUNCIONES PARA LEER METADATA
# ---------------------------------------------------------------------------

cluster_file(n) = sprintf("%s/%s_cluster_%d.dat", data_dir, model_name, n)

get_slope(n) = real(system(sprintf("awk 'NR==2 {print $2}' %s/%s_cluster_%d.dat", data_dir, model_name, n)))

get_intercept(n) = real(system(sprintf("awk 'NR==3 {print $2}' %s/%s_cluster_%d.dat", data_dir, model_name, n)))

get_r2(n) = real(system(sprintf("awk 'NR==4 {print $2}' %s/%s_cluster_%d.dat", data_dir, model_name, n)))

get_npoints(n) = int(system(sprintf("awk 'NR==6 {print $2}' %s/%s_cluster_%d.dat", data_dir, model_name, n)))

# ---------------------------------------------------------------------------
# PALETA DE COLORES VIVOS
# ---------------------------------------------------------------------------

array colors[15] = [ \
    "#E63946", "#F77F00", "#06D6A0", "#118AB2", "#8338EC", \
    "#FF006E", "#FFBE0B", "#06FFA5", "#4CC9F0", "#F72585", \
    "#7209B7", "#3A86FF", "#FB5607", "#FF006E", "#8338EC" \
]

# ---------------------------------------------------------------------------
# CALCULAR RANGOS PARA REGRESIONES
# ---------------------------------------------------------------------------

array xmins[n_clusters]
array xmaxs[n_clusters]
array slopes[n_clusters]
array intercepts[n_clusters]

# Leer metadata y calcular rangos
do for [i=0:n_clusters-1] {
    # Leer slope e intercept
    slopes[i+1] = get_slope(i)
    intercepts[i+1] = get_intercept(i)
    
    # Calcular rangos x (saltando las 7 primeras líneas de header)
    stats cluster_file(i) every ::7 using 1 nooutput
    xmins[i+1] = STATS_min
    xmaxs[i+1] = STATS_max
}

# ---------------------------------------------------------------------------
# DEFINIR FUNCIONES DE REGRESIÓN
# ---------------------------------------------------------------------------

do for [i=0:n_clusters-1] {
    eval sprintf("f%d(x) = slopes[%d] * x + intercepts[%d]", i, i+1, i+1)
}

# ---------------------------------------------------------------------------
# TERMINAL INTERACTIVO
# ---------------------------------------------------------------------------

set terminal qt size 1400,1000 enhanced font 'Arial,12' persist
# Alternativas:
# set terminal wxt size 1400,1000 enhanced font 'Arial,12' persist
# set terminal x11 size 1400,1000 enhanced font 'Arial,12' persist

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DEL PLOT
# ---------------------------------------------------------------------------

set title sprintf("RANSAC Clustering: %s", model_name) font "Arial,18" enhanced
set xlabel "Intensidad Water (x)" font "Arial,14"
set ylabel "Intensidad Deuterio (y)" font "Arial,14"

# Grid elegante
set grid xtics ytics mxtics mytics \
    lt -1 lc rgb "#CCCCCC" lw 1, \
    lt -1 lc rgb "#E5E5E5" lw 0.5

# Leyenda
set key outside right top vertical font "Arial,10" spacing 1.2 box

# Bordes y tics
set border lw 1.5
set tics nomirror out scale 0.75
set xtics font "Arial,11"
set ytics font "Arial,11"

# Auto-escala
set autoscale

# ---------------------------------------------------------------------------
# PLOT: PUNTOS + REGRESIONES
# ---------------------------------------------------------------------------

# Generar label con info del cluster
cluster_label(i) = sprintf("C%d (n=%d, R²=%.3f)", i, get_npoints(i), get_r2(i))

plot \
    for [i=0:n_clusters-1] cluster_file(i) every ::7 using 1:2 with points \
        pt 7 ps 0.5 lc rgb colors[(i % 15) + 1] \
        title cluster_label(i), \
    for [i=0:n_clusters-1] '+' using \
        (x_val = xmins[i+1] - 0.1*(xmaxs[i+1]-xmins[i+1]) + \
         $0/100.0 * 1.2*(xmaxs[i+1]-xmins[i+1]), x_val):\
        (value(sprintf("f%d(x_val)", i))) \
        with lines lw 2.5 lc rgb colors[(i % 15) + 1] notitle

print ""
print "===================================================================="
print "  VISUALIZACIÓN INTERACTIVA"
print sprintf("  Modelo: %s", model_name)
print sprintf("  Clusters: %d", n_clusters)
print ""
print "  CONTROLES:"
print "  - Click izquierdo + arrastrar: Hacer zoom en área"
print "  - Click derecho: Deshacer zoom / volver a vista completa"
print "  - Rueda del ratón: Zoom in/out"
print "  - Tecla 'h': Ayuda de comandos"
print ""
print "  Presiona Enter para exportar PDF..."
print "===================================================================="
print ""

pause -1

# ---------------------------------------------------------------------------
# EXPORTAR PDF
# ---------------------------------------------------------------------------

set terminal pdfcairo size 14,10 enhanced color font 'Arial,14' linewidth 3 pointscale 1.5
set output sprintf('%s_clusters.pdf', model_name)

set title sprintf("RANSAC Clustering: %s", model_name) font "Arial,20" enhanced
replot

set output

print ""
print "===================================================================="
print sprintf("  ✓ PDF generado: %s_clusters.pdf", model_name)
print "===================================================================="
print ""

# Volver a terminal interactivo
set terminal qt size 1400,1000 enhanced font 'Arial,12' persist
replot

pause -1 "Presiona Enter para salir..."

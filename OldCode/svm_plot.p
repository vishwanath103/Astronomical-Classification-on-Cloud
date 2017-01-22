set terminal png
set output "svm_plot.png"
#set size 1,0.75
#set autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
#set xtic auto                          # set xtics automatically
#set ytic auto  

set xrange[0:260]
set yrange[0:1000]
set border linewidth 1

set title "Time taken vs Size of training data"
set xlabel "Size of training data(rows/1000)"
set ylabel "Time(s)"
set grid
set key outside
plot "svm_plot.dat" using 1:3  with linespoints title 'Distributed' linestyle 2, "svm_plot.dat" using 1:2  with linespoints title 'Single' linestyle 72
#pause 20

set terminal png
set output "training_knn.png"
#set size 1,0.75
#set autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
#set xtic auto                          # set xtics automatically
#set ytic auto  

set xrange[15:40]
set yrange[0:12]
set border linewidth 1

set title "Time taken vs Size of training data for kNN"
set xlabel "Size of training data(rows/100000)"
set ylabel "Time(s)"
set grid
set key outside
plot "training_knn.dat" using 1:3  with linespoints title 'Distributed' linestyle 2, "training_knn.dat" using 1:2  with linespoints title 'Single' linestyle 72
#pause 20

set terminal png
set output "testing_knn.png"
#set size 1,0.75
#set autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
#set xtic auto                          # set xtics automatically
#set ytic auto  

set xrange[0:10]
set yrange[0:50]
set border linewidth 1

set title "Time taken vs Size of testing data for kNN"
set xlabel "Size of testing data(rows/10000)"
set ylabel "Time(s)"
set grid
set key outside
plot "testing_knn.dat" using 1:2  with linespoints title 'Distributed ' linestyle 2, "testing_knn.dat" using 1:3  with linespoints title 'Single' linestyle 72
#pause 20

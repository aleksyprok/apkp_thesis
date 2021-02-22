This Python 3 code solves the cold ideal linearised MHD equations using spectral and method of lines techniques.
For more information on how the code works, see Manual/manual.pdf.

control.py defines the variables, e.g. the background Alfven speed, the angle of the background magnetic field to the z-axis (alpha) and the number of harmonics to use (N).

eigens.py calculates the eigenfrequencies and eigenfunctions and the following integrals I1, I2, ..., I7 (see manual.pdf).

main.py is the main routine which solves the MHD equations.

line_along_x.py and line_along_z.py are similar to main.py except they also output the graphs needed for the thesis's associated figures.

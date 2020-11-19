import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import sympy as sp

# Objective of the program:
# -Perform least squares regression on i 2D data points by approximating x the vector of parameters of the function f.

# Mathematical explanation and definition of objects:
# -x is projected onto C(A) the column space of A to find the approximate solution ^x to Ax=b, where bi = f(X) for a given X
# and the row vector Ai contains the sum terms of the function of unknown coefficients f. 
# -All columns of A must be independant, if not the equation is rewritten as Â*C*^x=b where Â contains no dependance.

#type of function, get: # of columns in A (and size of x)
fntype = input("Enter a function type: ")
if fntype == "linear":
    print("yes linear")
    AcolN = 2
    #the first line in x is the coefficient, the second the offset

#read from input file containing data points, get: 
# # of datapoints (AlineN), values used to elaborate A (xvals), and b
AlineN = 0

X = np.empty
Y = np.empty
Xvals = np.array([])
b = np.array([])

#open file and pass through each line, which corresponds to a data point 
allpts = open('input.txt')
for ptsline in allpts:
    print(ptsline)

    #append x value of point
    if Xvals.size == 0:
        Xvals = np.array( [ int(ptsline.partition(";")[0].strip() ) ] )
    else:
        X = np.array( [ int(ptsline.partition(";")[0].strip() ) ] ) 
        Xvals = np.vstack( (Xvals, X) )

    #append y value of point to b
    if b.size == 0:
        b = np.array( [ int(ptsline.partition(";")[2].strip() ) ] )
    else:
        Y = np.array( [ int(ptsline.partition(";")[2].strip() ) ] ) 
        b = np.vstack( (b, Y) )

    AlineN = AlineN+1
allpts.close()

#determine framing of plot and axis markings, not adapted to general case
Xmin = np.amin(Xvals)
Xmax = np.amax(Xvals)
Xpts = (Xmax-Xmin)*5
Ymin = 0
Ymax = 0

print("----")
print("X and Y(b vector) values:")
print(Xvals)
print("---")
print(b)
print("----")

#elaborate the A matrix
if fntype == "linear":
    Aright = np.full((AlineN, 1), 1, dtype=int)
    A = np.concatenate( (Xvals, Aright), axis=1)

print("A matrix:")
print(A)
print("----")

#ensure ATA is invertible: find Â and C such that Â's columns are independant and Â*C = A
    #skip if all columns of A are independant
#find the reduced row echelon form of A


#iterate through all columns to find and register dependancy in C



#Obtain ^x by projecting onto C(A)
AT = np.transpose(A)
print(AT)
ATA = np.dot(AT, A)
print(ATA)
invATA = la.inv(ATA)
print(invATA)
ATb = np.dot(AT, b)
print(ATb)
solution = np.dot(invATA, ATb)
print(solution)

#plot points
plt.scatter(Xvals, b)

#plot function
#make the last value be rounded from the highest point x value, idem first (?)
xplotvals = np.linspace(Xmin, Xmax, Xpts)

if fntype == "linear":
    yplotvals = solution[0]*xplotvals+solution[1]

plt.plot(xplotvals, yplotvals)
#plt.draw() ??
plt.show()
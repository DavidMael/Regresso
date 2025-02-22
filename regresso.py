import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import sympy as sp

# Objective of the program:
# -Read i 2D data points from input.txt, assumed not to contain outliers, and the type of regression to perform on them
# -Perform least squares regression on this data by approximating s the vector of parameters of the function f.
# -Plot the data points and the obtained function

# Mathematical explanation and definition of objects:
# -s is projected onto C(A) the column space of A to find the approximate solution ŝ to As=b, where bi = f(t) for a given t
# and the row vector Ai contains the sum terms of the function of unknown coefficients f. 
# -All columns of A must be independant, if not the equation is rewritten as ÂCŝ=b where Â has full column rank.

#-----Taking input-----

#type of function, get: # of columns in A (and size of s)
fntype = input("Enter a function type: ")
if fntype == "linear":
    print("yes linear")
    AcolN = 2
    #the first line in s is the coefficient, the second the offset
elif fntype == "quadratic":
    print("yes quadratic")
    AcolN = 3
    #the first line in s is a, the second b, the third c in ax^2+bx+c

#read from input file containing data points, get: 
# # of datapoints (AlineN), X,Y values used to elaborate A (xvals), b
AlineN = 0
Xl=[]
bl=[]

#open file and pass through each line, which corresponds to a data point 
allpts = open('input.txt')
for ptsline in allpts:
    print(ptsline)
    #print(AlineN)

    #append x value of point
    Xl.append( int(ptsline.partition(";")[0].strip() ) )

    #append y value of point to b
    bl.append( int(ptsline.partition(";")[2].strip() ) )

    AlineN = AlineN+1
allpts.close()

Xvals = (sp.Matrix(Xl))
b = (sp.Matrix(bl))
#Xvals and b are now column vectors

print("----")
print("X values and b vector:")
print(Xvals)
print("---")
print(b)
print("----")

#-----Preparing Aŝ=b and performing Regression-----

#elaborate the A matrix, Xvals is not needed beyond this point and can be altered
if fntype == "linear":
    A = sp.ones(AlineN,1)
    A = A.col_insert(0, Xvals)
elif fntype == "quadratic":
    A = sp.ones(AlineN,1)
    A = A.col_insert(0, Xvals)
    for i in range(AlineN):
        Xvals[i, 0] = Xvals[i, 0]**2
    A = A.col_insert(0, Xvals)

print("A matrix:")
print(A)
print(AcolN)
print(AlineN)
print("----")

#ensure ATA is invertible: find Â and C such that Â's columns are independant and ÂC = A
    #skip if all columns of A are independant
#find the reduced row echelon form of A in R[0], R[1] containing the indices of pivot columns
R=A.rref()
print("RREF of A:")
print(R)
print("----")

#determine if A has full column rank, if not then assign C and A=Â
#This should only happen in certain specific cases, such as the Aŝ=b being underdetermined, some cases of the input containing 
# many repeated points, or functions that give proportional values over the enire set of input x values such as two unit steps
pivnum = len(R[1])
if pivnum < AcolN:
    C = R[0][0:pivnum, :]
    print("Coeff matrix C:")
    print(C)
    print("-----")

    #Delete dependant columns of A,
    #working backwards through columns of A and R[1] pivot indices to preserve meaning of indices in R[1]
    pivcount=pivnum-1
    for i in range (AcolN):
        #print("numbers")
        #print(i)
        #print(pivcount)
        if (AcolN-i-1) == R[1][pivcount]:
            pivcount -= 1
        else:
            A.col_del( (AcolN-i-1) )
    print("Full column rank matrix Â:")
    print(A)
    print("-----")

#Obtain ŝ by projecting onto C(A)
AT = A.T
print(AT)
ATA = AT*A
print(ATA)
invATA = ATA**-1
print(invATA)
ATb = AT*b
print(ATb)
solution = invATA*ATb
print(solution)

#if A was reduced to Â, copy ŝ as the intermediate solution ŷ and extend C and y to Ĉ and ŷ to solve Ĉŝ=ŷ
if pivnum<AcolN:

    sizextend = AcolN-pivnum

    #extend to the size of ŝ the intermediate solution vector ŷ with 0s if the system is underdetermined 
    #and with 1s if A isn't full column rank for any other reson
    if (AcolN>AlineN):
        yextend = np.zeros((sizextend, 1))
    else:
        yextend = np.ones((sizextend, 1))

    y=np.array(solution).astype(np.float64)
    y=np.concatenate((y, yextend ), axis=0)
    print("ŷ vector:")
    print(y)
    print("-----")

    #make Ĉŝ = ŷ solvable by extending Ĉ with rows of 1 1 and n 0s such that it is square and full column rank.
    print(C)
    C=np.array(C).astype(np.float64)
    print(C)
    for i in range (sizextend):
        z=np.zeros((1, AcolN))
        z[0,i]=1
        C=np.concatenate((C, z ), axis=0)

    print("Ĉ matrix:")
    print(C)
    print("-----")

    #solve Ĉŝ=ŷ for final solution
    solution = np.linalg.solve(C, y)

print("Function parameters")
for i in range( AcolN ):
    print(solution[i])

#-----Plotting results-----

#determine framing of plot and number of points to plot, not adapted to general case
Xmin = min(Xl)
Xmax = max(Xl)
Xpts = (Xmax-Xmin)*AlineN*10

#plot points
plt.scatter(Xl, b, label="Input data")

#plot function
xplotvals = np.linspace(Xmin, Xmax, Xpts)

if fntype == "linear":
    yplotvals = solution[0]*xplotvals+solution[1]
elif fntype == "quadratic":
    yplotvals = solution[0]*(xplotvals**2) + solution[1]*xplotvals + solution[2]

plt.plot(xplotvals, yplotvals, color='r', label="Regression curve")

plt.suptitle("%s regression of input data" % fntype.capitalize() )
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
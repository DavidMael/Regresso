import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import sympy as sp

# Objective of the program:
# -Perform least squares regression on i 2D data points by approximating x the vector of parameters of the function f.

# Mathematical explanation and definition of objects:
# -x is projected onto C(A) the column space of A to find the approximate solution ^x to Ax=b, where bi = f(t) for a given t
# and the row vector Ai contains the sum terms of the function of unknown coefficients f. 
# -All columns of A must be independant, if not the equation is rewritten as Â*C*^x=b where Â has full column rank.

#type of function, get: # of columns in A (and size of x)
fntype = input("Enter a function type: ")
if fntype == "linear":
    print("yes linear")
    AcolN = 2
    #the first line in x is the coefficient, the second the offset
elif fntype == "quadratic":
    print("yes quadratic")
    AcolN = 3
    #the first line in x is a, the second b, the third c in ax^2_bx+c

#read from input file containing data points, get: 
# # of datapoints (AlineN), values used to elaborate A (xvals), and b
AlineN = 0

X=0
Y=0
#Xvals = sp.zeros(1,1)
#b = sp.zeros(1,1)
Xl=[]
bl=[]

#open file and pass through each line, which corresponds to a data point 
allpts = open('input.txt')
for ptsline in allpts:
    print(ptsline)
    #print(AlineN)

    #append x value of point
    X = int(ptsline.partition(";")[0].strip() )
    Xl.append(X)

    #append y value of point to b
    Y = int(ptsline.partition(";")[2].strip() )
    bl.append(Y)

    AlineN = AlineN+1
allpts.close()

Xvals = (sp.Matrix(Xl))
b = (sp.Matrix(bl))
#Xvals and b are now column vectors

#determine framing of plot and number of points to plot, not adapted to general case
Xmin = min(Xl)
Xmax = max(Xl)
Xpts = (Xmax-Xmin)*5
Ymin = 0
Ymax = 0

print("----")
print("X and Y(b vector) values:")
print(Xvals)
print("---")
print(b)
print("----")

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

#ensure ATA is invertible: find Â and C such that Â's columns are independant and Â*C = A
    #skip if all columns of A are independant
#find the reduced row echelon form of A in R[0], R[1] containing the indices of pivot columns
R=A.rref()
print("RREF of A:")
print(R)
print("----")

#determine if A has full column rank, if not then assign C and A=Â
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
    print("full column rank matrix Â:")
    print(A)
    print("-----")


#Obtain ^x by projecting onto C(A)
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

#if A was reduced to Â, extend C and the incomplete solution s to Ĉ and ^s to find the solution ^x
if pivnum<AcolN:
    sizextend = AcolN-pivnum
    print(sizextend)

    y=np.array(solution).astype(np.float64)
    y=np.concatenate((y, np.ones((sizextend, 1)) ), axis=0)
    print("ŷ vector:")
    print(y)
    print("-----")

    print(C)
    C=np.array(C).astype(np.float64)
    print(C)
    for i in range (sizextend):
        z=np.zeros((1, AcolN))
        print(z)
        z[0,i]=1
        print(i)
        print(z)
        C=np.concatenate((C, z ), axis=0)

    print("Ĉ matrix:")
    print(C)
    print("-----")

    #solve Ĉ*^x=ŷ for final solution
    solution = np.linalg.solve(C, y)

#plot points
plt.scatter(Xl, b)

#plot function
#make the last value be rounded from the highest point x value, idem first
xplotvals = np.linspace(Xmin, Xmax, Xpts)

print("function parameters")
for i in range( AcolN ):
    print(solution[i])

if fntype == "linear":
    yplotvals = solution[0]*xplotvals+solution[1]
elif fntype == "quadratic":
    yplotvals = solution[0]*(xplotvals**2) + solution[1]*xplotvals + solution[2]

plt.plot(xplotvals, yplotvals)
#plt.draw() ??
plt.show()
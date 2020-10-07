import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

#fit a curve to input data (2D) by "solving" the equation Ax=b to find the function parameters in x

#type of function: # of columns in A (and size of x)
fntype = input("Enter a function type: ")
if fntype == "linear":
    print("yes linear")
    AcolN = 2
    #the first line in x is the coefficient, the second the offset

#read from input file containing data points: # of datapoints and b vector (yvals)
AlineN = 0

X = np.empty
Y = np.empty
Xvals = np.array([])
Yvals = np.array([])

allpts = open('input.txt')
for ptsline in allpts:
    print(ptsline)

    if Xvals.size == 0:
        Xvals = np.array( [ int(ptsline.partition(";")[0].strip() ) ] )
    else:
        X = np.array( [ int(ptsline.partition(";")[0].strip() ) ] ) 
        Xvals = np.vstack( (Xvals, X) )


    if Yvals.size == 0:
        Yvals = np.array( [ int(ptsline.partition(";")[2].strip() ) ] )
    else:
        Y = np.array( [ int(ptsline.partition(";")[2].strip() ) ] ) 
        Yvals = np.vstack( (Yvals, Y) )

    AlineN = AlineN+1
allpts.close()

Xmin = np.amin(Xvals)
Xmax = np.amax(Xvals)
Xpts = (Xmax-Xmin)*5
Ymin = 0
Ymax = 0

print("----")
print("X and Y(b vector) values:")
print(Xvals)
print("---")
print(Yvals)
print("----")

#elaborate equation terms A matrix

if fntype == "linear":
    Aright = np.full((AlineN, 1), 1, dtype=int)
    A = np.concatenate( (Xvals, Aright), axis=1)

print("A matrix:")
print(A)
print("----")

#"solve equation"

AT = np.transpose(A)
print(AT)
ATA = np.dot(AT, A)
print(ATA)
invATA = la.inv(ATA)
print(invATA)

ATb = np.dot(AT, Yvals)
print(ATb)
solution = np.dot(invATA, ATb)
print(solution)

#plot points
plt.scatter(Xvals, Yvals)

#plot function
#make the last value be rounded from the highest point x value, idem first
xplotvals = np.linspace(Xmin, Xmax, Xpts)

if fntype == "linear":
    yplotvals = solution[0]*xplotvals+solution[1]

plt.plot(xplotvals, yplotvals)
#plt.draw() ??
plt.show()
# part a)

import numpy as np
A=np.identity(5)
print(A)

# part b)

A[:,1]=3
print(A)


# part c)
# sum of elements of matrix
sum_mat=0
for i in range(0,5):
    for j in range(0,5):
        sum_mat=sum_mat+A[i,j]

print("The sum of all elements of matrix is : ",sum_mat)



#part d)

A=np.transpose(A)
print(A)


#part e)


# diagnol sum
sum_diag=0

for i in range(0,5):

    sum_diag=sum_diag+A[i,i]

print("The sum of the diagnol elements are : ",sum_diag)
# 3rd row sum

row3=0

for i in range(0,5):
    row3=row3+A[2,i]

print("The sum of row 3 elements are : ",row3)

# part f
print()
std_dev=np.sqrt(3)
B=np.random.normal(5,std_dev,(5,5))
print(B)

# part g)


C=np.ones((2,5))


C[0,]=B[0,]*B[1,]
C[1,]=B[2,]+B[3,] - B[4,]
print(C)

#part h)


D = np.multiply(C,[2,3,4,5,6])

print(D)

#part i)

#print("\n COV MAT     ")
X=np.array([2,4,6,8]).T
Y=np.array([6,5,4,3]).T
Z=np.array([1,3,5,7]).T


a=np.matrix([X,Y,Z])


print(np.cov(a))


# part j)

x = np.array([2,4,6,8,10,12,14,16,18,20]).T

x_bar=np.mean(x)
print(" mean of x is x_bar : ", x_bar)
x_mean_square=pow(x_bar,2)
print("square of mean of x x_mean_square :", x_mean_square)
x_square=np.square(x)
print("squares of elements in x is x_square : ", x_square)
x_square_mean=np.mean(x_square)
print("Mean of squares of x is x_square_mean : ", x_square_mean)

stddev_x=np.std(x)
print("standard deviation stddev_x is : ",stddev_x)
stddev_square=pow(stddev_x,2)
print("square of standard deviation is stddev_square : ", stddev_square)


RHS=(x_mean_square+stddev_square)

print(RHS, x_square_mean)
if(x_square_mean==(RHS)):
    print("True - equation verified. both sides answer is : ", RHS)
else:
    print("False (equation not verified)")
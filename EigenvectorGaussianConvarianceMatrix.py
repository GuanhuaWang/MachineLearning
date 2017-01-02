import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import pylab

mu_1 = 4
variance_1 = 4
sigma_1 = math.sqrt(variance_1)
X_1 = np.random.normal(mu_1, sigma_1, 100)
print "the mean of X_1 is %s" %(np.mean(X_1))

mu_2 = 3
variance_2 = 9
sigma_2 = math.sqrt(variance_2)
X_2 = X_1/2 + np.random.normal(mu_2, sigma_2, 100)
print "the mean of X_2 is %s" %(np.mean(X_2))

cov_matrix=np.cov(X_1,X_2)
print "the covarience matrix is %s" %(cov_matrix)
val, vec = np.linalg.eig(cov_matrix)
print "the eigenvalues are %s" %(val)
print "the eigenvectors are %s" %(vec)

plt.figure(1)
pylab.ylim([-15,15])
pylab.xlim([-15,15])
plt.scatter(X_1,X_2)


mu = (np.mean(X_1),np.mean(X_2))
Vec_A = [np.mean(X_1),np.mean(X_2),vec[0][0], vec[0][1]]
Vec_B = [np.mean(X_1),np.mean(X_2),vec[1][0], vec[1][1]]

soa =np.array([Vec_A])
X,Y,U,V = zip(*soa)
ax = plt.gca()
ax.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1/val[0])
soa2 =np.array([Vec_B])
X1,Y1,U1,V1 = zip(*soa2)
ax1 = plt.gca()
ax1.quiver(X1,Y1,U1,V1,angles='xy',scale_units='xy',scale=1/val[1])

plt.figure(2)
vec[[0,1]] = vec[[1,0]]
tran_vec=vec.transpose()
X=np.column_stack((X_1-np.mean(X_1),X_2-np.mean(X_2)))
New_X = np.dot(X,tran_vec)
pylab.ylim([-15,15])
pylab.xlim([-15,15])
plt.scatter(New_X[:,0],New_X[:,1])

plt.show()
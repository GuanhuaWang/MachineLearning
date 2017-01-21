#latent factor model for joke recommendation.

import scipy.io
import scipy as sp
import numpy as np
import csv
import matplotlib.pyplot as plt

def pca(data,dim):
    u,s,v = np.linalg.svd(train)
    low_vec = np.zeros((dim,100))
    for i in range (dim):
        low_vec[i]=v[i]
    low_vec1 = low_vec.T
    return np.dot(data,low_vec1),low_vec
    
def matrix_fac(r,p,q,sigma,steps):
    for step in range(steps):
        for i in range (r.shape[0]):
            for j in range (r.shape[1]):
                if r[i][j]!=0:
                    eij = r[i][j]-np.dot(p[i,:],q[:,j])
                    for k in range (p.shape[1]):
                        p[i][k]=p[i][k]+0.001*(eij*q[k][j]-sigma*p[i][k])
                        q[k][j]=q[k][j]+0.001*(eij*p[i][k]-sigma*q[k][j])
    
    return np.dot(p,q)
             
def mse(predict,train):    
    mse =0
    for i in range(train.shape[0]):
        for j in range(train.shape[1]):
            mse += (predict[i][j]-train[i][j])**2         
    print mse
    
def recommend(low_data,low_vec,que):
    recom = np.zeros((que.shape[0],2))
    que[:,1] = que[:,1]-1
    que[:,2] = que[:,2]-1 
    recom[:,0]=que[:,0]
    for i in range (que.shape[0]):
        if np.dot(low_data[que[i][1],:],low_vec[:,que[i][2]])>0:
            recom[i][1]=1
        else:
            recom[i][1]=0
    np.savetxt("pred.csv",recom,fmt='%d',delimiter=",")
    
if __name__ == "__main__":
    mat = scipy.io.loadmat('./joke_data/joke_train.mat')
    Nan_train = mat['train']
    train = np.nan_to_num(Nan_train)
#===============recommendation===============#
    low_data,low_vec = pca(train,5)
    que = np.genfromtxt('./joke_data/query.txt',delimiter=',')
    recommend(low_data,low_vec,que)
    
    
#=========latent factor with zeros===========#
    low_data,low_vec = pca(train,20)
    predict = np.dot(low_data,low_vec)
#=========gradient descent===============#
    dim = 20
    p = np.random.rand(train.shape[0],dim)
    q = np.random.rand(dim,train.shape[1])
    predict = matrix_fac(train,p,q,0.02,10)
#=========validation=============#
    val = np.genfromtxt('./joke_data/validation.txt',delimiter=',')
    val[:,0] = val[:,0]-1
    val[:,1] = val[:,1]-1
    hit =0
    for i in range (val.shape[0]):
        if predict[val[i][0]][val[i][1]]>0:
            pre=1
            if pre == val[i][2]:
                hit += 1
        elif predict[val[i][0]][val[i][1]]<=0:
            pre = 0
            if pre == val[i][2]:
                hit += 1
    acc=float(hit)/val.shape[0]
    print "the accuracy is %s (%s / %d)" %(acc,hit,val.shape[0])
    mse(predict,train)


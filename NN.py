from mnist import MNIST
import numpy as np
import scipy
import math
import random

NUM_CLASSES = 10

def data_std(x):
    return ((x.T - np.mean(x, axis=1)) / np.std(x, axis=1)).T

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    # Data Normalization
    X_train1 = data_std(X_train)
    X_test1 = data_std(X_test)
    return X_train1, labels_train, X_test1, labels_test

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.eye(NUM_CLASSES)[labels_train]

def add_bias(data):
    a,b = data.shape
    x = np.ones((a,(b+1)))
    x[:,:-1]=data
    return x
    
def shuffle(train,label):
    c = zip(train,label)
    random.shuffle(c)
    a = [e[0] for e in c]
    b = [e[1] for e in c]
    return a,b

def split(train,label):
    train1 = np.asarray([train[i] for i in range(0,50000)])
    train2 = np.asarray([train[i] for i in range(50000,60000)])
    label1 = np.asarray([label[i] for i in range(0,50000)])
    label2 = np.asarray([label[i] for i in range(50000,60000)])
    return train1,train2,label1,label2
def accu(a,b):
    b = a + b/a*10
    acc = float(a)/b
    print "The predict accuracy is %s " %acc
    
def NeuralNet(nin,nhid,nout,train,label,epsilon,iteration,decay,batch_size):
    #v n_hid * (n_in+1) matrix
    v = np.random.random((nhid,(nin+1)))-1  
    #w n_out * (n_hid+1) matrix
    w = np.random.random((nout,(nhid+1)))-1
    a,b = train.shape
    loss_array= np.zeros((1,1))
    for i in range (0,iteration):
        data_index = np.random.choice(a,size = batch_size, replace=True,p=None)
        X_batch = np.take(train,data_index,axis=0)
        Y_batch = np.take(label,data_index,axis=0)
        #forward propagation
        hin = np.dot(v,X_batch.T)
        h2 = np.maximum(hin,0.0)
        h3 = h2.reshape((batch_size,nhid))
        hout = add_bias(h3)
        hout = hout.T
        yin = np.dot(w,hout)#z2
        y1 = np.argmax(yin)
        #print yin.shape
        #print y1
        #yout = np.exp(yin-yin[y1]) / np.sum(np.exp(yin-yin[y1]), axis=0)
        yout = np.exp(yin) / np.sum(np.exp(yin), axis=0)
        yout = yout.T
        loss = np.sum(-np.log(yout[np.arange(batch_size),Y_batch]))
        loss /= batch_size
        one_hot = np.zeros_like(yout)
        one_hot[np.arange(batch_size),Y_batch]=1.0
        #backward propagation
        #y = label[i%a].reshape(10,1)
        #loss = float(np.dot(y.T,-np.log(yout)))
        if i%100==0:
            print "loss is %s" %loss
            loss_array= np.append(loss_array,loss)
            
        delta_w = np.dot((yout - one_hot).T,hout.T)
        delta_w /= batch_size
        #w_nobias = np.delete(w,200,1)
        #x=train[i%a].reshape(785,1)
        delta_h=np.dot(w.T,(yout-one_hot).T)
        w_nobias = np.delete(delta_h,200,0)
        dh = np.maximum(w_nobias,0.0)
        delta_v = np.dot(dh,X_batch)

        w -= epsilon*(decay**(i/(2*a)))*delta_w
        v -= epsilon*(decay**(i/(2*a)))*delta_v
        
    np.savetxt('w.csv', w, delimiter=',')
    np.savetxt('v.csv',v,delimiter=',')
    np.savetxt('loss.csv',loss_array,delimiter=',')
    return w,v

def predict(x,y,w,v):
    a,b= x.shape
    hit = 0
    y_prediction = np.array((0,0))
    for j in range(0,a):
        hin= np.dot(v,x[j])
        h2 = np.maximum(hin,0)
        h3 = h2.reshape(1,200)
        hout = add_bias(h3)
        hout = hout.T
        yin = np.dot(w,hout)
        yout = np.exp(yin) / np.sum(np.exp(yin), axis=0)
        y_1= yout.argmax(axis=0)
        y_prediction = np.append(y_prediction,y_1)
        y_pre = np.zeros((10,1))
        y_pre[y_1]=1        
        y_real = np.eye(NUM_CLASSES)[y[j]]
        if y_real[y_1]==y_pre[y_1]:
            hit +=1     
    accu(hit,a)
    return y_prediction

def predict_test(x,y,w,v):
    a,b= x.shape
    hit = 0
    y_prediction = np.array((0,0))
    for j in range(0,a):
        hin= np.dot(v,x[j])
        h2 = np.maximum(hin,0)
        h3 = h2.reshape(1,200)
        hout = add_bias(h3)
        hout = hout.T
        yin = np.dot(w,hout)
        yout = np.exp(yin) / np.sum(np.exp(yin), axis=0)
        y_1= yout.argmax(axis=0)
        y_prediction = np.append(y_prediction,y_1)
        y_pre = np.zeros((10,1))
        y_pre[y_1]=1        
        y_real = np.eye(NUM_CLASSES)[y[j]]
        if y_real[y_1]==y_pre[y_1]:
            hit +=1     
    return y_prediction
    
hid_num = 200
in_num = 784
out_num = 10

X_train, labels_train, X_test,labels_test = load_dataset()
#y_train = one_hot(labels_train)
X_train_bias = add_bias(X_train)
x_shu,y_shu = shuffle(X_train_bias,labels_train)
x_tra,x_val,y_tra,y_val = split(x_shu,y_shu)

#w = np.zeros((out_num,hid_num+1))
w = np.loadtxt("w.csv",delimiter=',')
#v = np.zeros((hid_num,in_num+1))
v = np.loadtxt("v.csv",delimiter=',')
w,v=NeuralNet(in_num,hid_num,out_num,x_tra,y_tra,5e-5,500000,0.9,20)
print "===============Training Accuracy================"
predict(x_tra,y_tra,w,v)
print "===============Validation Accuracy================"
predict(x_val,y_val,w,v)

x_test_bias = add_bias(X_test)
predict_test(x_test_bias,labels_test,w,v)



from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np

NUM_CLASSES = 10

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    X_train = X_train[:,:]
    X_test = X_test[:,:]
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train):
    ''' Build a model from X_train -> y_train '''
    dim = X_train.shape[1]
    label = y_train.shape[1]
    mat1 = np.zeros((dim,dim))
    mat2 = np.zeros((dim,label))
    n=0
    for x_i in X_train:
        y_i = y_train[n]
        mat1 += np.outer(x_i,x_i)
        mat2 += np.outer(x_i,y_i)
        n+=1
    while np.linalg.matrix_rank(mat1) != dim:
        mat1 = mat1 + 0.0003*np.eye(dim)
    model = np.dot(np.linalg.inv(mat1),mat2)   
    return model

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    label_int = np.array(labels_train)
    label_bool = np.zeros((labels_train.size,10), dtype=int)
    label_bool[np.arange(labels_train.size),label_int] = 1
    return label_bool

def predict(x,y, model):
    ''' From model and data points, output prediction vectors ''' 
    m = 0
    hits = 0
    DataSize = y.shape[0]
    for m in range(DataSize):
        xx=x[m]
        Predict_values = list(np.dot(model.T,xx))
        winners = [i for i, xx in enumerate(Predict_values) if xx==max(Predict_values)]
        winner = winners[len(winners)-1]
        z = [0 for xx in Predict_values]
        z[winner] =1
        prediction = z
        actual = list(y[m])
        if prediction == actual:
            hits += 1
    return hits

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    
    model = train(X_train, y_train)
    
    train_hits=predict(X_train,y_train,model)
    test_hits=predict(X_test,y_test,model)
    
    Train_accuracy = train_hits/float(y_train.shape[0])*100
    Test_accuracy = test_hits/float(y_test.shape[0])*100
    
    print "Training Accuracy = "+str(Train_accuracy)+"%","("+str(train_hits)+"/"+str(y_train.shape[0])+")"
    print "Test Accuracy = "+str(Test_accuracy)+"%","("+str(test_hits)+"/"+str(y_test.shape[0])+")"
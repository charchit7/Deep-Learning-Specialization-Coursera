import numpy as np
import copy

def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0.
    return w,b


def propagate(w, b, X, Y):

    m = X.shape[1]
    # epsilon = 1e-5
    A = sigmoid(np.dot(w.T,X) + b)
    cost = (-1/m) * (np.dot(Y,np.log(A).T) + np.dot((1-Y),np.log(1-A).T))

    dw = 1 / m *(np.dot(X,(A -Y).T))
    db = 1/m * (np.sum(A-Y))
    
    cost = np.squeeze(np.array(cost))
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def optimize(w,b,X,Y,num_iterations=100,learning_rate=0.009,print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []
    for i in range(num_iterations):

        #gradient and cost calculations
        grads, cost = propagate(w,b,X,Y)

        #retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        #update 
        w = w - learning_rate * dw
        b = b - learning_rate * db

        #record cost every 100th iterations
        if i%100==0:
            costs.append(cost)

            if print_cost:
                print("cost after %i:%f" %(i,cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w,b,X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        if A[0][i] > 0.5:
            Y_prediction[0][i] = 1.0
        else:
            Y_prediction[0][i] = 0.0

    return Y_prediction


def logisticregressionmodel(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    

    w,b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs =  optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
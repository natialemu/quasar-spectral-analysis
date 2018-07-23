import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn import preprocessing


class LinearRegression:

    def __init__(self,features, labels,train_ratio=0.75):
        self.featureMatrix = np.array(features)
        self.label = np.array(labels)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.featureMatrix,self.label,train_ratio)
        self.theta = np.array()
        for i in range(len(self.x_train)):
            self.theta[i] = random.random() #random number between 0 and 1
        self.standarizeData()

    def fit(self,gradiet_descent=True):
        if(gradiet_descent):
            self.fit_gradient_descent()
        else:
            self.fit_newton_method()

    def fit_gradient_descent(self):
        num_iter = 1000
        alpha = 1
        d_err_vector = np.multiply(np.add(np.matmul(self.theta,self.x_train),self.y_train),self.featureMatrix)
        for i in range(num_iter):
            self.theta = np.subtract(self.theta,np.multiply(alpha,d_err_vector))
            d_err_vector = np.multiply(np.add(np.matmul(self.theta,self.x_train),self.y_train),self.featureMatrix)

    def h(self,current_x, current_y):
        output_vector = np.matmul(self.theta,self.featureMatrix)
        return output_vector

    def converges(self,f_of_theta):
        threshold=0.5
        for err in f_of_theta:
            if err > threshold:
                return False
        return True

    def fit_newton_method(self):
        #inputs: Theta, X_train, Y_train
        # make sure Theta is initialized to random number in constructor
        # generate threshold
        # err_vector = Square(multiply(theta,X_train) + Y_train)*1/2
        # while (errHasConverged(err_vector,threshold)):  //this means every value of err_vector has to be above threshold
        #    d_err_vector = multiply(theta,X_train) = Y
        #    self.Theta = Self.theta - inverse(err_vector)*d_err_vector
        #    err_vector = Square(multiply(theta,X_train) + Y_train)*1/2
        threshold = 0.0001
        err_vector = np.multiply(np.square(np.add(np.matmul(self.theta,self.x_train),self.y_train)),0.5)
        while self.errHasConverged(err_vector,threshold):
            d_err_vector = np.multiply(np.add(np.matmul(self.theta,self.x_train),self.y_train),self.featureMatrix)
            self.theta = np.subtract(self.theta,np.multiply(np.linalg.inv(err_vector),d_err_vector))
            err_vector = np.multiply(np.square(np.add(np.matmul(self.theta,self.x_train),self.y_train)),0.5)

    def errHasConverged(self,err_vector,threshold):
        for error in err_vector:
            if error  > threshold:
                return False
        return True

    def standarizeData(self):
        scaler = preprocessing.StandardScaler()
        self.featureMatrix = scaler.fit_transform(self.featureMatrix)
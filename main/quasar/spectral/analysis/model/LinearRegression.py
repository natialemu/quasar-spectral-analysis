import numpy as np
from sklearn.model_selection import train_test_split


class LinearRegression:

    def __init__(self,features, labels,train_ratio=0.75):
        self.featureMatrix = np.array(features)
        self.label = np.array(labels)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.featureMatrix,self.label,train_ratio)
        self.theta = np.array()
        for i in range(len(self.x_train)):
            self.theta[i] = 0.5 #random number between 0 and 2



        #input
        #oupi

    def fit(self,gradiet_descent=True):
        if(gradiet_descent):
            self.fit_gradient_descent()
        else:
            self.fit_newton_method()

    def fit_gradient_descent(self):
        num_iter = 1000
        learning_rate = 1
        for i in range(num_iter):
            for j in range(len(self.x_train)):#for each example
                change_in_theta = learning_rate*(self.h(self.x_train[j],self.y_train[j]) - self.y_train[j])*self.x_train[j] #this is a list
                if(self.thetaHasntConverged(change_in_theta)):
                    self.updateTheta(change_in_theta)


    def h(self,current_x, current_y):



    def fit_newton_method(self):

    def updateTheta(self,theta_change):

    def thetaHasntConverged(self, theta_change):








    # Standardize the features

    #
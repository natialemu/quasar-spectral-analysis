import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn import preprocessing
from scipy.linalg import expm
class LocallyWightedRegression:

    def __init__(self, features, labels, train_ratio=0.75):
        self.featureMatrix = np.array(features)
        self.label = np.array(labels)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.featureMatrix, self.label,
                                                                                train_ratio)
        self.theta = np.array()
        for i in range(len(self.x_train)):
            self.theta[i] = random.random()  # random number between 0 and 1
        self.standarizeData()

    def fit(self,x_queries):
        num_iter = 1000
        alpha = 1
        d_err_vector = np.multiply(np.add(np.matmul(self.theta,self.x_train),self.y_train),self.featureMatrix)
        d_wighted_err_vector = np.multiply(self.getWeight(x_queries),d_err_vector)
        for i in range(num_iter):
            self.theta = np.subtract(self.theta, np.multiply(alpha, d_wighted_err_vector))
            d_err_vector = np.add(np.matmul(self.theta, self.x_train), self.y_train)
            d_wighted_err_vector = np.multiply(self.getWeight(x_queries), d_err_vector)

    def getWeight(self,x_queries):
        weights = []
        for example in self.featureMatrix:
            exponent_matrix = np.square(np.subtract(x_queries,example))
            current_weight = expm(exponent_matrix)
            weights.append(current_weight)
        return weights

    def h(self, current_x, current_y):
        self.fit(current_x)
        output_vector = np.matmul(self.theta, self.featureMatrix)
        return output_vector

    def converges(self, f_of_theta):
        threshold = 0.5
        for err in f_of_theta:
            if err > threshold:
                return False
        return True

    def errHasConverged(self, err_vector, threshold):
        for error in err_vector:
            if error > threshold:
                return False
        return True

    def standarizeData(self):
        scaler = preprocessing.StandardScaler()
        self.featureMatrix = scaler.fit_transform(self.featureMatrix)
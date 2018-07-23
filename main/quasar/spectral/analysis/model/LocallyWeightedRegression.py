
class LocallyWightedRegression:

'''

Differences between Linear Regression and Locally Weighted Regression:
     1.Error function = 0.5*Weight(h(x_train) - y_train)**2
          Wight = e**(x-x_query)**2/learning_rate
     2. No fitting during init. everytime predict is called, first you call
        fit with the query param passed, then once that is done, continue prediction

'''
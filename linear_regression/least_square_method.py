import pandas as pd
import numpy as np
import os

class LSM:
    
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        """
        This function calculates coefficient and y-intercept for linear equation that best fits the given data.
        Note that there are only one feature considered for this function
        y=m*x + b, where m is coefficient and b is y-intercept
        m = (sigma((xi-mean_x)(yi-mean_y)))/(sigma((xi-mean_x)**2))
        b = mean_y - m*mean_x
        """
        mean_x = np.mean(X_train.values)
        mean_y = np.mean(y_train.values)
        num = np.sum((X_train - mean_x) * (y_train - mean_y))
        deno = np.sum((X_train - mean_x) ** 2)
        self.m = num / deno
        self.b = mean_y - self.m * mean_x

    def predict(self, X_test):
        return self.m * X_test + self.b


df = pd.read_csv(os.getcwd() + "/linear_regression/tvmarketing.csv")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['TV'], df['Sales'], test_size=0.3, random_state = 1)
model = LSM()
model.fit(X_train, y_train)
y_actual = model.predict(X_test)
assert y_actual.shape[0] == df.shape[0] * 0.3
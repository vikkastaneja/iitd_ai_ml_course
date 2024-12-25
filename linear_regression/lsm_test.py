import sys
import os
import pytest
print(os.curdir)
sys.path.append('./')

def test_generated_distribution():
    current_dir = os.path.dirname(__file__)
    sys.path.append(current_dir)

    import numpy as np
    import pandas as pd
    from least_square_method import LSM

    df = pd.read_csv(os.getcwd() + "/linear_regression/tvmarketing.csv")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, _ = train_test_split(df['TV'], df['Sales'], test_size=0.3, random_state = 1)
    model = LSM()
    model.fit(X_train, y_train)
    model.predict(X_test)
    
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()
    linear_model.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))
    linear_model.predict(pd.DataFrame(X_test))

    assert "{:.2f}".format(model.m) == "{:.2f}".format(linear_model.coef_[0][0])
    assert "{:.2f}".format(model.b) == "{:.2f}".format(linear_model.intercept_[0])

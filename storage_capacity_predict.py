import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# http://www.stat.yale.edu/Courses/1997-98/101/anovareg.htm
# https://en.wikipedia.org/wiki/Coefficient_of_determination
from sklearn import linear_model


def main():
    pass


def get_boundary(days, full_percent):
    start_day = -10

    plt.scatter(days, full_percent,  color='red')
    # Train the model using the training sets
    TODO = -20
    for i in range(-10, TODO, -1):
        x = days[i:]
        y = full_percent[i:]
        print(type(x))
        print(type(y))
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        preds = regr.predict(x)

        plt.plot(x, preds, color='blue', linewidth=3)
    plt.show()




if __name__ == '__main__':
    df=pd.read_csv('data.csv', sep=',')
    vals = df.values
    days = vals[:, 0]
    days = np.reshape(days, (-1, 1))
    full_percent = vals[:, 1]
    days = np.array([day-100  for day in days])
    b = get_boundary(days, full_percent)

    #plt.xticks(())
    #plt.yticks(())


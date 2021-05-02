import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score
from sklearn import linear_model

# https://en.wikipedia.org/wiki/Coefficient_of_determination
# https://www.solarwinds.com/storage-resource-monitor/use-cases/storage-capacity-planning
# https://www.manageengine.com/network-monitoring/storage-capacity-forecasting-planning.html
# https://cloud.netapp.com/blog/when-to-buy-new-storage-predict-your-future-data-use

def get_fit(sample_days, full_percent, display_plot = False):
    start_day = -10
    if display_plot is True:
        plt.scatter(sample_days, full_percent,  color='red')
    # Train the model using the training sets
    earliest_day = -100
    max_r2 = -sys.maxsize - 1
    max_idx = None
    for i in range(start_day, earliest_day, -1):
        x = sample_days[i:]
        y = full_percent[i:]
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        preds = regr.predict(x)
        r2 = r2_score(y, preds)
        if r2 > max_r2:
            max_r2 = r2
            max_idx = i
            max_x = x
            max_preds = preds
            max_reg_model = regr
    if display_plot is True:
        plt.plot(max_x, max_preds, color='green', linewidth=3)
        plt.show()
    return  max_reg_model

if __name__ == '__main__':
    df=pd.read_csv('data.csv', sep=',')
    vals = df.values
    days = vals[:, 0]
    days = np.reshape(days, (-1, 1))
    full_percent = vals[:, 1]
    days = np.array([day-100  for day in days])
    good_fit = get_fit(days, full_percent)
    print(good_fit)

    #plt.xticks(())
    #plt.yticks(())


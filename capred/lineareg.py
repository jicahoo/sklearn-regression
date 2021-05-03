from datetime import date
from typing import List
import pandas as pd
import sys
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score


class DateAndFullPercent(object):
    def __int__(self, collect_date: date, full_percent:int):
        self.collect_date = collect_date
        self.full_percent = full_percent


class PieceLinearReg(object):
    def __init__(self, train_data: List[DateAndFullPercent]):
        self.train_data = train_data
        self.model = None

    @staticmethod
    def load_csv(csv_path):
        df = pd.read_csv(csv_path, sep=',')
        df['Day'] = pd.to_datetime(df['Day'])
        last_day = df.at[df.shape[0] - 1, 'Day']
        def m(day):
            return (day - last_day).total_seconds() / (24 * 3600)

        df['Day'] = df['Day'].map(m)
        return df

    @staticmethod
    def convert_to_numpy_types(train_data: List[DateAndFullPercent]):
        pass

    @staticmethod
    def get_fit(sample_days, full_percent, display_plot=False):
        start_day = -10
        if display_plot is True:
            plt.scatter(sample_days, full_percent, color='red')
        # Train the model using the training sets
        earliest_day = -100
        max_r2 = -sys.maxsize - 1
        max_idx = None
        max_x = None
        max_preds = None
        max_reg_model = None
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
        return max_reg_model

    def train(self):
        sample_days, full_percent = PieceLinearReg.convert_to_numpy_types(self.train_data)
        self.model = PieceLinearReg.get_fit(sample_days, full_percent)
        return self.model


if __name__ == '__main__':
    r = PieceLinearReg.load_csv('../date_percent.csv')
    print(r)
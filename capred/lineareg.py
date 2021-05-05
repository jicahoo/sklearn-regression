import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sqlalchemy import create_engine


class PieceLinearReg(object):
    def __init__(self):
        '''
        train_data: pandas DataFrame(Date, FullPercent)
        model: the final Linear Function output.
        r_2: https://en.wikipedia.org/wiki/Coefficient_of_determination
        start_day_of_best_fit: the start day to get the best linear model. The best model means the model with hightest
                                self.r_2
        min_days: the least days we need to include to train our model.
        '''
        self.train_data = None
        self.model = None
        self.r_2 = None
        self.start_day_of_best_fit = None
        self.min_days = 10

    @staticmethod
    def from_csv(csv_path):
        p = PieceLinearReg()
        p.train_data = PieceLinearReg._load_csv_file(csv_path)
        return p

    @staticmethod
    def from_postgres(postgres_url, table_name):
        p = PieceLinearReg()
        p.train_data = PieceLinearReg._load_postgres_table(postgres_url, table_name)
        return p

    @staticmethod
    def _load_csv_file(csv_path):
        df = pd.read_csv(csv_path, sep=',')
        return PieceLinearReg._convert_data(df)

    @staticmethod
    def _convert_data(df):
        '''
        Convert date to relative date format. e.g.   
        From:
            +----------+-------------+
            |   DAY    | PERCENTFULL |
            +----------+-------------+
            |          |             |
            | 1/1/2021 | 16          |
            |          |             |
            | 1/2/2021 | 13          |
            |          |             |
            | 1/3/2021 | 22          |
            |          |             |
            | 1/4/2021 | 25          |
            |          |             |
            | 1/5/2021 | 18          |
            |          |             |
            | 1/6/2021 | 15          |
            |          |             |
            | 1/7/2021 | 26          |
            |          |             |
            | 1/8/2021 | 27          |
            |          |             |
            | 1/9/2021 | 20          |
            +----------+-------------+
        To:
            +-----+-------------+
            | DAY | PERCENTFULL |
            +-----+-------------+
            |     |             |
            | -8  | 16          |
            |     |             |
            | -7  | 13          |
            |     |             |
            | -6  | 22          |
            |     |             |
            | -5  | 25          |
            |     |             |
            | -4  | 18          |
            |     |             |
            | -3  | 15          |
            |     |             |
            | -2  | 26          |
            |     |             |
            | -1  | 27          |
            |     |             |
            | 0   | 20          |
            +-----+-------------+
        :param df:  pandas DataFrame with columen: Date and FullPercent
        :return: 
        '''
        df['Day'] = pd.to_datetime(df['Day'])
        last_day = df.at[df.shape[0] - 1, 'Day']
        def m(day):
            return (day - last_day).total_seconds() / (24 * 3600)

        df['Day'] = df['Day'].map(m)
        return df

    @staticmethod
    def _load_postgres_table(postgres_url, table_name):
        engine = create_engine(postgres_url)
        df = pd.read_sql_table(table_name, engine)
        df = df.drop('index', 1)
        return PieceLinearReg._convert_data(df)

    def fit(self, display_plot=False):
        vals = self.train_data.values
        days = vals[:, 0]
        sample_days = np.reshape(days, (-1, 1))
        full_percent = vals[:, 1]
        base_day = 0 - self.min_days
        if display_plot is True:
            plt.scatter(sample_days, full_percent, color='red')
        # Train the model using the training sets
        earliest_day = -100
        max_idx, max_preds, max_r2, max_reg_model, max_x = PieceLinearReg.get_best_subset_for_linear_regression(
                                                                earliest_day, base_day,
                                                                full_percent, sample_days)
        if display_plot is True:
            plt.plot(max_x, max_preds, color='blue', linewidth=3)
            plt.show()
        self.model = max_reg_model
        self.r_2 = max_r2
        self.start_day_of_best_fit = max_idx
        return self.model

    @staticmethod
    def get_best_subset_for_linear_regression(earliest_day, start_day, full_percent, sample_days):
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
        return max_idx, max_preds, max_r2, max_reg_model, max_x

    def __str__(self):
        fmt_str = \
            "Learned Formula       : y = {:.2f}*x + {:.2f}\n"\
            "start_day_of_best_fit : {}, \n" \
            "R^2                   : {:.2f},\n" \
            .format(
                    self.model.coef_[0],
                    self.model.intercept_,
                    self.start_day_of_best_fit,
                    self.r_2
            )
        return fmt_str

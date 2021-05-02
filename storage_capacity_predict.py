import pandas as pd

if __name__ == '__main__':
    df=pd.read_csv('data.csv', sep=',')
    vals = df.values
    print(type(vals))
    print(vals)
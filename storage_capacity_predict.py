import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df=pd.read_csv('data.csv', sep=',')
    vals = df.values
    print(type(vals))
    print(vals)
    days = vals[:,0]
    full_percent = vals[:, 1]
    print(days)
    print(full_percent)
    plt.scatter(days, full_percent,  color='black')

    #plt.xticks(())
    #plt.yticks(())

    plt.show()

# sklearn-regression

# Introduction
* It a simple demo program to use LinearRegression to do the task of predicting the storage usage trend.

# How to run and main logic
* prepare **python3** environment
* pip install -i requirements.txt
* In Pycharm, run main.py. The main.py will:
    * Load data from sample_data/date_percent.csv
    * It will find a best subset data to fit the Linear Function.
    * It will plot the train data and fitted linear function
    * It will output the learned model info.
    
# About supported data source
* PieceLinearReg can suport two kinds of datasource
    * One is csv file via PieceLinearReg.from_csv method
    * One is PostgreSQL via PieceLinearReg.from_postgre method.
# Plot exmaple:
![image](https://user-images.githubusercontent.com/4143339/117252180-3006e980-ae78-11eb-8bae-8d7dd73caa7c.png)


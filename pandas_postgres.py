import pandas as pd
from sqlalchemy import create_engine
import capred.lineareg as cl

# follows django database settings format, replace with your own settings
DATABASES = {
    'sklearn':{
        'NAME': 'sklearn',
        'USER': 'sklearn',
        'PASSWORD': 'sklearn',
        'HOST': 'localhost',
        'PORT': 5432,
    },
}

# choose the database to use
db = DATABASES['sklearn']

# construct an engine connection string
engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
    user = db['USER'],
    password = db['PASSWORD'],
    host = db['HOST'],
    port = db['PORT'],
    database = db['NAME'],
)

# create sqlalchemy engine
engine = create_engine(engine_string)

# read a table from database into pandas dataframe, replace "tablename" with your table name
df = pd.read_sql_table('hello',engine)
print(df)
#df = pd.read_csv('date_percent.csv', sep=',')
#df.to_sql("date_percent", engine)
df = pd.read_sql_table('date_percent',engine)
print(df)
print(df.info)

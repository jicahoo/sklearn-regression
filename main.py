from sqlalchemy import create_engine

from capred.lineareg import PieceLinearReg


def test_postgres():
    DATABASES = {
        'sklearn': {
            'NAME': 'sklearn',
            'USER': 'sklearn',
            'PASSWORD': 'sklearn',
            'HOST': 'localhost',
            'PORT': 5432,
        },
    }
    # choose the database to use
    db = DATABASES['sklearn']
    engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
        user=db['USER'],
        password=db['PASSWORD'],
        host=db['HOST'],
        port=db['PORT'],
        database=db['NAME'],
    )
    p = PieceLinearReg.from_postgres(engine_string, 'date_percent')
    p.fit(True)
    print(p)


def test_csv():
    r = PieceLinearReg.from_csv('sample_data/date_percent.csv')
    r.fit(display_plot=True)
    print(r)


if __name__ == '__main__':
    test_csv()

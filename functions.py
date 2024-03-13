from datetime import datetime
import config, requests, psycopg2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

class evaluation:
    def __init__(self, startDate: datetime, table = None, currency = None) -> None:
        # when created, query the database, preprocess, and call the model api
        # raise error if both table and currency are not provided
        # raise error if the query data is not large enough to perform analytics (<= 1440 rows)
        if table is None and currency is None:
            raise ValueError("At least one of 'table' or 'currency' must be provided in the constructor.")
        print(f"analysis is done at {datetime.now()}")
        self.table = table if table is not None else f"crypto_ind_{currency}"
        self.startDate = startDate
        print("querying")
        self.df = self.query()
        if len(self.df["time"]) <= 1440:
            raise ValueError("The provided startDate must precede the latest data in the data warehouse by at least one day.")
        print("processing")
        self.input, self.actual = self.preprocess()
        print("requesting to the model server")
        self.pred = self.predict()
        print("completed")
        
        

    def query(self):
        # query indicator from the database starting from startDate
        # query from the table if table is not null
        # else query by using currency
        conn = psycopg2.connect(
            database="seniorproj_maindb",
            user=config.DATABASE_USERNAME,
            password=config.DATABASE_PASSWORD,
            host=config.DATABASE_HOST,
            port=5432)
        cursor = conn.cursor()
        query = f"""SELECT time, currency, close, close_minmax_scale,
        ma7_25h_scale, ma25_99h_scale, ma7_25d_scale from {self.table} where time >= '{self.startDate}'"""
        cursor.execute(query = query)
        results = cursor.fetchall()
        columns = ['time', 'currency','close' ,'close_minmax_scale',
                   'ma7_25h_scale', 'ma25_99h_scale', 'ma7_25d_scale']
        dataframe = pd.DataFrame([dict(zip(columns, result)) for result in results])
        cursor.close()
        conn.close()
        return dataframe
    
    def preprocess(self):
        # extract the indicators from the query results
        x = self.df.iloc[:-1440]
        x = x[['close_minmax_scale', 'ma7_25h_scale',
               'ma25_99h_scale', 'ma7_25d_scale']]
        x = x.values
        close = self.df["close"]
        # calculate the growth of the close price respectively to the previous day
        y = []
        for i in range(len(self.df) - 1440):
            y.append((close[i+1440]-close[i])/close[i])
        return x, y
    
    def predict(self):
        # call the api to the model hosting server (url from env file)
        # request_body is numpy array with shape of (n, 4)
        # response body is a list of predicted growth (percentage)
        url = config.MODEL_URL
        # print(self.input)
        response = requests.post(url, json = self.input.tolist())
        if response.status_code == 200:
            return response.json()
            # print("requested")
            # print(response.json())
        else:
            raise Exception(f"Error: {response.status_code} - {response.reason}")
    
    def plot(self):
        # plot the actual growth vs the predicted growth
        df = self.df
        x = list(df["time"][1440:])

        # Plotting the line
        plt.plot(x, self.actual, label='Actual growth')
        plt.plot(x, self.pred, label='Predicted growth')
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.ylabel('Growth')
        plt.title(f'Comparison between predicted and actual growth of {df["currency"][0]}')
        plt.legend()
        plt.show()

    def plot_price(self):
        # plot the close price of the currency
        df = self.df
        plt.plot(df["time"], df["close"], label='price')
        plt.xlabel('time')
        plt.xticks(rotation=45)
        plt.ylabel('price')
        plt.title(f'{df["currency"][0]} price vs time')
        plt.legend()
        plt.show()
    
    def classification_report(self):
        # return the sklearn.metrics.classfication_report
        # growth >= 0 is treated as class 1
        # growth < 0 is treated as class 0
        self.actual_class = [1 if e >= 0 else 0 for e in self.actual]
        self.pred_class = [1 if e >= 0 else 0 for e in self.pred]
        report = metrics.classification_report(self.actual_class, self.pred_class)
        return report
    
    def invest(self, thresh = 0):
        # to be implemented later
        # calculate accuracy of model concerning pred higher than threshold
        # calculate cumulative percentage gain and loss if invest only when pred higher than threshold
        accuracy = 0
        performance = -1e9
        return {"accuracy": accuracy, "cum_performance": performance}
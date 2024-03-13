from datetime import datetime, date
import config, requests, psycopg2, os, csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from binance_historical_data import BinanceDataDumper
from transform import transform
class evaluation:
    def __init__(self, startDate: datetime, table = None, currency = None, query = True, url = None) -> None:
        # when created, query the database, preprocess, and call the model api
        # raise error if both table and currency are not provided
        # raise error if the query data is not large enough to perform analytics (<= 1440 rows)
        # if query is false, do not call any methods
        self.url = url if url is not None else config.MODEL_URL
        if query:
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
        close = list(self.df["close"])
        # calculate the growth of the close price respectively to the previous day
        y = []
        for i in range(len(self.df) - 1440):
            y.append((close[i+1440]-close[i])/close[i])
        return x, y
    
    def predict(self):
        # call the api to the model hosting server (url from env file)
        # request_body is numpy array with shape of (n, 4)
        # response body is a list of predicted growth (percentage)
        # print(self.input)
        response = requests.post(self.url, json = self.input.tolist())
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
    
class localEvaluation(evaluation):
    def __init__(self, currency: str, url=None):
        super().__init__(startDate=None, query=False, url=url)
        self.currency = currency
        self.raw_data = self.load_csv()
        self.raw_data.set_index('time', inplace=True)
        self.df = self.transform()
        if len(self.df["time"]) <= 1440:
            raise ValueError("The provided startDate must precede the latest data in the data warehouse by at least one day.")
        print("processing")
        self.input, self.actual = self.preprocess()
        print("requesting to the model server")
        self.pred = self.predict()
        print("completed")
        
    @staticmethod
    def download_data(currency:str, startDate: date):
        data_dumper = BinanceDataDumper(
            path_dir_where_to_dump=".",
            asset_class="spot",  # spot, um, cm
            data_type="klines",  # aggTrades, klines, trades
            data_frequency="1m",
        )
        # print(startDate)
        data_dumper.dump_data(
        tickers=[currency],
        date_start=startDate,
        date_end=None,
        is_to_update_existing=True,
        tickers_to_exclude=["UST"]
        )
        
    @staticmethod
    def readfiles(dir_path, files, currency):
        return_df = pd.DataFrame()
        for file in files:
            with open(rf"{dir_path}\{file}", 'r', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                selected_data = [
                    {
                        "currency":currency,
                        "time":datetime.utcfromtimestamp(int(row[0])/1000),
                        "open":row[1],
                        "high":row[2],
                        "low":row[3],
                        "close":row[4],
                        "volume":row[5]
                    }
                    for row in data]
                df = pd.DataFrame(selected_data)
                return_df = pd.concat([return_df, df], ignore_index=True)
        return return_df
    
    def load_csv(self):
        currency = self.currency
        df = pd.DataFrame()
        dir_path = rf".\spot\daily\klines\{currency}\1m"
        files = os.listdir(dir_path)
        temp = localEvaluation.readfiles(currency=currency, dir_path=dir_path, files=files)
        df = pd.concat([temp, df], ignore_index=True)
        dir_path = rf".\spot\monthly\klines\{currency}\1m"
        files = os.listdir(dir_path)
        temp = localEvaluation.readfiles(currency=currency, dir_path=dir_path, files=files)
        df = pd.concat([temp, df], ignore_index=True)
        df["close"] = df["close"].astype(float)
        return df
    
    def transform(self):
        df = transform(self.raw_data)
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        # print(df.columns)
        return df
    

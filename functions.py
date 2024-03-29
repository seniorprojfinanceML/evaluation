from datetime import datetime, date, timedelta
import config, requests, psycopg2, os, csv, pytz
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from binance_historical_data import BinanceDataDumper
from transform import transform
class Evaluation:
    def __init__(self, startDate: datetime, table = None, currency = None,
                 query = True, url = None, model = "CBR", version = None, alias = None) -> None:
        # when created, query the database, preprocess, and call the model api
        # raise error if both table and currency are not provided
        # raise error if the query data is not large enough to perform analytics (<= 1440 rows)
        # if query is false, do not call any methods
        self.url = url if url is not None else config.MODEL_URL
        self.model = model
        self.version = version
        self.alias = alias

        if query:
            if table is None and currency is None:
                raise ValueError("At least one of 'table' or 'currency' must be provided in the constructor.")
            print(f"analysis is done at {datetime.now()}")
            self.table = table if table is not None else f"crypto_ind_{currency}"
            self.startDate = startDate
            print("querying")
            self.input, self.df = self.query()
            if len(self.df["time"]) <= 1440:
                raise ValueError("The provided startDate must precede the latest data in the data warehouse by at least one day.")
            print("processing")
            self.actual = self.preprocess()
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
        query = f"""SELECT time, currency, close,
        ma7_25h_scale, ma25_99h_scale, ma7_25d_scale from {self.table} where time >= '{self.startDate}'"""
        cursor.execute(query = query)
        results = cursor.fetchall()
        columns = ['time', 'currency','close',
                   'ma7_25h_scale', 'ma25_99h_scale', 'ma7_25d_scale']
        
        query = f"""SELECT close_minmax_scale, time from {self.table} where time >= '{(self.startDate)-timedelta(days=28)}'"""
        cursor.execute(query = query)
        price_results = cursor.fetchall()
        # print(price_results[:6])
        
        time_diff = [1, 25, 49, 73, 97, 121, 145, 169, 193, 217, 241, 265, 289, 313, 337, 361, 385, 409, 433, 467, 491, 515, 539, 563, 587, 611, 635, 672]
        
        x = []
        # -1440 to ensure that the last day data is not used in the evaluaton (Since we will not have the actual growth for these values)
        for i in range (len(results)-1440):
            l = []
            t = []
            # Since price_results start from -672hrs, i+40320 => price at the current time
            l.append(price_results[i+40320][0])
            t.append(price_results[i+40320][1])
            l.extend([results[i][3],results[i][4],results[i][5]])
            # i+672 => current time so i+40320-60 => price from -1 hour
            for e in time_diff:
                l.append(price_results[i+40320-e*60][0])
                t.append(price_results[i+40320-e*60][1])
            print(t)
            x.append(l)
        dataframe = pd.DataFrame([dict(zip(columns, result)) for result in results])
        cursor.close()
        conn.close()
        return x, dataframe
    
    def preprocess(self):
        close = list(self.df["close"])
        # calculate the growth of the close price respectively to the previous day
        y = []
        for i in range(len(self.df) - 1440):
            y.append((close[i+1440]-close[i])/close[i])
        return y
    
    def predict(self):
        # call the api to the model hosting server (url from env file)
        # request_body is numpy array with shape of (n, 4)
        # response body is a list of predicted growth (percentage)
        # print(self.input)
        req_body = {
            'input': self.input
        }
        if self.model is not None:
            req_body["model"] = self.model
        if self.version is not None:
            req_body["version"] = self.version
        if self.alias is not None:
            req_body["alias"] = self.alias
        response = requests.post(self.url, json = req_body)
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
    
    def mse(self):
        return mean_squared_error(self.actual, self.pred)
    
class LocalEvaluation(Evaluation):
    def __init__(self, currency: str, url=None, model = None, version = None, alias = None):
        super().__init__(startDate=None, query=False, url=url, model = model, version = version, alias = alias)
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
                        "time":datetime.fromtimestamp((int(row[0])/1000), tz=pytz.UTC),
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
        temp = LocalEvaluation.readfiles(currency=currency, dir_path=dir_path, files=files)
        df = pd.concat([temp, df], ignore_index=True)
        dir_path = rf".\spot\monthly\klines\{currency}\1m"
        files = os.listdir(dir_path)
        temp = LocalEvaluation.readfiles(currency=currency, dir_path=dir_path, files=files)
        df = pd.concat([temp, df], ignore_index=True)
        df["close"] = df["close"].astype(float)
        return df
    
    def transform(self):
        df = transform(self.raw_data)
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        # print(df.columns)
        return df
    

import pandas as pd
import numpy as np


class DataPipeline:

    def __init__(self, filepath, filepath2):
        self.filepath = filepath
        self.df = None
        self.risk_free_rate = filepath2
        self.rfr = None
        

    def load_characteristics(self):
        self.df = pd.read_csv(self.filepath)
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce', format='%Y%m%d')
        self.df['year_month'] = self.df['DATE'].dt.to_period('M')
        self.df = self.df.sort_values(['permno', 'DATE']).reset_index(drop=True)
        self.df['monthly_ret_raw'] = self.df.groupby('permno')['mom1m'].shift(-1)

        print(f"Loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        return self

    def load_risk_free_rate(self):
        self.rfr = pd.read_csv(self.filepath2, skiprows=4)
        self.rfr = self.rfr.rename(columns = {'Unnamed:0':'year_month'})
        self.rfr['year_month'] = pd.to_datetime(self.rfr['year_month'], errors='coerce', format = '%Y%m').dt.to_period('M')
        self.rfr = self.rfr.drop(columns=['Mkt-RF', 'SMB', 'HML'])
        self.rfr['RF'] = self.rfr['RF'].astype(np.float32)/100
        return self
    
    def clean_target(self):
        self.df = pd.merge(self.df, self.rfr, on='year_month', how='inner')
        self.df['monthly_excess_ret'] = self.df['monthly_ret_raw'] - self.df['RF']
        return self

    def impute_missing(self):
        exclude = ['monthly_ret_raw', 'permno', 'monthly_excess_ret', 'RF']
        num_cols = [c for c in self.df.select_dtypes(include='number').columns if c not in exclude]

        # fill all numeric columns at once with cross-sectional monthly median
        self.df[num_cols] = self.df[num_cols].fillna(
            self.df.groupby('year_month')[num_cols].transform('median')
        )

        # drop any rows where imputation couldn't fill (whole month was NaN)
        mask = self.df[num_cols].notna().all(axis=1)
        self.df = self.df.loc[mask].reset_index(drop=True)

        print(f"After imputation: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self

    def normalize(self):
        exclude = ['DATE', 'year_month', 'permno', 'monthly_ret_raw', 'monthly_excess_ret', 'RF']
        num_cols = [c for c in self.df.select_dtypes(include='number').columns if c not in exclude]

        for col in num_cols:
            self.df[col] = self.df.groupby('year_month')[col].transform(
                lambda x: (x.rank(method='average') - 1) / (len(x) - 1) * 2 - 1
            )

        print("Features normalized to [-1, 1]")
        return self
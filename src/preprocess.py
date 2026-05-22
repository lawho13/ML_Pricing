import pandas as pd
import numpy as np


class DataPipeline:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.indicator_matrix = None

    def load(self):
        self.df = pd.read_csv(self.filepath)
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce', format='%Y%m%d')
        self.df['year_month'] = self.df['DATE'].dt.to_period('M')
        self.df = self.df.sort_values(['permno', 'DATE']).reset_index(drop=True)
        self.df['ret'] = self.df.groupby('permno')['mom1m'].shift(-1)

        print(f"Loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        return self

    def impute_missing(self):
        num_cols = self.df.select_dtypes(include='number').columns

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
        exclude = ['DATE', 'year_month', 'permno', 'ret']
        num_cols = [c for c in self.df.select_dtypes(include='number').columns if c not in exclude]

        for col in num_cols:
            self.df[col] = self.df.groupby('year_month')[col].transform(
                lambda x: (x.rank(method='average') - 1) / (len(x) - 1) * 2 - 1
            )

        print("Features normalized to [-1, 1]")
        return self
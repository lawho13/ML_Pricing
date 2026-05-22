import pandas as pd
import numpy as np


class ExpandingWindow:

    def __init__(self, initial_train_years=10, val_years=1, test_years=1):
        self.initial_train_years = initial_train_years
        self.val_years = val_years
        self.test_years = test_years

    def split(self, df, date_col='DATE'):

        # sort all rows in the dataframe by their dates
        df = df.sort_values(date_col).reset_index(drop=True)

        start_date = df[date_col].min()
        end_date   = df[date_col].max()

        # let the current train_end be start_date plus the arg from the constructor for train years

        train_end = start_date + pd.DateOffset(years=self.initial_train_years)

        while True:
            # the backtesting loop - here, we set val_end to the date train_end + val from constructor, likewise with test_end
            val_end  = train_end + pd.DateOffset(years=self.val_years)
            test_end = val_end   + pd.DateOffset(years=self.test_years)

            # if this is the case, we've exceeded window length - break
            if test_end > end_date:
                break
            
            # these are all the valid data frames to return - we have a training dataframe, a validation data frame, and a test data frame - all within their respective date timelines
            train_df = df[df[date_col] <  train_end]
            val_df   = df[(df[date_col] >= train_end) & (df[date_col] < val_end)]
            test_df  = df[(df[date_col] >= val_end)   & (df[date_col] < test_end)]

            # in tuple format, return the data frames
            yield train_df, val_df, test_df

            # set train_end to test_end to adjust the window and move the time series window up for another testing round
            train_end = test_end

    # this method gives us X - input variables - and Y - target variables, for any dataframe
    @staticmethod
    def get_features_and_target(df, target_col='ret'):
        drop_cols = ['DATE', 'year_month', 'permno', target_col]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns]) 
        y = df[target_col]
        return X, y
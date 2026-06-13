import pandas as pd
import numpy as np


class ExpandingWindow:

    def __init__(self, initial_train_years=10, val_years=1, test_years=1):
        self.initial_train_years = initial_train_years
        self.val_years = val_years
        self.test_years = test_years

    def split_paper(df, initial_train_years=18, val_years=12, test_years=1, date_col="DATE"):
    
        df = df.sort_values(date_col).reset_index(drop=True)

        start_date = df[date_col].min()
        end_date   = df[date_col].max()
        train_end = start_date + pd.DateOffset(years=initial_train_years)
        
        while True:
            val_start = train_end
            val_end   = val_start + pd.DateOffset(years=val_years)

            test_start = val_end
            test_end   = test_start + pd.DateOffset(years=test_years)

            if test_end > end_date:
                break

            train_df = df[df[date_col] < train_end]

            val_df = df[
                (df[date_col] >= val_start) &
                (df[date_col] <  val_end)
            ]

            test_df = df[
                (df[date_col] >= test_start) &
                (df[date_col] <  test_end)
            ]

            print(
                f"Train: {train_df[date_col].min().date()} → "
                f"{train_df[date_col].max().date()} | "
                f"Val: {val_df[date_col].min().date()} → "
                f"{val_df[date_col].max().date()} | "
                f"Test: {test_df[date_col].min().date()} → "
                f"{test_df[date_col].max().date()}"
            )

            yield train_df, val_df, test_df

            # move forward ONE year only
            train_end = train_end + pd.DateOffset(years=1)

    # this method gives us X - input variables - and Y - target variables, for any dataframe
    @staticmethod
    def get_features_and_target(df, target_col='monthly_excess_ret'):
        drop_cols = ['DATE', 'year_month', 'permno', 'monthly_ret_raw', 'RF', target_col]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        temp_df = df.dropna(subset=feature_cols + [target_col])

        X = temp_df[feature_cols].values
        y = temp_df[target_col].values

        return X, y, temp_df
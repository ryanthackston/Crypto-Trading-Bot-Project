import numpy as np
import pandas as pd
import os
from cfg import *

def tv_split():
    time_step = BS
    os.makedirs('data3', exist_ok=True)
    df = pd.read_csv('data3/Test-Bitcoin-Final_label.csv', parse_dates=True)
    train_size = int(len(df) * 0.9)
    test_size = len(df) - train_size
    tv_df, test_df = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    train_size = int(len(tv_df) * .8)
    valid_size = len(tv_df) - train_size
    train_df, valid_df = tv_df.iloc[0:train_size], tv_df.iloc[train_size:len(tv_df)]
    train_df['Set'] = 'Train'
    valid_df['Set'] = 'Valid'
    test_df['Set'] = 'Test'
    train_df.to_csv('data3/train.csv', index=False)
    valid_df.to_csv('data3/valid.csv', index=False)
    test_df.to_csv('data3/test.csv', index=False)

def main():
    tv_split()
if __name__=="__main__":
    main()
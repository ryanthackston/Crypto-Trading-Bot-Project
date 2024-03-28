import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import boxcox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima
import arch
from sklearn.utils.class_weight import compute_class_weight
from cfg import *

def label_threshold(row, median_garch_cond_vola):

    if row[WAIT_LABEL] < median_garch_cond_vola:
        row['Decide'] = LABEL[2]  # we will wait because the price didnt change as much
    else:
        if row['Arima_fore_err'] >= 0:
            row['Decide'] = LABEL[0]  # price goes up
        else:
            row['Decide'] = LABEL[1]  # price goes down

    return row

def forecast(read_path: str):

    col_name1 = attribute[0]
    col_name2 = attribute[1]
    write_path = 'data3/Test-Bitcoin-Arima-Garch.csv'
    df = pd.read_csv(read_path, parse_dates=True)
    df['Log_open'] = np.log(df[col_name1])
    df['Log_close'] = np.log(df[col_name2])
    df['LogRet_open'] = df['Log_open'].diff()
    df2 = df.iloc[1:]
    # fit ARIMA on returns
    arima_model_fitted = pmdarima.auto_arima(df2['Log_open'])
    p, d, q = arima_model_fitted.order
    arima_residuals = arima_model_fitted.arima_res_.resid
    arima_forecasts = arima_model_fitted.arima_res_.forecasts
    arima_forecasts_errors= arima_model_fitted.arima_res_.forecasts_error
    arima_forecast_errors_cov = arima_model_fitted.arima_res_.forecasts_error_cov
    # fit a GARCH(1,1) model on the residuals of the ARIMA model
    garch = arch.arch_model(df2['LogRet_open'], p=1, q=1)
    garch_fitted = garch.fit()
    garch_resid = garch_fitted.resid
    garch_cond_vola = garch_fitted.conditional_volatility
    print(arima_residuals.shape)
    df2['Arima_resid'] = pd.DataFrame(arima_residuals.transpose())
    df2['Arima_fore'] = pd.DataFrame(arima_forecasts.transpose())
    df2['Arima_fore_err'] = pd.DataFrame(arima_forecasts_errors.transpose())
    #df2['Arima_fore_err_cov'] = pd.DataFrame(arima_forecast_errors_cov[0].transpose())
    df2['Garch_resid'] = pd.DataFrame(garch_resid.transpose())
    df2['Garch_cond_vola'] = pd.DataFrame(garch_cond_vola.transpose())

    # Use ARIMA to predict mu
    predicted_mu = arima_model_fitted.predict(n_periods=1)[0]
    # # Use GARCH to predict the residual
    garch_forecast = garch_fitted.forecast(horizon=1)
    predicted_et = garch_forecast.mean['h.1'].iloc[-1]
    # # Combine both models' output: yt = mu + et
    prediction = predicted_mu + predicted_et
    df2['Arima_fore'].iloc[-1] = prediction
    df2['Garch_resid'].iloc[-1] = predicted_et
    df2['Real_fore'] = df2['Arima_fore'] + df2['Garch_resid']
    df2.to_csv(write_path, index=False)



def label_csv():
    df = pd.read_csv('data3/Test-Bitcoin-Arima-Garch.csv', parse_dates=['Date'], index_col='Date', keep_date_col=True)
    median_vola = np.median(df['Garch_cond_vola'].to_numpy())
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['second'] = df.index.second
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df_label = pd.DataFrame()
    for idx, row in df.iterrows():
        row2 = label_threshold(row, median_vola)
        row2 = row2.to_frame().transpose()
        df_label = df_label.append(row2)
    df_label.to_csv('data3/Test-Bitcoin-Final_label.csv', index=False)

def compute_class_weights():
    df = pd.read_csv('data3/Test-Bitcoin-Final_label.csv')
    y = df['Decide'].to_numpy()
    class_weights = compute_class_weight(class_weight='balanced', classes=LABEL, y=y)
    print(class_weights)

def main():
    forecast(read_path=read_path4)
    label_csv()
    compute_class_weights()
if __name__=='__main__':
    main()
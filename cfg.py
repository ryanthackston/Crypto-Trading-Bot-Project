

# fitonecycle hypeerparameters
lr_range = [1e-5, 1e-2]
betas = [0.95, 0.999]
momentum_range = [0.85, 0.95]
k = 5
alpha = 0.5
wd = 0
eps = 1e-8
alpha = 0.5

# Training hyperparameters
epochs = 50
BS = 256
pooling = 'avg'
squeeze = True
dims = 32
worker = 4

# Preprocessing parameters
read_path2 = 'data2/Test-Bitcoin-Data-Overall.csv'
read_path3 = 'data2/Test-Bitcoin-Arima-Garch.csv'
read_path4 = 'data3/flerny.csv'



LABEL =['Buy', 'Sell', 'Wait']
attribute = ['Open', 'Close', 'High', 'Low', 'Volume']

WAIT_LABEL = 'Garch_cond_vola'
INVEST_LABEL = 'Arima_fore_err'

train_label = ['Open', 'Close', 'High', 'Low', 'Volume', 'Log_open', 'Log_close', 'LogRet_open', 'Arima_fore', 'Arima_fore_err', 'Garch_resid', 'Garch_cond_vola', 'hour', 'minute', 'day', 'month']
x_label = ['Open', 'Close', 'High', 'Low', 'Volume', 'Log_open', 'Log_close', 'LogRet_open', 'Arima_fore', 'Arima_fore_err', 'Garch_resid', 'Garch_cond_vola']
y_label = ['Decide', 'Encode']

class_weights2 = [1.34092357, 1.32581843, 0.66666922]


# Model
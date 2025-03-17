import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from statsmodels.tsa.stattools import coint
from scipy import stats
import time
import warnings
warnings.filterwarnings("ignore")

# Classe pour le filtre de Kalman
class KalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-3):
        self.x = 0.0
        self.P = 1.0
        self.Q = process_variance
        self.R = measurement_variance

    def update(self, measurement):
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred
        return self.x

# Configuration Alpaca
API_KEY = "VOTRE_API_KEY"  # Remplacez par votre clé
SECRET_KEY = "VOTRE_SECRET_KEY"  # Remplacez par votre clé
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Télécharger des données historiques
tickers = ["NVDA", "META"]
request_params = StockBarsRequest(
    symbol_or_symbols=tickers,
    timeframe=TimeFrame.Minute,
    start="2024-10-01 09:30:00-04:00",
    end="2024-10-01 16:00:00-04:00"
)
bars = client.get_stock_bars(request_params)

# Préparer les données
bars_df = bars.df
bars_df_reset = bars_df.reset_index()
prices = bars_df_reset.pivot(index='timestamp', columns='symbol', values='close').dropna()
returns = prices.pct_change().dropna()

# Paramètres HFT
window_size = 20
capital = 100
max_leverage = 2
coint_window = 60
vol_window = 20
portfolio = [capital]
stop_loss = -1.0
min_trade_size = 0.01

# Initialisation des filtres Kalman
kf_nvda = KalmanFilter()
kf_meta = KalmanFilter()

# Simulation HFT
actual_returns = []
beta = 1.0
start_time = time.time()

for t in range(window_size, len(returns)):
    iter_start = time.time()
    
    train_prices = prices.iloc[:t + 1]
    train_returns = returns.iloc[:t]
    current_nvda, current_meta = prices.iloc[t]['NVDA'], prices.iloc[t]['META']
    
    vol_window_data = train_returns.tail(vol_window)
    vol_nvda = vol_window_data['NVDA'].std() * np.sqrt(390)
    vol_meta = vol_window_data['META'].std() * np.sqrt(390)
    vol_avg = (vol_nvda + vol_meta) / 2
    corr_dynamic = vol_window_data['NVDA'].corr(vol_window_data['META'])
    
    if t >= coint_window and (t - window_size) % coint_window == 0:
        coint_result = coint(train_prices['NVDA'].tail(coint_window), train_prices['META'].tail(coint_window))
        if coint_result[1] < 0.05:
            beta = np.polyfit(train_prices['NVDA'].tail(coint_window), train_prices['META'].tail(coint_window), 1)[0]
    
    error_threshold_nvda = 0.05 * (1 + vol_nvda)
    error_threshold_meta = 0.05 * (1 + vol_meta)
    arb_threshold = 0.8 * (1 + vol_avg)
    
    expected_nvda = kf_nvda.update(current_nvda)
    expected_meta = kf_meta.update(current_meta)
    error_nvda = current_nvda - expected_nvda
    error_meta = current_meta - expected_meta
    
    spread = train_prices['NVDA'] - beta * train_returns['META']
    z_score = stats.zscore(spread)[-1]
    arb_signal = 1 if z_score < -arb_threshold else (-1 if z_score > arb_threshold else 0)
    
    error_signal_nvda = -np.sign(error_nvda) if abs(error_nvda) > error_threshold_nvda else 0
    error_signal_meta = -np.sign(error_meta) if abs(error_meta) > error_threshold_meta else 0
    corr_signal = 1 if corr_dynamic < 0.4 else -1 if corr_dynamic > 0.9 else 0
    
    nvda_signal = error_signal_nvda
    meta_signal = error_signal_meta
    if arb_signal != 0:
        nvda_signal = arb_signal
        meta_signal = -arb_signal
    else:
        combined_signal = error_signal_nvda + error_signal_meta + corr_signal
        nvda_signal = np.sign(combined_signal) if abs(combined_signal) > 0.5 else 0
        meta_signal = np.sign(combined_signal) if abs(combined_signal) > 0.5 else 0
    
    leverage = max_leverage if vol_avg < 0.015 else 1.0
    
    total_vol = vol_nvda + vol_meta
    weight_nvda = (vol_meta / total_vol) if total_vol > 0 else 0.5
    weight_meta = (vol_nvda / total_vol) if total_vol > 0 else 0.5
    
    nvda_amount = (leverage * capital * weight_nvda) * nvda_signal
    meta_amount = (leverage * capital * weight_meta) * meta_signal
    nvda_qty = max(min_trade_size, abs(nvda_amount) / current_nvda) if nvda_amount != 0 else 0
    meta_qty = max(min_trade_size, abs(meta_amount) / current_meta) if meta_amount != 0 else 0
    position = (nvda_amount + meta_amount) / 2
    
    actual_return = returns.iloc[t].mean()
    daily_pnl = position * actual_return
    if daily_pnl < stop_loss:
        daily_pnl = stop_loss
    capital += daily_pnl
    portfolio.append(capital)
    actual_returns.append(actual_return)
    
    if t % 20 == 0:
        print(f"Minute {t} ({prices.index[t]}): Capital = {capital:.2f}, Leverage = {leverage:.1f}")
        print(f"  NVDA: Signal = {nvda_signal} (Achat/Vente: {nvda_amount:.2f}$ ≈ {nvda_qty:.2f} actions), Error = {error_nvda:.2f}")
        print(f"  META: Signal = {meta_signal} (Achat/Vente: {meta_amount:.2f}$ ≈ {meta_qty:.2f} actions), Error = {error_meta:.2f}")
        print(f"  Arb Signal = {arb_signal}, Corr = {corr_dynamic:.2f}, Beta = {beta:.2f}, PnL = {daily_pnl:.2f}")
        print(f"  Temps d'exécution: {time.time() - iter_start:.3f} sec")
        print("-" * 50)

portfolio_series = pd.Series(portfolio, index=prices.index[:len(portfolio)])
sharpe_ratio = (portfolio_series.pct_change().mean() / portfolio_series.pct_change().std()) * np.sqrt(390)
rendement = (capital - 100) / 100 * 100
volatilite = portfolio_series.pct_change().std() * np.sqrt(390) * 100
drawdown_max = (portfolio_series / portfolio_series.cummax() - 1).min() * 100

print(f"Capital final : {capital:.2f}")
print(f"Rendement total : {rendement:.2f}%")
print(f"Volatilité (annualisée intraday) : {volatilite:.2f}%")
print(f"Ratio de Sharpe (intraday) : {sharpe_ratio:.2f}")
print(f"Drawdown maximum : {drawdown_max:.2f}%")
print(f"Temps total d'exécution : {time.time() - start_time:.2f} sec")

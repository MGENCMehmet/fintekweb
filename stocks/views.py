from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import plotly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "V", "JNJ",
    "WMT", "JPM", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "VZ", "ADBE",
    "NFLX", "CMCSA", "KO", "MRK", "PFE", "ABT", "NKE", "PEP", "INTC", "CSCO",
    "XOM", "CVX", "ORCL", "CRM", "ACN", "COST", "AMD", "QCOM", "TXN", "HON",
    "LLY", "MCD", "AMGN", "IBM", "GS", "WFC", "T", "MDT", "BMY", "CAT",
    "GE", "MMM", "BA", "NOW", "SPGI", "NEE", "DHR", "LMT", "BLK", "SCHW",
    "TMO", "ISRG", "AXP", "SYK", "C", "PM", "UNP", "LOW", "GILD", "PLD",
    "USB", "CB", "CCI", "CL", "ZTS", "CI", "MO", "ICE", "BKNG", "ADI",
    "SCHD", "ADP", "SO", "DUK", "NSC", "MMC", "BSX", "ITW", "VRTX", "PGR",
    "HUM", "MDLZ", "PNC", "EL", "BDX", "ECL", "EW", "KMB", "AON", "MRNA",
    "AKBNK.IS", "GARAN.IS", "SAHOL.IS", "THYAO.IS", "ASELS.IS",
    "BIMAS.IS", "TCELL.IS", "EREGL.IS", "TKFEN.IS"
]


def home(request):
    return render(request, 'stocks/home.html')


def stock_prices(request):
    context = {
        'tickers': TICKERS
    }
    return render(request, 'stocks/stock_prices.html', context)


def predictions(request):
    context = {
        'tickers': TICKERS
    }
    return render(request, 'stocks/predictions.html', context)


@csrf_exempt
def get_stock_data(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        tickers = data.get('tickers', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        if not tickers:
            return JsonResponse({'error': 'En az bir hisse seçiniz'})

        try:
            fig = go.Figure()

            for ticker in tickers:
                df = yf.download(ticker, start=start_date, end=end_date)
                #print(f"Veri çekildi, satır sayısı: {len(df)}")
                #print(df.head())
                data = yf.download(ticker, start='2023-01-01', end='2023-06-01')
                #print(data)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if df.empty or len(df) < 2:
                    return JsonResponse({'error': f'{ticker} için yeterli veri yok. Daha uzun tarih aralığı seçiniz.'})

                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Open'],
                    mode='lines',
                    name=f'{ticker} Open',
                    line=dict(width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['High'],
                    mode='lines',
                    name=f'{ticker} High',
                    line=dict(width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Low'],
                    mode='lines',
                    name=f'{ticker} Low',
                    line=dict(width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name=f'{ticker} Close',
                    line=dict(width=3)
                ))

            fig.update_layout(
                title=f'{", ".join(tickers)} Fiyat Değerleri',
                xaxis_title='Tarih',
                yaxis_title='Fiyat',
                height=600,
                hovermode='x unified',
                template='plotly_white'
            )

            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
            return JsonResponse({'graph': graph_json})

        except Exception as e:
            return JsonResponse({'error': f'Veri alınırken hata: {str(e)}'})

    return JsonResponse({'error': 'Geçersiz istek'})


@csrf_exempt
def predict_stock(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        ticker = data.get('ticker')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        if not ticker:
            return JsonResponse({'error': 'Bir hisse seçiniz'})

        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 10:
                return JsonResponse({'error': 'Seçilen tarih aralığında yeterli veri yok. Daha uzun tarih aralığı seçiniz.'})

            close_prices = df['Close']
            df_values = close_prices.values.reshape(-1, 1)
            df_train_len = int(np.ceil(len(df_values) * 0.95))

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_values)

            train_data = scaled_data[0:df_train_len, :]

            if len(train_data) < 4:
                return JsonResponse({'error': 'Yeterli eğitim verisi yok, daha uzun tarih aralığı seçiniz.'})

            x_train = []
            y_train = []

            for i in range(3, len(train_data)):
                x_train.append(train_data[i-3:i, 0])
                y_train.append(train_data[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

            test_data = scaled_data[df_train_len - 3:, :]

            if len(test_data) < 4:
                return JsonResponse({'error': 'Yeterli test verisi yok, daha uzun tarih aralığı seçiniz.'})

            x_test = []
            y_test = df_values[df_train_len:, :]

            for i in range(3, len(test_data)):
                x_test.append(test_data[i-3:i, 0])

            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            predictions = model.predict(x_test, verbose=0)
            predictions = scaler.inverse_transform(predictions)

            train = close_prices[:df_train_len]
            valid = close_prices[df_train_len:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train.index,
                y=train.values,
                mode='lines',
                name='Eğitim Verisi',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=valid.index,
                y=valid.values,
                mode='lines',
                name='Gerçek Değerler',
                line=dict(color='orange', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=valid.index,
                y=predictions.flatten(),
                mode='lines',
                name='Tahminler',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                title=f'{ticker} için Fiyat Tahmini',
                xaxis_title='Tarih',
                yaxis_title='Fiyat',
                height=600,
                hovermode='x unified',
                template='plotly_white'
            )

            graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)

            mse = np.mean((valid.values - predictions.flatten()) ** 2)
            rmse = np.sqrt(mse)

            return JsonResponse({
                'graph': graph_json,
                'rmse': rmse,
                'message': 'Tahmin başarıyla tamamlandı!'
            })

        except Exception as e:
            return JsonResponse({'error': f'Tahmin yapılırken hata: {str(e)}'})

    return JsonResponse({'error': 'Geçersiz istek'})

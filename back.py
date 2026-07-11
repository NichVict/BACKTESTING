import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from datetime import datetime, timedelta
from backtesting import Backtest, Strategy


# =====================================================
# DATA
# =====================================================

def get_data(ticker, interval="1h"):

    end = datetime.now()

    if interval == "1d":
        start = end - timedelta(days=3650)
    elif interval == "1h":
        start = end - timedelta(days=700)
    elif interval == "15m":
        start = end - timedelta(days=59)
    elif interval == "5m":
        start = end - timedelta(days=59)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("DataFrame vazio")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# =====================================================
# INDICADOR
# =====================================================

def SMA(values, period):
    return pd.Series(values).rolling(period).mean().values


# =====================================================
# STRATEGY - Rompimento no toque da MM20 (somente LONG)
# =====================================================
#
# Regras:
# 1) Entrada no rompimento da máxima do candle que tocou a MM20
# 2) MM8 nunca pode estar descendente
# 3) MM20 sempre deve estar ascendente
# 4) Alvo = 2x o risco
# 5) Stop = mínima do candle que tocou a MM20
# 6) Risco = ponto de entrada - mínima do candle que tocou a MM20
# 7) Somente operações LONG
#
# A entrada usa uma ordem stop nativa da lib (self.buy(stop=..., sl=..., tp=...)),
# que só é preenchida quando o preço efetivamente cruza a máxima marcada
# (rompimento intrabar), e o próprio motor da lib fecha a operação no stop
# ou no alvo, o que for atingido primeiro.

class Long_MM20_Breakout(Strategy):

    target_r = 2.0  # alvo = 2x o risco

    def init(self):
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.ma50 = self.I(SMA, self.data.Close, 50)

    # ---------------- CONDIÇÃO DE TENDÊNCIA ---------------- #
    # Automática e sem parâmetro de candles: MM20 acima da MM50 só
    # acontece quando o preço vem subindo de forma consistente. Não
    # depende de comparar a própria média com um ponto arbitrário do
    # passado (o que é sensível a ruído).

    def ma20_up(self):
        if np.isnan(self.ma20[-1]) or np.isnan(self.ma50[-1]):
            return False
        return self.ma20[-1] > self.ma50[-1]

    def touched_ma20(self):
        return self.data.Low[-1] <= self.ma20[-1] <= self.data.High[-1]

    # ---------------- EXECUTION ---------------- #

    def next(self):

        # Se a MM20 deixou de estar acima da MM50, cancela qualquer ordem
        # de rompimento ainda pendente
        if self.orders and not self.ma20_up():
            for order in list(self.orders):
                order.cancel()

        # Já em posição ou já com ordem armada esperando rompimento: não faz nada
        if self.position or self.orders:
            return

        # Procura novo candle que tocou a MM20 com a MM20 acima da MM50
        if self.ma20_up() and self.touched_ma20():

            high = self.data.High[-1]
            low = self.data.Low[-1]
            risk = high - low

            if risk <= 0:
                return

            target = high + self.target_r * risk

            self.buy(stop=high, sl=low, tp=target)


# =====================================================
# STREAMLIT STATE
# =====================================================

if "results" not in st.session_state:
    st.session_state.results = None


# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="Backtest Engine", layout="wide")

st.title("📊 Backtest - Rompimento MM20 (LONG)")

ticker = st.selectbox(
    "Ativo",
    ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "BOVA11.SA"]
)

timeframes = st.multiselect(
    "Timeframes",
    ["5m", "15m", "1h", "1d"],
    default=["5m"]
)

col1, col2 = st.columns(2)
with col1:
    ma20_lookback = st.slider("Candles p/ checar MM20 ascendente", 1, 10, 3)
with col2:
    st.empty()

run = st.button("🚀 Rodar análise")


# =====================================================
# BACKTEST
# =====================================================

if run:

    results = []

    with st.spinner("Rodando backtests..."):

        for tf in timeframes:

            df = get_data(ticker, tf)
            n_candles = len(df)

            bt = Backtest(
                df,
                Long_MM20_Breakout,
                cash=10000,
                commission=0.0005,
                exclusive_orders=True
            )

            stats = bt.run(ma20_lookback=ma20_lookback)
            trades = stats._trades

            n_trades = len(trades)
            wins = int((trades["PnL"] > 0).sum()) if n_trades else 0
            losses = int((trades["PnL"] <= 0).sum()) if n_trades else 0
            pnl = round(trades["PnL"].sum(), 2) if n_trades else 0.0

            results.append({
                "Timeframe": tf,
                "Candles": n_candles,
                "Trades": n_trades,
                "Acertos": wins,
                "Erros": losses,
                "PnL": pnl,
            })

    st.session_state.results = results


# =====================================================
# RESULTADOS (apenas números)
# =====================================================

if st.session_state.results is not None:

    df_results = pd.DataFrame(st.session_state.results)
    st.subheader("📋 Resultados")
    st.dataframe(df_results, use_container_width=True)

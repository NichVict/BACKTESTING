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

    if interval == "1h":
        start = end - timedelta(days=700)
    else:
        start = datetime(2018, 1, 1)

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

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    return df


# =====================================================
# INDICADOR
# =====================================================

def SMA(values, period):
    return pd.Series(values).rolling(period).mean().values


# =====================================================
# STRATEGY
# =====================================================

class MA_EntryEngine_V7(Strategy):

    def init(self):
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.ma50 = self.I(SMA, self.data.Close, 50)
        self.ma200 = self.I(SMA, self.data.Close, 200)

        self.entry_price = None
        self.stop = None
        self.partial_taken = False

    def trend_up(self):
        price = self.data.Close[-1]

        return (
            self.ma20[-1] > self.ma50[-1] > self.ma200[-1]
            and self.ma20[-1] > self.ma20[-5]
            and self.ma50[-1] > self.ma50[-5]
            and (self.ma20[-1] - self.ma50[-1]) / price > 0.005
            and price > self.data.Close[-10]
        )

    def trend_down(self):
        price = self.data.Close[-1]

        return (
            self.ma20[-1] < self.ma50[-1] < self.ma200[-1]
            and self.ma20[-1] < self.ma20[-5]
            and self.ma50[-1] < self.ma50[-5]
            and (self.ma50[-1] - self.ma20[-1]) / price > 0.005
            and price < self.data.Close[-10]
        )

    def pullback_long(self):
        return self.data.Low[-1] <= self.ma20[-1] <= self.data.High[-1]

    def pullback_short(self):
        return self.data.Low[-1] <= self.ma20[-1] <= self.data.High[-1]

    def long_trigger(self):
        return self.data.Close[-1] > self.data.High[-2]

    def short_trigger(self):
        return self.data.Close[-1] < self.data.Low[-2]

    def next(self):

        price = self.data.Close[-1]

        if not self.position:

            if self.trend_up() and self.pullback_long() and self.long_trigger():
                self.buy()
                self.entry_price = price
                self.stop = self.data.Low[-2]
                self.partial_taken = False

            elif self.trend_down() and self.pullback_short() and self.short_trigger():
                self.sell()
                self.entry_price = price
                self.stop = self.data.High[-2]
                self.partial_taken = False

        else:

            if self.position.is_long:

                if price <= self.stop:
                    self.position.close()
                    return

                risk = self.entry_price - self.stop

                if not self.partial_taken and price >= self.entry_price + risk:
                    self.position.close(0.5)
                    self.partial_taken = True

                if price >= self.entry_price + 2 * risk:
                    self.position.close()

            else:

                if price >= self.stop:
                    self.position.close()
                    return

                risk = self.stop - self.entry_price

                if not self.partial_taken and price <= self.entry_price - risk:
                    self.position.close(0.5)
                    self.partial_taken = True

                if price <= self.entry_price - 2 * risk:
                    self.position.close()


# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(page_title="Backtest Engine", layout="wide")

st.title("📊 Backtest MA Engine V7")

tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "BOVA11.SA"]

ticker = st.selectbox("Escolha o ativo", tickers)
interval = st.selectbox("Timeframe", ["1h", "15m"])

run = st.button("🚀 Rodar Backtest")

if run:

    with st.spinner("Rodando backtest..."):

        df = get_data(ticker, interval)

        bt = Backtest(
            df,
            MA_EntryEngine_V7,
            cash=10000,
            commission=0.0005,
            trade_on_close=True,
            exclusive_orders=True
        )

        stats = bt.run()
        trades = stats._trades.copy()

    # =================================================
    # MÉTRICAS
    # =================================================

    st.subheader("📈 Estatísticas")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Trades", len(trades))
    col2.metric("Win Rate", f"{(trades[trades['PnL'] > 0].shape[0] / len(trades) * 100) if len(trades) > 0 else 0:.2f}%")
    col3.metric("PnL Total", f"{trades['PnL'].sum():.2f}" if len(trades) > 0 else "0")

    # =================================================
    # DATAFRAME BONITO
    # =================================================

    st.subheader("📋 Trades")

    if len(trades) > 0:

        styled = trades.copy()
        styled["Side"] = np.where(styled["Size"] > 0, "LONG", "SHORT")

        st.dataframe(
            styled,
            use_container_width=True
        )

        # =================================================
        # EQUITY CURVE
        # =================================================

        st.subheader("📉 Equity Curve")

        equity = stats._equity_curve

        st.line_chart(equity["Equity"])

    else:
        st.warning("Nenhum trade executado.")

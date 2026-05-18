import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from datetime import datetime, timedelta
from backtesting import Backtest, Strategy


# =====================================================
# DATA
# =====================================================

def get_data(ticker, interval="1d"):

    end = datetime.now()

    if interval == "1d":
        start = end - timedelta(days=3650)

    elif interval == "1h":
        start = end - timedelta(days=700)

    elif interval == "15m":
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
# STRATEGY
# =====================================================

class MA_EntryEngine_V7(Strategy):

    def init(self):

        self.tf = self.data.df.attrs.get("tf", "1h")

        # ajuste dinâmico de médias
        if self.tf == "15m":
            self.ma20 = self.I(SMA, self.data.Close, 10)
            self.ma50 = self.I(SMA, self.data.Close, 30)
            self.ma200 = None  # desliga

        elif self.tf == "1h":
            self.ma20 = self.I(SMA, self.data.Close, 20)
            self.ma50 = self.I(SMA, self.data.Close, 50)
            self.ma200 = self.I(SMA, self.data.Close, 200)

        else:  # 1d
            self.ma20 = self.I(SMA, self.data.Close, 20)
            self.ma50 = self.I(SMA, self.data.Close, 50)
            self.ma200 = self.I(SMA, self.data.Close, 200)

        self.entry_price = None
        self.stop = None
        self.partial_taken = False

    def trend_up(self):
    
        price = self.data.Close[-1]
    
        if self.ma200 is None:
            return (
                self.ma20[-1] > self.ma50[-1]
                and self.ma20[-1] > self.ma20[-3]
                and price > self.data.Close[-8]
            )
    
        return (
            self.ma20[-1] > self.ma50[-1] > self.ma200[-1]
            and self.ma20[-1] > self.ma20[-5]
            and self.ma50[-1] > self.ma50[-5]
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

st.title("📊 Multi-Timeframe Backtest Engine")

ticker = st.selectbox(
    "Ativo",
    ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "BOVA11.SA"]
)

timeframes = st.multiselect(
    "Timeframes",
    ["15m", "1h", "1d"],
    default=["1h", "1d"]
)

run = st.button("🚀 Rodar análise comparativa")


# =====================================================
# EXECUÇÃO
# =====================================================

if run:

    results = []          # resumo comparativo
    all_trades = {}       # trades por timeframe
    equity_curves = {}    # equity por timeframe

    with st.spinner("Rodando backtests em múltiplos timeframes..."):

        for tf in timeframes:

            # -------------------------------
            # DATA
            # -------------------------------
            df = get_data(ticker, tf)

            # importante para estratégia adaptativa (se você usar)
            df.attrs["tf"] = tf

            # -------------------------------
            # BACKTEST
            # -------------------------------
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

            # -------------------------------
            # MÉTRICAS
            # -------------------------------
            winrate = (
                (trades["PnL"] > 0).sum() / len(trades) * 100
                if len(trades) > 0 else 0
            )

            pnl_total = trades["PnL"].sum() if len(trades) > 0 else 0

            results.append({
                "Timeframe": tf,
                "Trades": len(trades),
                "WinRate (%)": round(winrate, 2),
                "PnL Total": round(pnl_total, 2)
            })

            all_trades[tf] = trades
            equity_curves[tf] = stats._equity_curve["Equity"]


    # =====================================================
    # RESUMO COMPARATIVO
    # =====================================================

    st.subheader("📊 Comparação de Timeframes")

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)


    # =====================================================
    # DETALHE DO MELHOR TIMEFRAME
    # =====================================================

    best_tf = results_df.sort_values("PnL Total", ascending=False).iloc[0]["Timeframe"]

    st.success(f"🏆 Melhor timeframe: {best_tf}")


    # =====================================================
    # DETALHES POR TIMEFRAME
    # =====================================================

    st.subheader("📋 Trades por Timeframe")

    selected_tf = st.selectbox(
        "Ver trades do timeframe:",
        timeframes
    )

    trades = all_trades[selected_tf]

    if len(trades) > 0:

        styled = trades.copy()
        styled["Side"] = np.where(styled["Size"] > 0, "LONG", "SHORT")

        st.dataframe(styled, use_container_width=True)

        st.subheader("📉 Equity Curve")

        st.line_chart(equity_curves[selected_tf])

    else:
        st.warning(f"Nenhum trade executado no TF {selected_tf}")

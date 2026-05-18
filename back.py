import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from datetime import datetime, timedelta
from backtesting import Backtest, Strategy

import plotly.graph_objects as go


# =====================================================
# DATA
# =====================================================

def get_data(ticker, interval="1h", days=365):

    end = datetime.now()

    # =====================================================
    # LIMITES REAIS DO YAHOO FINANCE
    # =====================================================

    limits = {
        "15m": 59,
        "1h": 700,
        "1d": 3650
    }

    max_days = limits.get(interval, 3650)

    # garante que nunca pede mais do que o Yahoo permite
    final_days = min(days, max_days)

    start = end - timedelta(days=final_days)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()  # não quebra o app

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

        self.tf = getattr(self.data, "tf", "1h")

        if self.tf == "15m":
            self.ma20 = self.I(SMA, self.data.Close, 10)
            self.ma50 = self.I(SMA, self.data.Close, 30)
            self.ma200 = None
        else:
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

        if self.ma200 is None:
            return (
                self.ma20[-1] < self.ma50[-1]
                and self.ma20[-1] < self.ma20[-3]
                and price < self.data.Close[-8]
            )

        return (
            self.ma20[-1] < self.ma50[-1] < self.ma200[-1]
            and self.ma20[-1] < self.ma20[-5]
            and self.ma50[-1] < self.ma50[-5]
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
                self.stop = self.data.Low[-2]
                self.buy(sl=self.stop)
                self.entry_price = price
                self.partial_taken = False

            elif self.trend_down() and self.pullback_short() and self.short_trigger():
                self.stop = self.data.High[-2]
                self.sell(sl=self.stop)
                self.entry_price = price
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
# STREAMLIT STATE
# =====================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "all_trades" not in st.session_state:
    st.session_state.all_trades = None

if "equity_curves" not in st.session_state:
    st.session_state.equity_curves = None

if "price_data" not in st.session_state:
    st.session_state.price_data = None


# =====================================================
# UI
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

backtest_period = st.selectbox(
    "Período do Backtest",
    ["5 anos", "4 anos", "3 anos", "2 anos", "1 ano", "6 meses"],
    index=2
)

trade_filter = st.selectbox(
    "Operações",
    ["Todas", "Long", "Short"],
    index=0
)

period_map = {
    "5 anos": 365 * 5,
    "4 anos": 365 * 4,
    "3 anos": 365 * 3,
    "2 anos": 365 * 2,
    "1 ano": 365,
    "6 meses": 180
}

backtest_days = period_map[backtest_period]

run = st.button("🚀 Rodar análise")


# =====================================================
# BACKTEST
# =====================================================

if run:

    results = []
    all_trades = {}
    equity_curves = {}
    price_data = {}

    with st.spinner("Rodando backtests..."):

        for tf in timeframes:

            df = get_data(ticker, tf, days=backtest_days)

            if df.empty:
                st.warning(f"Sem dados para {tf} no período selecionado")
                continue
            df.tf = tf

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
            equity = stats._equity_curve["Equity"]

            pnl = trades["PnL"].sum() if len(trades) else 0
            winrate = (trades["PnL"] > 0).mean() * 100 if len(trades) else 0

            gross_profit = trades.loc[trades["PnL"] > 0, "PnL"].sum()
            gross_loss = abs(trades.loc[trades["PnL"] < 0, "PnL"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

            expectancy = trades["PnL"].mean() if len(trades) else 0

            returns = equity.pct_change().dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() != 0 else 0

            peak = equity.cummax()
            drawdown = equity / peak - 1
            max_dd = drawdown.min()

            results.append({
                "Timeframe": tf,
                "Trades": len(trades),
                "WinRate (%)": round(winrate, 2),
                "PnL": round(pnl, 2),
                "Sharpe": round(sharpe, 2),
                "ProfitFactor": round(profit_factor, 2),
                "Expectancy": round(expectancy, 2),
                "MaxDD (%)": round(max_dd * 100, 2)
            })

            all_trades[tf] = trades
            equity_curves[tf] = equity
            price_data[tf] = df

    st.session_state.results = results
    st.session_state.all_trades = all_trades
    st.session_state.equity_curves = equity_curves
    st.session_state.price_data = price_data


# =====================================================
# RESULTS
# =====================================================

if st.session_state.results is not None:

    df_results = pd.DataFrame(st.session_state.results)

    st.subheader("📊 Comparação de Timeframes")
    st.dataframe(df_results, use_container_width=True)

    best = df_results.sort_values("PnL", ascending=False).iloc[0]["Timeframe"]
    st.success(f"🏆 Melhor timeframe: {best}")

    selected = st.selectbox(
        "Ver detalhes do TF",
        df_results["Timeframe"].tolist()
    )

    trades = st.session_state.all_trades[selected].copy()

    if len(trades) > 0:

        if trade_filter == "Long":
            trades = trades[trades["Size"] > 0]
        elif trade_filter == "Short":
            trades = trades[trades["Size"] < 0]

        equity = st.session_state.equity_curves[selected]
        df_price = st.session_state.price_data[selected]

        styled = trades.copy()
        styled["Side"] = np.where(styled["Size"] > 0, "LONG", "SHORT")
        styled = styled.fillna("—")

        st.subheader("📋 Trades")
        st.dataframe(styled, use_container_width=True)

        peak = equity.cummax()
        drawdown = equity / peak - 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=equity, name="Equity"))
        fig.add_trace(go.Scatter(y=drawdown, name="Drawdown", yaxis="y2"))

        fig.update_layout(
            title="Equity + Drawdown",
            yaxis=dict(title="Equity"),
            yaxis2=dict(title="Drawdown", overlaying="y", side="right"),
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=trades["PnL"], nbinsx=30))
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(y=df_price["Close"], name="Preço"))

        buys = trades[trades["Size"] > 0]
        sells = trades[trades["Size"] < 0]

        fig3.add_trace(go.Scatter(
            x=buys.index,
            y=buys["EntryPrice"],
            mode="markers",
            name="Long Entries"
        ))

        fig3.add_trace(go.Scatter(
            x=sells.index,
            y=sells["EntryPrice"],
            mode="markers",
            name="Short Entries"
        ))

        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("Nenhum trade nesse timeframe.")

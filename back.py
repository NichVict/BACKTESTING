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

def get_data(ticker, interval="1h"):

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
# INDICADORES
# =====================================================

def SMA(values, period):
    return pd.Series(values).rolling(period).mean().values


# =====================================================
# STRATEGY 1 - PULLBACK MULTI-MA (original)
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

    # ---------------- TREND ---------------- #

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

    # ---------------- ENTRY ---------------- #

    def pullback_long(self):
        return self.data.Low[-1] <= self.ma20[-1] <= self.data.High[-1]

    def pullback_short(self):
        return self.data.Low[-1] <= self.ma20[-1] <= self.data.High[-1]

    def long_trigger(self):
        return self.data.Close[-1] > self.data.High[-2]

    def short_trigger(self):
        return self.data.Close[-1] < self.data.Low[-2]

    # ---------------- EXECUTION ---------------- #

    def next(self):

        price = self.data.Close[-1]

        if not self.position:

            # LONG
            if self.trend_up() and self.pullback_long() and self.long_trigger():

                self.stop = self.data.Low[-2]
                self.buy(sl=self.stop)

                self.entry_price = price
                self.partial_taken = False

            # SHORT
            elif self.trend_down() and self.pullback_short() and self.short_trigger():

                self.stop = self.data.High[-2]
                self.sell(sl=self.stop)

                self.entry_price = price
                self.partial_taken = False

        else:

            # LONG
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

            # SHORT
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
# STRATEGY 2 - PC (PONTO CONTÍNUO) - Stormer / Palex
# =====================================================
#
# Regras:
# 1) Tendência definida pela inclinação da MA (default 21 periodos)
# 2) Aguarda o candle "tocar" a MA (min <= MA <= max) sem a MA mudar de direção
# 3) Marca a máxima/mínima do candle que tocou
# 4) Fica "armado" enquanto a MA não reverter, aguardando rompimento
# 5) Entrada no rompimento da máxima (compra) ou mínima (venda) do candle marcado
# 6) Stop na mínima/máxima do próprio candle marcado
# 7) Gestão: parcial em 1R + stop total em 2R (mesmo padrão do V7, pode trocar
#    por "alvo = amplitude do candle projetada" se preferir o critério original
#    do Stormer)
#
# Parâmetros expostos como atributos de classe (podem ser sobrescritos via
# bt.run(ma_period=..., use_volume_filter=...) na Streamlit UI)

class PC_Setup(Strategy):

    ma_period = 21
    ma_lookback = 5          # candles atrás p/ checar inclinação da MA
    use_volume_filter = False
    volume_ma_period = 20

    def init(self):

        self.tf = getattr(self.data, "tf", "1h")

        period = self.ma_period
        if self.tf == "15m" and self.ma_period == 21:
            # em TF menor, reduz um pouco o período p/ não ficar "lento" demais
            period = 14

        self.ma = self.I(SMA, self.data.Close, period)

        if self.use_volume_filter:
            self.vol_ma = self.I(SMA, self.data.Volume, self.volume_ma_period)
        else:
            self.vol_ma = None

        # estado do sinal armado (aguardando rompimento)
        self.pending_dir = None      # 'long' ou 'short'
        self.pending_high = None
        self.pending_low = None

        self.entry_price = None
        self.stop = None
        self.partial_taken = False

    # ---------------- TREND ---------------- #

    def ma_up(self):
        lb = self.ma_lookback
        if len(self.ma) <= lb or np.isnan(self.ma[-lb]):
            return False
        return self.ma[-1] > self.ma[-lb]

    def ma_down(self):
        lb = self.ma_lookback
        if len(self.ma) <= lb or np.isnan(self.ma[-lb]):
            return False
        return self.ma[-1] < self.ma[-lb]

    # ---------------- TOUCH ---------------- #

    def touched_ma(self):
        return self.data.Low[-1] <= self.ma[-1] <= self.data.High[-1]

    # ---------------- VOLUME FILTER ---------------- #

    def volume_ok(self):
        if not self.use_volume_filter or self.vol_ma is None:
            return True
        if np.isnan(self.vol_ma[-1]):
            return True
        return self.data.Volume[-1] > self.vol_ma[-1]

    # ---------------- EXECUTION ---------------- #

    def next(self):

        price = self.data.Close[-1]

        if not self.position:

            # ---- Já existe sinal armado: checa cancelamento / rompimento ---- #
            if self.pending_dir == "long":

                if not self.ma_up():
                    self.pending_dir = None  # MA reverteu, cancela

                elif self.data.Close[-1] > self.pending_high and self.volume_ok():
                    self.stop = self.pending_low
                    self.buy(sl=self.stop)
                    self.entry_price = price
                    self.partial_taken = False
                    self.pending_dir = None

            elif self.pending_dir == "short":

                if not self.ma_down():
                    self.pending_dir = None

                elif self.data.Close[-1] < self.pending_low and self.volume_ok():
                    self.stop = self.pending_high
                    self.sell(sl=self.stop)
                    self.entry_price = price
                    self.partial_taken = False
                    self.pending_dir = None

            # ---- Nenhum sinal armado: procura novo toque na MA ---- #
            else:

                if self.ma_up() and self.touched_ma():
                    self.pending_dir = "long"
                    self.pending_high = self.data.High[-1]
                    self.pending_low = self.data.Low[-1]

                elif self.ma_down() and self.touched_ma():
                    self.pending_dir = "short"
                    self.pending_high = self.data.High[-1]
                    self.pending_low = self.data.Low[-1]

        else:

            # ---- Gestão da posição aberta (parcial 1R + full 2R) ---- #

            if self.position.is_long:

                if price <= self.stop:
                    self.position.close()
                    return

                risk = self.entry_price - self.stop
                if risk <= 0:
                    return

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
                if risk <= 0:
                    return

                if not self.partial_taken and price <= self.entry_price - risk:
                    self.position.close(0.5)
                    self.partial_taken = True

                if price <= self.entry_price - 2 * risk:
                    self.position.close()


STRATEGIES = {
    "Pullback Multi-MA (V7)": MA_EntryEngine_V7,
    "PC - Ponto Contínuo (Stormer/Palex)": PC_Setup,
}


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

setup_name = st.selectbox(
    "Setup",
    list(STRATEGIES.keys())
)

strategy_kwargs = {}

if setup_name.startswith("PC"):
    col1, col2 = st.columns(2)
    with col1:
        ma_period = st.selectbox("Período da MA (PC)", [9, 20, 21], index=2)
    with col2:
        use_volume_filter = st.checkbox("Filtrar por volume no rompimento", value=False)

    strategy_kwargs["ma_period"] = ma_period
    strategy_kwargs["use_volume_filter"] = use_volume_filter

run = st.button("🚀 Rodar análise")


# =====================================================
# BACKTEST
# =====================================================

if run:

    results = []
    all_trades = {}
    equity_curves = {}
    price_data = {}

    strategy_cls = STRATEGIES[setup_name]

    with st.spinner("Rodando backtests..."):

        for tf in timeframes:

            df = get_data(ticker, tf)
            df.tf = tf

            bt = Backtest(
                df,
                strategy_cls,
                cash=10000,
                commission=0.0005,
                trade_on_close=True,
                exclusive_orders=True
            )

            stats = bt.run(**strategy_kwargs)
            trades = stats._trades.copy()

            equity = stats._equity_curve["Equity"]

            # ================= METRICS ================= #

            pnl = trades["PnL"].sum() if len(trades) else 0

            winrate = (trades["PnL"] > 0).mean() * 100 if len(trades) else 0

            # Profit Factor
            gross_profit = trades.loc[trades["PnL"] > 0, "PnL"].sum()
            gross_loss = abs(trades.loc[trades["PnL"] < 0, "PnL"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

            # Expectancy
            expectancy = trades["PnL"].mean() if len(trades) else 0

            # Returns
            returns = equity.pct_change().dropna()

            sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() != 0 else 0

            # Drawdown
            peak = equity.cummax()
            drawdown = (equity / peak - 1)
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

    trades = st.session_state.all_trades[selected]
    equity = st.session_state.equity_curves[selected]
    df_price = st.session_state.price_data[selected]

    # ================= TRADES ================= #

    if len(trades) > 0:

        styled = trades.copy()
        styled["Side"] = np.where(styled["Size"] > 0, "LONG", "SHORT")

        styled = styled.fillna("—")

        st.subheader("📋 Trades")
        st.dataframe(styled, use_container_width=True)

        # ================= EQUITY + DRAWDOWN ================= #

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

        # ================= PNL DISTRIBUTION ================= #

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=trades["PnL"], nbinsx=30))

        fig2.update_layout(
            title="Distribuição de PnL por Trade",
            template="plotly_dark"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ================= PRICE + ENTRIES ================= #

        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            y=df_price["Close"],
            name="Preço"
        ))

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

        fig3.update_layout(
            title="Preço + Entradas",
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("Nenhum trade nesse timeframe.")

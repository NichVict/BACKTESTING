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
# INDICADORES
# =====================================================

def SMA(values, period):
    return pd.Series(values).rolling(period).mean().values


def ATR(high, low, close, period):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean().values


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
# STRATEGY 2 - SETUP PRÓPRIO (alinhamento MM8/20/50/200 + MM20)
# =====================================================
#
# Regras (conforme definido pelo usuário):
# 1) Alinhamento das médias:
#      Alta:  MA200 < MA50 < MA20 < MA8
#      Baixa: MA200 > MA50 > MA20 > MA8
# 2) Tendência clara: inclinação das médias (MA20 subindo/descendo) e/ou
#    estrutura de topos e fundos do preço (higher-highs/higher-lows para
#    alta, lower-highs/lower-lows para baixa) — filtro de estrutura é
#    opcional/aproximado, ligado via checkbox na UI.
# 3) Entrada: rompimento da máxima (compra) ou mínima (venda) do candle que
#    tocou a MA20, respeitando o alinhamento.
# 4) Stop: mínima (compra) ou máxima (venda) do candle que tocou a MA20.
# 5) Alvo: 2R e 3R (R = distância entre a mínima do candle de toque e o
#    preço de entrada) — parcial em 2R, restante corre até 3R OU até romper
#    um stop móvel baseado em ATR (chandelier exit), o que vier primeiro.
#
# Parâmetros expostos como atributos de classe (sobrescrevíveis via
# bt.run(**kwargs) na UI)

class MA_Alignment_Setup(Strategy):

    atr_period = 14
    atr_mult = 3.0
    ma_lookback = 5              # candles atrás p/ checar inclinação da MA20
    use_structure_filter = False # filtro extra de topos/fundos (aproximado)
    structure_window = 10        # tamanho da janela p/ comparar topos/fundos

    def init(self):

        self.tf = getattr(self.data, "tf", "1h")

        self.ma8 = self.I(SMA, self.data.Close, 8)
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.ma50 = self.I(SMA, self.data.Close, 50)
        self.ma200 = self.I(SMA, self.data.Close, 200)

        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)

        # estado do sinal armado (aguardando rompimento)
        self.pending_dir = None      # 'long' ou 'short'
        self.pending_high = None
        self.pending_low = None

        self.entry_price = None
        self.stop = None            # stop fixo inicial (mínima/máxima do candle de toque)
        self.trail_extreme = None   # maior alta (long) / menor baixa (short) desde a entrada
        self.partial_taken = False

    # ---------------- ALINHAMENTO DAS MÉDIAS ---------------- #

    def aligned_up(self):
        return self.ma200[-1] < self.ma50[-1] < self.ma20[-1] < self.ma8[-1]

    def aligned_down(self):
        return self.ma200[-1] > self.ma50[-1] > self.ma20[-1] > self.ma8[-1]

    # ---------------- INCLINAÇÃO ---------------- #

    def ma20_up(self):
        lb = self.ma_lookback
        if len(self.ma20) <= lb or np.isnan(self.ma20[-lb]):
            return False
        return self.ma20[-1] > self.ma20[-lb]

    def ma20_down(self):
        lb = self.ma_lookback
        if len(self.ma20) <= lb or np.isnan(self.ma20[-lb]):
            return False
        return self.ma20[-1] < self.ma20[-lb]

    # ---------------- ESTRUTURA DE TOPOS/FUNDOS (aproximada) ---------------- #

    def structure_up(self):
        if not self.use_structure_filter:
            return True

        w = self.structure_window
        if len(self.data.Close) < 2 * w:
            return False

        low_recent = min(self.data.Low[-w:])
        low_prior = min(self.data.Low[-2 * w:-w])
        return low_recent > low_prior  # fundo mais alto

    def structure_down(self):
        if not self.use_structure_filter:
            return True

        w = self.structure_window
        if len(self.data.Close) < 2 * w:
            return False

        high_recent = max(self.data.High[-w:])
        high_prior = max(self.data.High[-2 * w:-w])
        return high_recent < high_prior  # topo mais baixo

    def trend_up(self):
        return self.aligned_up() and self.ma20_up() and self.structure_up()

    def trend_down(self):
        return self.aligned_down() and self.ma20_down() and self.structure_down()

    # ---------------- TOQUE NA MM20 ---------------- #

    def touched_ma20(self):
        return self.data.Low[-1] <= self.ma20[-1] <= self.data.High[-1]

    # ---------------- EXECUTION ---------------- #

    def next(self):

        price = self.data.Close[-1]

        if not self.position:

            # ---- Sinal já armado: checa cancelamento / rompimento ---- #
            if self.pending_dir == "long":

                if not self.trend_up():
                    self.pending_dir = None  # perdeu alinhamento/tendência, cancela

                elif self.data.Close[-1] > self.pending_high:
                    self.stop = self.pending_low
                    self.buy(sl=self.stop)
                    self.entry_price = price
                    self.trail_extreme = price
                    self.partial_taken = False
                    self.pending_dir = None

            elif self.pending_dir == "short":

                if not self.trend_down():
                    self.pending_dir = None

                elif self.data.Close[-1] < self.pending_low:
                    self.stop = self.pending_high
                    self.sell(sl=self.stop)
                    self.entry_price = price
                    self.trail_extreme = price
                    self.partial_taken = False
                    self.pending_dir = None

            # ---- Nenhum sinal armado: procura novo toque na MM20 ---- #
            else:

                if self.trend_up() and self.touched_ma20():
                    self.pending_dir = "long"
                    self.pending_high = self.data.High[-1]
                    self.pending_low = self.data.Low[-1]

                elif self.trend_down() and self.touched_ma20():
                    self.pending_dir = "short"
                    self.pending_high = self.data.High[-1]
                    self.pending_low = self.data.Low[-1]

        else:

            # ---- Gestão da posição aberta ---- #
            atr_val = self.atr[-1]
            has_atr = not np.isnan(atr_val)

            if self.position.is_long:

                self.trail_extreme = max(self.trail_extreme, self.data.High[-1])

                risk = self.entry_price - self.stop
                if risk <= 0:
                    return

                # stop móvel (chandelier exit) - só sobe, nunca desce
                effective_stop = self.stop
                if has_atr:
                    atr_stop = self.trail_extreme - self.atr_mult * atr_val
                    effective_stop = max(effective_stop, atr_stop)

                if price <= effective_stop:
                    self.position.close()
                    return

                if not self.partial_taken and price >= self.entry_price + 2 * risk:
                    self.position.close(0.5)
                    self.partial_taken = True

                if price >= self.entry_price + 3 * risk:
                    self.position.close()

            else:

                self.trail_extreme = min(self.trail_extreme, self.data.Low[-1])

                risk = self.stop - self.entry_price
                if risk <= 0:
                    return

                effective_stop = self.stop
                if has_atr:
                    atr_stop = self.trail_extreme + self.atr_mult * atr_val
                    effective_stop = min(effective_stop, atr_stop)

                if price >= effective_stop:
                    self.position.close()
                    return

                if not self.partial_taken and price <= self.entry_price - 2 * risk:
                    self.position.close(0.5)
                    self.partial_taken = True

                if price <= self.entry_price - 3 * risk:
                    self.position.close()


STRATEGIES = {
    "Pullback Multi-MA (V7)": MA_EntryEngine_V7,
    "Alinhamento MM8/20/50/200 + toque MM20": MA_Alignment_Setup,
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
    ["5m", "15m", "1h", "1d"],
    default=["5m"]
)

setup_name = st.selectbox(
    "Setup",
    list(STRATEGIES.keys())
)

strategy_kwargs = {}

if setup_name.startswith("Alinhamento"):
    col1, col2, col3 = st.columns(3)
    with col1:
        atr_mult = st.slider("Multiplicador ATR (trailing stop)", 1.0, 5.0, 3.0, 0.5)
    with col2:
        ma_lookback = st.selectbox("Candles p/ checar inclinação da MM20", [3, 5, 8, 10], index=1)
    with col3:
        use_structure_filter = st.checkbox("Exigir topos/fundos ascendentes/descendentes", value=False)

    strategy_kwargs["atr_mult"] = atr_mult
    strategy_kwargs["ma_lookback"] = ma_lookback
    strategy_kwargs["use_structure_filter"] = use_structure_filter

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

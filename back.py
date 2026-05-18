import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta


class BreakoutATRStrategy(Strategy):
    atr_period = 14
    channel_period = 20
    atr_mult = 2
    rr = 2

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.atr = self.I(
            lambda h, l, c: ta.volatility.AverageTrueRange(
                high=pd.Series(h),
                low=pd.Series(l),
                close=pd.Series(c),
                window=self.atr_period
            ).average_true_range().values,
            high, low, close
        )

        self.channel_high = self.I(
            lambda h: pd.Series(h).rolling(self.channel_period).max().values,
            high
        )

        self.channel_low = self.I(
            lambda l: pd.Series(l).rolling(self.channel_period).min().values,
            low
        )

    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]

        upper = self.channel_high[-2]
        lower = self.channel_low[-2]

        # sem posição
        if not self.position:

            # BUY breakout
            if price > upper:
                stop = price - atr * self.atr_mult
                tp = price + (price - stop) * self.rr

                self.buy(sl=stop, tp=tp)

            # SELL breakout
            elif price < lower:
                stop = price + atr * self.atr_mult
                tp = price - (stop - price) * self.rr

                self.sell(sl=stop, tp=tp)


# ===== DADOS =====
tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA"]

for t in tickers:
    print(f"\n=== {t} ===")

    df = yf.download(t, interval="5m", period="60d")
    df.dropna(inplace=True)

    bt = Backtest(
        df,
        BreakoutATRStrategy,
        cash=10_000,
        commission=0.001
    )

    stats = bt.run()
    print(stats)

    bt.plot()

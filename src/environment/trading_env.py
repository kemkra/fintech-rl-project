from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DATA_FILE = Path("data/processed/stock_features.csv")
DEFAULT_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA5",
    "MA20",
    "RSI",
    "MACD",
    "Volatility",
]


@dataclass
class TradingState:
    step: int
    cash: float
    shares: int
    portfolio_value: float


class TradingEnv:
    """
    A simple single-asset trading environment.

    Actions:
    0 = hold
    1 = buy with all available cash
    2 = sell all held shares
    """

    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        data,
        ticker=None,
        initial_cash=100_000.0,
        transaction_cost_pct=0.001,
        feature_columns=None,
    ):
        self.initial_cash = float(initial_cash)
        self.transaction_cost_pct = float(transaction_cost_pct)
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if self.transaction_cost_pct < 0:
            raise ValueError("transaction_cost_pct cannot be negative.")

        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
        self.data = self._prepare_data(data, ticker)

        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value = self.initial_cash

    def _prepare_data(self, data, ticker):
        df = data.copy()
        df.columns = [str(column).strip() for column in df.columns]

        required_columns = set(["Date", "Ticker", "Close"] + list(self.feature_columns))
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        if ticker is not None:
            df = df[df["Ticker"] == ticker].copy()
            if df.empty:
                raise ValueError(f"No data found for ticker: {ticker}")

        tickers = sorted(df["Ticker"].unique())
        if len(tickers) != 1:
            raise ValueError(
                "TradingEnv supports one ticker at a time. "
                f"Available tickers: {tickers}"
            )

        df["Date"] = pd.to_datetime(df["Date"])
        for column in self.feature_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=self.feature_columns)
        if df.empty:
            raise ValueError("No usable rows remain after cleaning environment data.")

        return df.sort_values("Date").reset_index(drop=True)

    @property
    def ticker(self):
        return self.data["Ticker"].iloc[0]

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value = self.initial_cash
        return self._get_observation()

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        if action not in [self.HOLD, self.BUY, self.SELL]:
            raise ValueError("Action must be 0 (hold), 1 (buy), or 2 (sell).")

        previous_value = self.portfolio_value
        current_price = self._current_price()

        if action == self.BUY:
            self._buy(current_price)
        elif action == self.SELL:
            self._sell(current_price)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        self.portfolio_value = self._calculate_portfolio_value()
        reward = self.portfolio_value - previous_value

        info = {
            "date": self.data.loc[self.current_step, "Date"],
            "ticker": self.ticker,
            "cash": self.cash,
            "shares": self.shares,
            "price": self._current_price(),
            "portfolio_value": self.portfolio_value,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        market_features = self.data.loc[self.current_step, self.feature_columns]
        portfolio_features = pd.Series(
            {
                "Cash": self.cash,
                "Shares": self.shares,
                "Portfolio_Value": self.portfolio_value,
            }
        )
        observation = pd.concat([market_features, portfolio_features])
        return observation.to_numpy(dtype=np.float32)

    def _current_price(self):
        return float(self.data.loc[self.current_step, "Close"])

    def _calculate_portfolio_value(self):
        return self.cash + self.shares * self._current_price()

    def _buy(self, price):
        if self.cash <= 0:
            return

        effective_price = price * (1 + self.transaction_cost_pct)
        shares_to_buy = int(self.cash // effective_price)

        if shares_to_buy <= 0:
            return

        total_cost = shares_to_buy * effective_price
        self.cash -= total_cost
        self.shares += shares_to_buy

    def _sell(self, price):
        if self.shares <= 0:
            return

        proceeds = self.shares * price * (1 - self.transaction_cost_pct)
        self.cash += proceeds
        self.shares = 0

    def get_state(self):
        return TradingState(
            step=self.current_step,
            cash=self.cash,
            shares=self.shares,
            portfolio_value=self.portfolio_value,
        )


def load_processed_data(file_path=PROCESSED_DATA_FILE):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Run src/features/feature_engineering.py first."
        )
    return pd.read_csv(file_path)


if __name__ == "__main__":
    data = load_processed_data()
    env = TradingEnv(data, ticker="AAPL")

    observation = env.reset()
    print(f"Ticker: {env.ticker}")
    print(f"Observation size: {len(observation)}")

    for action in [TradingEnv.BUY, TradingEnv.HOLD, TradingEnv.SELL]:
        _, reward, done, info = env.step(action)
        print(
            f"Action: {action}, Reward: {reward:.2f}, "
            f"Portfolio Value: {info['portfolio_value']:.2f}, Done: {done}"
        )

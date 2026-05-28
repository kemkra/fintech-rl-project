from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DATA_FILE = Path("data/processed/stock_features.csv")
DEFAULT_FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "Daily_Return",
    "MA5",
    "MA20",
    "RSI",
    "MACD",
    "Volatility",
]


@dataclass
class PortfolioState:
    step: int
    date: pd.Timestamp
    portfolio_value: float
    cash_weight: float
    asset_weights: np.ndarray
    turnover: float


class PortfolioEnv:
    """
    Multi-asset portfolio management environment.

    Action format:
    - A continuous vector of length n_assets + 1.
    - action[0] is cash preference.
    - action[1:] are asset preferences.
    - The vector is normalized into non-negative portfolio weights that sum to 1.

    Reward:
    - portfolio_return: next-step portfolio return after transaction cost.
    - log_return: log(1 + portfolio_return), optionally minus volatility penalty.
    """

    def __init__(
        self,
        data,
        tickers=None,
        initial_cash=100_000.0,
        transaction_cost_pct=0.001,
        feature_columns=None,
        reward_mode="log_return",
        risk_penalty=0.0,
        risk_window=20,
    ):
        self.initial_cash = float(initial_cash)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
        self.reward_mode = str(reward_mode)
        self.risk_penalty = float(risk_penalty)
        self.risk_window = int(risk_window)

        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if self.transaction_cost_pct < 0:
            raise ValueError("transaction_cost_pct cannot be negative.")
        if self.reward_mode not in {"portfolio_return", "log_return"}:
            raise ValueError("reward_mode must be one of: portfolio_return, log_return.")
        if self.risk_window <= 1:
            raise ValueError("risk_window must be greater than 1.")

        self.data, self.tickers, self.dates = self._prepare_data(data, tickers)
        self.n_assets = len(self.tickers)
        self.asset_returns = self._build_asset_returns()
        self.feature_tensor = self._build_feature_tensor()

        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.weights = self._initial_weights()
        self.portfolio_returns = []
        self.turnover = 0.0

    def _prepare_data(self, data, tickers):
        df = data.copy()
        df.columns = [str(column).strip() for column in df.columns]
        required_columns = set(["Date", "Ticker", "Close"] + list(self.feature_columns))
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        df["Date"] = pd.to_datetime(df["Date"])
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        selected_tickers = sorted(df["Ticker"].unique().tolist()) if tickers is None else sorted({str(ticker).upper() for ticker in tickers})
        if not selected_tickers:
            raise ValueError("At least one ticker is required.")

        df = df[df["Ticker"].isin(selected_tickers)].copy()
        if df.empty:
            raise ValueError(f"No rows found for tickers: {selected_tickers}")

        numeric_columns = set(["Close"] + list(self.feature_columns))
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=list(numeric_columns))
        price_table = df.pivot(index="Date", columns="Ticker", values="Close").sort_index()
        price_table = price_table.dropna(axis=0, how="any")
        available_tickers = [ticker for ticker in selected_tickers if ticker in price_table.columns]
        price_table = price_table[available_tickers]
        if len(available_tickers) < 2:
            raise ValueError("PortfolioEnv requires at least two assets with overlapping dates.")
        if len(price_table) < 3:
            raise ValueError("PortfolioEnv requires at least three overlapping dates.")

        common_dates = price_table.index
        df = df[df["Date"].isin(common_dates) & df["Ticker"].isin(available_tickers)].copy()
        return df.sort_values(["Date", "Ticker"]).reset_index(drop=True), available_tickers, common_dates

    def _build_asset_returns(self):
        close = self.data.pivot(index="Date", columns="Ticker", values="Close").sort_index()
        close = close[self.tickers]
        returns = close.pct_change().fillna(0.0)
        return returns.to_numpy(dtype=np.float64)

    def _build_feature_tensor(self):
        frames = []
        for column in self.feature_columns:
            table = self.data.pivot(index="Date", columns="Ticker", values=column).sort_index()
            frames.append(table[self.tickers].to_numpy(dtype=np.float64))
        tensor = np.stack(frames, axis=2)

        means = np.nanmean(tensor, axis=(0, 1), keepdims=True)
        stds = np.nanstd(tensor, axis=(0, 1), keepdims=True)
        stds = np.where(stds == 0, 1.0, stds)
        return np.nan_to_num((tensor - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)

    def _initial_weights(self):
        weights = np.zeros(self.n_assets + 1, dtype=np.float64)
        weights[0] = 1.0
        return weights

    @property
    def action_size(self):
        return self.n_assets + 1

    @property
    def observation_size(self):
        return len(self._get_observation())

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.weights = self._initial_weights()
        self.portfolio_returns = []
        self.turnover = 0.0
        return self._get_observation()

    def step(self, action):
        if self.current_step >= len(self.dates) - 1:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        target_weights = self._normalize_action(action)
        previous_value = self.portfolio_value
        self.turnover = float(np.abs(target_weights - self.weights).sum())
        transaction_cost = previous_value * self.transaction_cost_pct * self.turnover

        next_asset_returns = self.asset_returns[self.current_step + 1]
        portfolio_return_before_cost = float(np.dot(target_weights[1:], next_asset_returns))
        next_value_before_cost = previous_value * (1.0 + portfolio_return_before_cost)
        next_value = max(next_value_before_cost - transaction_cost, 0.0)
        portfolio_return = next_value / previous_value - 1.0

        self.current_step += 1
        self.portfolio_value = next_value
        self.weights = target_weights
        self.portfolio_returns.append(portfolio_return)

        reward = self._calculate_reward(portfolio_return)
        done = self.current_step >= len(self.dates) - 1
        info = {
            "date": self.dates[self.current_step],
            "tickers": self.tickers,
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "transaction_cost": float(transaction_cost),
            "turnover": self.turnover,
            "cash_weight": float(self.weights[0]),
            "asset_weights": self.weights[1:].copy(),
        }
        return self._get_observation(), reward, done, info

    def _normalize_action(self, action):
        weights = np.asarray(action, dtype=np.float64).reshape(-1)
        if len(weights) != self.action_size:
            raise ValueError(f"Action must have length {self.action_size}.")
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total <= 0:
            return self._initial_weights()
        return weights / total

    def _calculate_reward(self, portfolio_return):
        if self.reward_mode == "portfolio_return":
            reward = portfolio_return
        else:
            reward = np.log1p(max(portfolio_return, -0.999999))

        if self.risk_penalty > 0 and len(self.portfolio_returns) >= 2:
            window = self.portfolio_returns[-self.risk_window:]
            reward -= self.risk_penalty * float(np.std(window))
        return float(reward)

    def _get_observation(self):
        market_features = self.feature_tensor[self.current_step].reshape(-1)
        portfolio_features = np.array(
            [
                self.portfolio_value / self.initial_cash,
                self.turnover,
                *self.weights,
            ],
            dtype=np.float64,
        )
        return np.concatenate([market_features, portfolio_features]).astype(np.float32)

    def get_state(self):
        return PortfolioState(
            step=self.current_step,
            date=self.dates[self.current_step],
            portfolio_value=self.portfolio_value,
            cash_weight=float(self.weights[0]),
            asset_weights=self.weights[1:].copy(),
            turnover=self.turnover,
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
    env = PortfolioEnv(data, tickers=["AAPL", "MSFT", "QQQ"])
    observation = env.reset()
    print(f"Tickers: {env.tickers}")
    print(f"Action size: {env.action_size}")
    print(f"Observation size: {len(observation)}")
    action = np.ones(env.action_size)
    _, reward, done, info = env.step(action)
    print(f"Reward: {reward:.6f}, Value: {info['portfolio_value']:.2f}, Done: {done}")

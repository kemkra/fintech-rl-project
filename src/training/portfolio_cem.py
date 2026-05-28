from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.portfolio_env import PortfolioEnv
from src.evaluation.metrics import calculate_performance_metrics


PROCESSED_DATA_FILE = Path("data/processed/stock_features.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("reports/results")
PORTFOLIO_RL_EQUITY_FILE = RESULTS_DIR / "portfolio_rl_equity_curve.csv"
PORTFOLIO_RL_METRICS_FILE = RESULTS_DIR / "portfolio_rl_metrics.csv"
PORTFOLIO_RL_MODEL_FILE = MODELS_DIR / "portfolio_cem_policy.npz"


class LinearSoftmaxPolicy:
    def __init__(self, observation_size, action_size, weights=None, bias=None):
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.weights = np.zeros((self.observation_size, self.action_size), dtype=np.float64) if weights is None else weights
        self.bias = np.zeros(self.action_size, dtype=np.float64) if bias is None else bias

    @property
    def parameter_size(self):
        return self.observation_size * self.action_size + self.action_size

    def set_parameters(self, params):
        params = np.asarray(params, dtype=np.float64)
        expected = self.parameter_size
        if len(params) != expected:
            raise ValueError(f"Expected {expected} parameters, got {len(params)}.")
        split = self.observation_size * self.action_size
        self.weights = params[:split].reshape(self.observation_size, self.action_size)
        self.bias = params[split:]

    def get_parameters(self):
        return np.concatenate([self.weights.reshape(-1), self.bias])

    def act(self, observation):
        logits = np.asarray(observation, dtype=np.float64) @ self.weights + self.bias
        logits = logits - np.max(logits)
        exp_logits = np.exp(np.clip(logits, -30, 30))
        total = exp_logits.sum()
        if total <= 0 or not np.isfinite(total):
            return np.ones(self.action_size) / self.action_size
        return exp_logits / total

    def save(self, path, tickers=None, feature_columns=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            bias=self.bias,
            tickers=np.array(tickers or [], dtype=object),
            feature_columns=np.array(feature_columns or [], dtype=object),
        )


def load_feature_data(file_path=PROCESSED_DATA_FILE):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Load and process market data first.")
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def split_by_date(df, train_ratio=0.7):
    dates = sorted(pd.to_datetime(df["Date"]).unique())
    if len(dates) < 10:
        raise ValueError("At least 10 dates are required for train/evaluation split.")
    split_index = max(3, min(int(len(dates) * float(train_ratio)), len(dates) - 3))
    split_date = dates[split_index]
    train_df = df[df["Date"] < split_date].copy()
    eval_df = df[df["Date"] >= split_date].copy()
    return train_df, eval_df, pd.Timestamp(split_date)


def run_episode(env, policy, capture_curve=False):
    observation = env.reset()
    done = False
    total_reward = 0.0
    records = []
    while not done:
        action = policy.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if capture_curve:
            record = {
                "Date": info["date"],
                "Portfolio_Value": info["portfolio_value"],
                "Portfolio_Return": info["portfolio_return"],
                "Turnover": info["turnover"],
                "Cash_Weight": info["cash_weight"],
            }
            for ticker, weight in zip(info["tickers"], info["asset_weights"]):
                record[f"Weight_{ticker}"] = float(weight)
            records.append(record)
    equity_curve = pd.DataFrame(records)
    return float(total_reward), equity_curve


def train_cem_policy(
    train_data,
    tickers=None,
    initial_cash=100_000.0,
    transaction_cost_pct=0.001,
    generations=8,
    population_size=24,
    elite_fraction=0.25,
    noise_scale=0.2,
    random_seed=42,
):
    rng = np.random.default_rng(random_seed)
    probe_env = PortfolioEnv(
        train_data,
        tickers=tickers,
        initial_cash=initial_cash,
        transaction_cost_pct=transaction_cost_pct,
    )
    policy = LinearSoftmaxPolicy(probe_env.observation_size, probe_env.action_size)
    mean = policy.get_parameters()
    std = np.ones_like(mean) * float(noise_scale)
    elite_count = max(1, int(population_size * elite_fraction))
    history = []
    best_params = mean.copy()
    best_reward = -np.inf

    for generation in range(1, int(generations) + 1):
        candidates = rng.normal(loc=mean, scale=std, size=(int(population_size), len(mean)))
        rewards = []
        for params in candidates:
            candidate_policy = LinearSoftmaxPolicy(probe_env.observation_size, probe_env.action_size)
            candidate_policy.set_parameters(params)
            env = PortfolioEnv(
                train_data,
                tickers=probe_env.tickers,
                initial_cash=initial_cash,
                transaction_cost_pct=transaction_cost_pct,
            )
            reward, _ = run_episode(env, candidate_policy)
            rewards.append(reward)

        rewards = np.asarray(rewards)
        elite_indices = rewards.argsort()[-elite_count:]
        elites = candidates[elite_indices]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 1e-3)
        generation_best_index = int(np.argmax(rewards))
        if rewards[generation_best_index] > best_reward:
            best_reward = float(rewards[generation_best_index])
            best_params = candidates[generation_best_index].copy()

        history.append(
            {
                "Generation": generation,
                "Best_Reward": float(rewards.max()),
                "Mean_Reward": float(rewards.mean()),
                "Elite_Mean_Reward": float(rewards[elite_indices].mean()),
            }
        )

    policy.set_parameters(best_params)
    return policy, pd.DataFrame(history), probe_env.tickers


def evaluate_policy(
    data,
    policy,
    tickers,
    initial_cash=100_000.0,
    transaction_cost_pct=0.001,
):
    env = PortfolioEnv(
        data,
        tickers=tickers,
        initial_cash=initial_cash,
        transaction_cost_pct=transaction_cost_pct,
    )
    _, equity_curve = run_episode(env, policy, capture_curve=True)
    if equity_curve.empty:
        raise ValueError("Evaluation produced an empty equity curve.")
    metrics = calculate_performance_metrics(equity_curve)
    metrics["Strategy"] = "PortfolioCEM"
    metrics["Ticker"] = "Portfolio"
    metrics["Asset_Set"] = ",".join(tickers)
    metrics["comparison_level"] = "portfolio"
    metrics["Tickers"] = ",".join(tickers)
    metrics["Initial_Cash"] = float(initial_cash)
    metrics["Transaction_Cost_Pct"] = float(transaction_cost_pct)
    return equity_curve, metrics


def run_portfolio_cem_training(
    data_file=PROCESSED_DATA_FILE,
    tickers=None,
    initial_cash=100_000.0,
    transaction_cost_pct=0.001,
    generations=8,
    population_size=24,
    elite_fraction=0.25,
    noise_scale=0.2,
    train_ratio=0.7,
    random_seed=42,
    equity_file=PORTFOLIO_RL_EQUITY_FILE,
    metrics_file=PORTFOLIO_RL_METRICS_FILE,
    model_file=PORTFOLIO_RL_MODEL_FILE,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_feature_data(data_file)
    train_df, eval_df, split_date = split_by_date(df, train_ratio=train_ratio)
    policy, training_history, trained_tickers = train_cem_policy(
        train_data=train_df,
        tickers=tickers,
        initial_cash=initial_cash,
        transaction_cost_pct=transaction_cost_pct,
        generations=generations,
        population_size=population_size,
        elite_fraction=elite_fraction,
        noise_scale=noise_scale,
        random_seed=random_seed,
    )
    equity_curve, metrics = evaluate_policy(
        data=eval_df,
        policy=policy,
        tickers=trained_tickers,
        initial_cash=initial_cash,
        transaction_cost_pct=transaction_cost_pct,
    )
    metrics["Train_Split_Date"] = split_date.date().isoformat()
    metrics["Generations"] = int(generations)
    metrics["Population_Size"] = int(population_size)
    metrics["Elite_Fraction"] = float(elite_fraction)
    metrics["Noise_Scale"] = float(noise_scale)
    metrics["Random_Seed"] = int(random_seed)

    equity_file = Path(equity_file)
    metrics_file = Path(metrics_file)
    model_file = Path(model_file)
    equity_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    equity_curve.to_csv(equity_file, index=False)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_file, index=False)
    history_file = metrics_file.with_name("portfolio_rl_training_history.csv")
    training_history.to_csv(history_file, index=False)
    policy.save(model_file, tickers=trained_tickers)

    return {
        "strategy": "PortfolioCEM",
        "tickers": trained_tickers,
        "train_split_date": split_date.date().isoformat(),
        "equity_file": str(equity_file),
        "metrics_file": str(metrics_file),
        "training_history_file": str(history_file),
        "model_file": str(model_file),
        "metrics": metrics,
    }


if __name__ == "__main__":
    result = run_portfolio_cem_training()
    print(result)

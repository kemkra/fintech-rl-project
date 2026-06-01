import json
from io import StringIO
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from src.evaluation.risk_analysis import calculate_risk_analysis
from src.storage import runtime_store
from src.tools import market_tools
from src.llm import assistant


def synthetic_feature_data():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    rows = []
    for ticker, start_price, drift in [
        ("AAPL", 100.0, 0.0010),
        ("MSFT", 80.0, 0.0008),
        ("SPY", 400.0, 0.0005),
    ]:
        price = start_price
        for index, date in enumerate(dates):
            daily_return = drift + ((index % 7) - 3) * 0.0008
            price *= 1 + daily_return
            rows.append(
                {
                    "Date": date.date().isoformat(),
                    "Ticker": ticker,
                    "Open": price,
                    "High": price * 1.01,
                    "Low": price * 0.99,
                    "Close": price,
                    "Volume": 1_000_000 + index,
                    "Daily_Return": daily_return,
                    "MA5": price,
                    "MA20": price,
                    "RSI": 50.0,
                    "MACD": 0.0,
                    "Volatility": 0.01,
                }
            )
    return pd.DataFrame(rows)


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


class CoreToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        market_tools.configure_runtime_storage(session_id="test_session", root=self.root)
        self.feature_df = synthetic_feature_data()
        market_tools.PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.feature_df.to_csv(market_tools.PROCESSED_DATA_FILE, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_runtime_store_saves_dataset_registry(self):
        raw_df = self.feature_df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
        runtime_store.save_runtime_dataset(
            db_file=self.root / "runtime.db",
            session_id="test_session",
            dataset_id="synthetic",
            raw_df=raw_df,
            feature_df=self.feature_df,
            scope="project",
        )

        status = runtime_store.get_runtime_status(self.root / "runtime.db", session_id="test_session")

        self.assertTrue(status["available"])
        self.assertEqual(status["datasets"][0]["dataset_id"], "synthetic")
        self.assertEqual(status["datasets"][0]["row_count"], len(raw_df))
        self.assertEqual(status["datasets"][0]["feature_row_count"], len(self.feature_df))

    def test_runtime_diagnostics_and_cleanup_preserve_current_session(self):
        old_paths = runtime_store.ensure_runtime(session_id="old_session", root=self.root)
        current_paths = runtime_store.ensure_runtime(session_id="test_session", root=self.root)
        old_file = old_paths["session_dir"] / "old.txt"
        current_file = current_paths["session_dir"] / "current.txt"
        old_file.write_text("old", encoding="utf-8")
        current_file.write_text("current", encoding="utf-8")
        old_timestamp = pd.Timestamp.now().timestamp() - 48 * 3600
        os.utime(old_paths["session_dir"], (old_timestamp, old_timestamp))

        diagnostics = runtime_store.get_runtime_diagnostics(root=self.root, current_session_id="test_session")
        cleanup = runtime_store.cleanup_old_sessions(root=self.root, current_session_id="test_session", max_age_hours=24)

        self.assertGreaterEqual(diagnostics["session_count"], 2)
        self.assertEqual(cleanup["removed_count"], 1)
        self.assertFalse(old_paths["session_dir"].exists())
        self.assertTrue(current_paths["session_dir"].exists())

    def test_risk_analysis_outputs_core_metrics(self):
        summary, rolling = calculate_risk_analysis(self.feature_df, benchmark_ticker="SPY", rolling_window=20)

        self.assertEqual(set(summary["Ticker"]), {"AAPL", "MSFT", "SPY"})
        self.assertIn("VaR_95", summary.columns)
        self.assertIn("CVaR_95", summary.columns)
        self.assertIn("Max_Drawdown_Duration_Days", summary.columns)
        self.assertIn("Beta_vs_Benchmark", summary.columns)
        self.assertGreater(len(rolling), 0)
        self.assertIn("Rolling_Volatility", rolling.columns)
        self.assertIn("Rolling_Sharpe", rolling.columns)

    def test_market_tool_risk_analysis_and_report(self):
        risk = market_tools.run_risk_analysis(data_scope="project", benchmark_ticker="SPY", rolling_window=20)
        report = market_tools.build_unified_report(
            title="Synthetic Report",
            data_scope="project",
            include_research=False,
            include_strategy=False,
            include_portfolio=False,
        )

        self.assertTrue(risk["available"])
        self.assertTrue(Path(risk["summary_file"]).exists())
        self.assertTrue(Path(risk["rolling_file"]).exists())
        self.assertTrue(report["risk_summary"])
        self.assertTrue(Path(report["markdown_file"]).exists())
        self.assertIn("## Risk Analysis", Path(report["markdown_file"]).read_text(encoding="utf-8"))

    def test_strategy_comparison_and_export_zip(self):
        write_csv(
            market_tools.BUY_HOLD_METRICS_FILE,
            [
                {
                    "Ticker": "AAPL",
                    "Strategy": "BuyHold",
                    "start_date": "2024-01-01",
                    "end_date": "2024-04-19",
                    "start_value": 100000,
                    "end_value": 110000,
                    "total_return": 0.10,
                    "annualized_return": 0.30,
                    "annualized_volatility": 0.20,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": -0.05,
                    "win_rate": 0.55,
                }
            ],
        )
        write_csv(
            market_tools.MA_METRICS_FILE,
            [
                {
                    "Ticker": "AAPL",
                    "Strategy": "MovingAverage",
                    "start_date": "2024-01-01",
                    "end_date": "2024-04-19",
                    "start_value": 100000,
                    "end_value": 105000,
                    "total_return": 0.05,
                    "annualized_return": 0.15,
                    "annualized_volatility": 0.18,
                    "sharpe_ratio": 0.8,
                    "max_drawdown": -0.04,
                    "win_rate": 0.50,
                }
            ],
        )

        comparison = market_tools.run_strategy_comparison(data_scope="project")
        inventory = market_tools.list_exportable_artifacts(
            data_scope="project",
            include_data=True,
            include_figures=False,
            include_strategy_results=True,
            include_models=False,
            include_research=False,
            include_reports=False,
        )
        selected_codes = [record["artifact_code"] for record in inventory["records"][:2]]
        export = market_tools.export_selected_artifacts(
            artifact_codes=selected_codes,
            data_scope="project",
            export_name="synthetic_export",
        )

        self.assertTrue(comparison["available"])
        self.assertGreaterEqual(comparison["rows"], 2)
        self.assertTrue(export["exported"])
        with zipfile.ZipFile(export["export_file"]) as archive:
            names = set(archive.namelist())
            metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
        self.assertIn("metadata.json", names)
        self.assertEqual(set(metadata["artifact_codes"]), set(selected_codes))
        self.assertGreaterEqual(len(names), 2)

    def test_llm_registry_exposes_risk_tools(self):
        tool_names = {schema["function"]["name"] for schema in assistant.build_tool_schemas(allow_write_tools=True)}

        self.assertIn("run_risk_analysis", tool_names)
        self.assertIn("get_risk_summary", tool_names)
        self.assertIn("get_risk_rolling_metrics", tool_names)
        self.assertIs(assistant.TOOL_FUNCTIONS["run_risk_analysis"], market_tools.run_risk_analysis)

    def test_research_citations_are_structured(self):
        fundamentals = {
            "source": "yfinance.get_info",
            "retrieved_at": "2026-01-01T00:00:00",
            "records": [
                {
                    "Ticker": "AAPL",
                    "available": True,
                    "longName": "Apple Inc.",
                    "source_url": "https://finance.yahoo.com/quote/AAPL",
                }
            ],
        }
        macro = {
            "source": "yfinance.history",
            "period": "6mo",
            "retrieved_at": "2026-01-01T00:00:00",
            "records": [
                {
                    "Ticker": "SPY",
                    "Name": "S&P 500 ETF",
                    "available": True,
                    "source_url": "https://finance.yahoo.com/quote/SPY",
                }
            ],
        }
        news = {
            "source": "Yahoo Finance RSS",
            "retrieved_at": "2026-01-01T00:00:00",
            "records": [
                {
                    "ticker": "AAPL",
                    "title": "Apple test headline",
                    "publisher": "Yahoo Finance",
                    "url": "https://finance.yahoo.com/news/test",
                    "published": "2026-01-01T00:00:00",
                }
            ],
        }

        citations = market_tools.build_research_citations(
            fundamentals=fundamentals,
            macro=macro,
            news=news,
            retrieved_at="2026-01-01T00:00:00",
        )

        self.assertEqual(len(citations), 3)
        self.assertEqual({citation["category"] for citation in citations}, {"fundamentals", "macro_market", "news_headline"})
        self.assertTrue(all(citation["citation_id"] for citation in citations))
        self.assertTrue(all("source_quality_score" in citation for citation in citations))

    def test_uploaded_csv_import_pipeline(self):
        upload_df = self.feature_df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]].copy()
        buffer = StringIO()
        upload_df.to_csv(buffer, index=False)
        buffer.seek(0)

        result = market_tools.import_uploaded_price_data(
            uploaded_file=buffer,
            run_eda_after=True,
            run_baseline_after=True,
        )

        self.assertEqual(result["data_source_mode"], "uploaded_csv")
        self.assertEqual(result["processed_ticker_count"], 3)
        self.assertTrue(Path(result["processed_file"]).exists())
        self.assertTrue(Path(result["risk_summary_file"]).exists())
        self.assertTrue(Path(result["strategy_comparison_file"]).exists())

    def test_uploaded_csv_single_ticker_aliases(self):
        upload_df = self.feature_df[self.feature_df["Ticker"] == "AAPL"][["Date", "Close"]].rename(
            columns={"Date": "timestamp", "Close": "adj close"}
        )

        normalized = market_tools.normalize_uploaded_price_data(upload_df, default_ticker="aapl")

        self.assertEqual(normalized["Ticker"].unique().tolist(), ["AAPL"])
        self.assertTrue({"Date", "Ticker", "Open", "High", "Low", "Close", "Volume"}.issubset(normalized.columns))
        self.assertEqual(len(normalized), len(upload_df))


if __name__ == "__main__":
    unittest.main()

"""
CIFT Markets - Command Line Interface

Typer-based CLI for development and operations tasks.
"""

import typer
from rich.console import Console
from rich.table import Table

from cift.core.config import settings
from cift.core.logging import logger

app = typer.Typer(
    name="cift",
    help="CIFT Markets - Computational Intelligence for Financial Trading CLI",
    add_completion=False,
)
console = Console()


@app.command()
def info():
    """Display system information and configuration."""
    table = Table(title="CIFT Markets Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Environment", settings.app_env)
    table.add_row("Debug Mode", str(settings.app_debug))
    table.add_row("API URL", settings.app_url)
    table.add_row("PostgreSQL", settings.postgres_url)
    table.add_row("QuestDB HTTP", settings.questdb_http_url)
    table.add_row("Redis", settings.redis_url)
    table.add_row("Kafka", settings.kafka_bootstrap_servers)
    table.add_row("MLflow", settings.mlflow_tracking_uri)
    
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(True, help="Enable auto-reload"),
):
    """Start the FastAPI server."""
    import uvicorn
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "cift.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@app.command()
def db_init():
    """Initialize the database schema."""
    logger.info("Initializing database...")
    # TODO: Implement database initialization
    console.print("[green]✓[/green] Database initialized successfully")


@app.command()
def create_user(
    email: str = typer.Option(..., prompt=True),
    username: str = typer.Option(..., prompt=True),
    password: str = typer.Option(..., prompt=True, hide_input=True),
    is_superuser: bool = typer.Option(False, "--admin", help="Create as admin"),
):
    """Create a new user."""
    logger.info(f"Creating user: {username}")
    # TODO: Implement user creation
    console.print(f"[green]✓[/green] User '{username}' created successfully")


@app.command()
def download_data(
    symbols: str = typer.Option(..., help="Comma-separated list of symbols"),
    start_date: str = typer.Option(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., help="End date (YYYY-MM-DD)"),
    provider: str = typer.Option("polygon", help="Data provider (polygon/alpaca)"),
):
    """Download historical market data."""
    logger.info(f"Downloading data for {symbols} from {start_date} to {end_date}")
    # TODO: Implement data download
    console.print("[green]✓[/green] Data downloaded successfully")


@app.command()
def train_model(
    model_name: str = typer.Argument(..., help="Model name (hawkes/transformer/hmm/gnn/xgboost)"),
    symbols: str = typer.Option("AAPL,MSFT,GOOGL", help="Training symbols"),
    start_date: str = typer.Option(..., help="Training start date"),
    end_date: str = typer.Option(..., help="Training end date"),
):
    """Train a machine learning model."""
    logger.info(f"Training {model_name} model...")
    # TODO: Implement model training
    console.print(f"[green]✓[/green] Model '{model_name}' trained successfully")


@app.command()
def backtest(
    strategy: str = typer.Argument(..., help="Strategy name"),
    symbols: str = typer.Option("AAPL", help="Symbols to backtest"),
    start_date: str = typer.Option(..., help="Backtest start date"),
    end_date: str = typer.Option(..., help="Backtest end date"),
    initial_capital: float = typer.Option(100000.0, help="Initial capital"),
    data_path: str = typer.Option(
        None,
        help="Optional CSV/Parquet containing at least: returns, signal (and optionally timestamp).",
    ),
    commission_bps: float = typer.Option(1.0, help="Commission in bps per position change"),
    slippage_bps: float = typer.Option(1.0, help="Slippage in bps per position change"),
    periods_per_year: int = typer.Option(252, help="Annualization periods (252 daily, 52 weekly, etc.)"),
):
    """Run a backtest."""
    logger.info(f"Running backtest for {strategy}...")
    if not data_path:
        raise typer.BadParameter(
            "This CLI backtest currently requires --data-path (CSV/Parquet with returns+signal)."
        )

    import polars as pl

    df = pl.read_parquet(data_path) if data_path.lower().endswith(".parquet") else pl.read_csv(data_path)
    required = {"returns", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(f"Missing required columns in {data_path}: {sorted(missing)}")

    from cift.backtest import backtest_positions

    result = backtest_positions(
        returns=df["returns"].to_numpy(),
        positions=df["signal"].to_numpy(),
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        periods_per_year=periods_per_year,
    )

    table = Table(title=f"Backtest Results: {strategy}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k in [
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "turnover",
        "final_portfolio_value",
    ]:
        table.add_row(k, str(round(float(result.metrics.get(k, 0.0)), 6)))
    console.print(table)
    console.print("[green]✓[/green] Backtest completed")


@app.command()
def walkforward(
    data_path: str = typer.Option(..., help="CSV/Parquet with at least timestamp, close"),
    timestamp_col: str = typer.Option("timestamp", help="Timestamp column name"),
    close_col: str = typer.Option("close", help="Close/price column name"),
    horizon_bars: int = typer.Option(1, help="Forward-return horizon in bars"),
    n_splits: int = typer.Option(5, help="Number of CV folds"),
    embargo_bars: int = typer.Option(0, help="Embargo in bars"),
    commission_bps: float = typer.Option(1.0, help="Commission in bps per position change"),
    slippage_bps: float = typer.Option(1.0, help="Slippage in bps per position change"),
    periods_per_year: int = typer.Option(252, help="Annualization periods"),
    threshold: float = typer.Option(0.0, help="Signal threshold"),
    model: str = typer.Option("baseline", help="Model: baseline|logreg|xgboost"),
    n_lags: int = typer.Option(10, help="Number of lag features for logreg/xgboost"),
    n_trials: int = typer.Option(1, help="Selection trials for DSR (if you searched many variants)"),
    holdout_bars: int = typer.Option(0, help="Reserve final N bars as strict time holdout"),
    tune: bool = typer.Option(False, help="Tune logreg (C, threshold) via inner purged CV"),
    tune_splits: int = typer.Option(3, help="Inner CV splits for tuning"),
    c_grid: str = typer.Option("0.1,1.0,10.0", help="Comma-separated C values to try"),
    threshold_grid: str = typer.Option("0.0,0.02,0.05", help="Comma-separated threshold values to try"),
    use_fracdiff: bool = typer.Option(False, help="Apply fractional differentiation (De Prado) to close"),
    fracdiff_d: float = typer.Option(0.4, help="FracDiff order (0 < d < 1)"),
    use_vol_features: bool = typer.Option(False, help="Add volatility + vol momentum features"),
    vol_window: int = typer.Option(20, help="Window for volatility features"),
    use_triple_barrier: bool = typer.Option(False, help="Use Triple Barrier Method (PT/SL/Time) labels"),
    tb_pt: float = typer.Option(2.0, help="Triple Barrier profit-take multiplier (barrier = pt * vol)"),
    tb_sl: float = typer.Option(2.0, help="Triple Barrier stop-loss multiplier (barrier = sl * vol)"),
    tb_min_ret: float = typer.Option(0.0, help="Triple Barrier minimum return for non-zero label"),
    use_meta_labeling: bool = typer.Option(False, help="Use meta-labeling (De Prado) for bet sizing/filtering"),
    meta_model: str = typer.Option("xgboost", help="Meta-model type: logreg|xgboost"),
    meta_use_sizing: bool = typer.Option(True, help="Use continuous bet sizing (vs binary filter)"),
    meta_threshold: float = typer.Option(0.5, help="Meta-model threshold for binary filter mode"),
    use_sample_weights: bool = typer.Option(False, help="Use sample weights based on avg uniqueness (De Prado)"),
    use_ta_features: bool = typer.Option(False, help="Add standard TA features (RSI, MACD, BB, ATR, MFI)"),
    use_micro_features: bool = typer.Option(False, help="Add microstructure features (Spread, Efficiency, BP Vol)"),
    vol_target: float = typer.Option(0.0, help="Annualized volatility target (e.g. 0.15). 0.0 to disable."),
    tune_model: bool = typer.Option(False, help="Enable hyperparameter tuning (Random Search) for XGBoost"),
):
    """Leakage-safe walk-forward evaluation.

    This command exists to make Sharpe claims measurable and reproducible.
    It uses purged+embargo splits and outputs an out-of-sample metric stream.
    """
    from cift.ml.evaluation.walkforward import run_walkforward

    report = run_walkforward(
        data_path=data_path,
        timestamp_col=timestamp_col,
        close_col=close_col,
        horizon_bars=horizon_bars,
        n_splits=n_splits,
        embargo_bars=embargo_bars,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        periods_per_year=periods_per_year,
        threshold=threshold,
        model=model,
        n_lags=n_lags,
        n_trials=n_trials,
        holdout_bars=holdout_bars,
        tune=tune,
        tune_splits=tune_splits,
        c_grid=c_grid,
        threshold_grid=threshold_grid,
        use_fracdiff=use_fracdiff,
        fracdiff_d=fracdiff_d,
        use_vol_features=use_vol_features,
        vol_window=vol_window,
        use_triple_barrier=use_triple_barrier,
        tb_pt=tb_pt,
        tb_sl=tb_sl,
        tb_min_ret=tb_min_ret,
        use_meta_labeling=use_meta_labeling,
        meta_model=meta_model,
        meta_use_sizing=meta_use_sizing,
        meta_threshold=meta_threshold,
        use_sample_weights=use_sample_weights,
        use_ta_features=use_ta_features,
        use_micro_features=use_micro_features,
        vol_target=vol_target,
        tune_model=tune_model,
    )

    table = Table(title="Walk-Forward (Purged+Embargo) — Out-of-Sample Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in report.metrics.items():
        table.add_row(str(k), str(v))
    console.print(table)


if __name__ == "__main__":
    app()

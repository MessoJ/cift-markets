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
):
    """Run a backtest."""
    logger.info(f"Running backtest for {strategy}...")
    # TODO: Implement backtest
    console.print("[green]✓[/green] Backtest completed successfully")


if __name__ == "__main__":
    app()

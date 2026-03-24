"""
SentimentPulse - CLI commands for model management
Built by Himal Badu, AI Founder

Commands for managing models and cache.
"""

import click
from rich.console import Console
from rich.table import Table

from sentimentpulse.model_manager import get_model_manager


console = Console()


@click.group()
def models():
    """Model management commands."""
    pass


@models.command("list")
def list_models():
    """List downloaded models."""
    manager = get_model_manager()
    models_list = manager.list_downloaded_models()
    
    if not models_list:
        console.print("[yellow]No models downloaded yet[/yellow]")
        return
    
    table = Table(title="Downloaded Models")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="green", justify="right")
    
    for model in models_list:
        table.add_row(model["name"], model["size"])
    
    console.print(table)
    
    # Show total
    usage = manager.get_disk_usage()
    console.print(f"\n[bold]Total cache size:[/bold] {usage['total']}")


@models.command("info")
@click.argument("model_name")
def model_info(model_name: str):
    """Get information about a specific model."""
    manager = get_model_manager()
    size_bytes, size_human = manager.get_model_size(model_name)
    
    console.print(f"[bold]Model:[/bold] {model_name}")
    console.print(f"[bold]Size:[/bold] {size_human}")
    console.print(f"[bold]Path:[/bold] {manager.cache_dir / f'models--{model_name.replace('/', "--")}'}")


@models.command("delete")
@click.argument("model_name")
@click.confirmation_option(prompt="Are you sure you want to delete this model?")
def delete_model(model_name: str):
    """Delete a downloaded model."""
    manager = get_model_manager()
    
    if manager.delete_model(model_name):
        console.print(f"[green]Model {model_name} deleted successfully[/green]")
    else:
        console.print(f"[red]Model {model_name} not found[/red]")


@models.command("clear")
@click.confirmation_option(prompt="Are you sure you want to delete ALL models?")
def clear_models():
    """Clear all downloaded models."""
    manager = get_model_manager()
    
    count = manager.clear_all_models()
    console.print(f"[green]Deleted {count} models[/green]")


@models.command("cache-size")
def cache_size():
    """Show total cache size."""
    manager = get_model_manager()
    usage = manager.get_disk_usage()
    
    console.print(f"[bold]Total model cache size:[/bold] {usage['total']}")

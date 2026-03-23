"""
SentimentPulse CLI - Production-Grade Command Line Interface
Built by Himal Badu, AI Founder

A beautiful, feature-rich CLI for sentiment analysis with:
- Interactive mode
- Batch processing
- Rich output formatting
- Progress indicators
"""

import os
import sys
import json
import logging
from typing import List, Optional
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich import box

from loguru import logger
from sentimentpulse import get_engine, analyze_sentiment, analyze_batch


# ============================================================================
# Console Setup
# ============================================================================

console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging for CLI."""
    logger.remove()
    level = "DEBUG" if verbose else "WARNING"
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> - <level>{message}</level>",
        level=level
    )


# ============================================================================
# CLI Commands
# ============================================================================

@click.group()
@click.version_option(version="2.0.0", prog_name="SentimentPulse")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """
    SentimentPulse - Production-grade sentiment analysis CLI.
    
    Built by Himal Badu, AI Founder
    """
    setup_logging(verbose)


@cli.command()
@click.argument("text", required=False)
@click.option("--verbose", "-vv", is_flag=True, help="Show detailed scores")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def analyze(text: str, verbose: bool, json_output: bool):
    """Analyze sentiment of text."""
    # Input handling
    if not text:
        text = click.prompt(
            "[bold cyan]Enter text to analyze[/bold cyan]",
            type=str
        )
    
    if not text.strip():
        console.print("[red]Error:[/red] Text cannot be empty")
        sys.exit(1)
    
    # Show loading state
    with console.status("[bold green]Loading model & analyzing...") as status:
        try:
            engine = get_engine()
            engine.load_model()
            result = engine.analyze(text, verbose=verbose)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
    
    # Display result
    if json_output:
        console.print_json(json.dumps(result))
    else:
        _display_result(result)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output JSON file")
@click.option("--verbose", "-vv", is_flag=True, help="Show detailed scores")
@click.option("--limit", "-l", type=int, help="Limit number of texts to process")
def batch(input_file: Path, output: Optional[Path], verbose: bool, limit: Optional[int]):
    """Analyze batch of texts from file."""
    console.print(f"[bold]Loading texts from:[/bold] {input_file}")
    
    # Read input
    try:
        content = input_file.read_text(encoding="utf-8").strip()
        
        # Try JSON first
        try:
            texts = json.loads(content)
            if isinstance(texts, str):
                texts = [texts]
            elif isinstance(texts, dict):
                texts = list(texts.values())
        except json.JSONDecodeError:
            # Fallback to line-by-line
            texts = [line.strip() for line in content.split('\n') if line.strip()]
        
        if limit:
            texts = texts[:limit]
            
    except Exception as e:
        console.print(f"[red]Error reading file:[/red] {e}")
        sys.exit(1)
    
    if not texts:
        console.print("[red]Error:[/red] No texts found in input file")
        sys.exit(1)
    
    total = len(texts)
    console.print(f"[bold green]Analyzing {total} texts...[/bold green]\n")
    
    # Process with progress
    try:
        engine = get_engine()
        engine.load_model()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing...", total=100)
            
            results = engine.analyze_batch(texts, verbose=verbose, show_progress=False)
            
            progress.update(task, completed=100)
        
    except Exception as e:
        console.print(f"[red]Error during analysis:[/red] {e}")
        sys.exit(1)
    
    # Display results table
    _display_batch_results(results, texts, verbose)
    
    # Save to file
    if output:
        output_data = {
            "results": results,
            "total": len(results),
            "texts": texts if limit and limit <= 10 else "truncated"
        }
        output.write_text(json.dumps(output_data, indent=2))
        console.print(f"\n[bold green]Results saved to:[/bold] {output}")
    
    # Summary
    _display_summary(results)


@cli.command()
@click.argument("text", required=False)
def shell(text: str):
    """Interactive sentiment analysis shell."""
    console.print(Panel(
        "[bold cyan]SentimentPulse Interactive Shell[/bold cyan]\n"
        "Type [bold]exit[/bold] or [bold]quit[/bold] to stop\n"
        "Type [bold]clear[/bold] to clear screen",
        box=box.DOUBLE
    ))
    
    while True:
        try:
            prompt_text = text if text else None
            text = click.prompt(
                "\n[bold green]›[/bold green] ",
                default=prompt_text,
                show_default=False
            )
            
            if text.lower() in ('exit', 'quit', 'q'):
                console.print("[bold cyan]Goodbye! 👋[/bold cyan]")
                break
            
            if text.lower() == 'clear':
                console.clear()
                continue
            
            if not text.strip():
                continue
            
            # Analyze
            with console.status("[bold]Analyzing..."):
                result = analyze_sentiment(text)
            
            # Display
            _display_result(result)
            text = None  # Reset for next iteration
            
        except KeyboardInterrupt:
            console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


@cli.command()
def stats():
    """Show cache statistics."""
    try:
        engine = get_engine()
        cache_stats = engine.get_cache_stats()
        
        table = Table(title="Cache Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Cache Hits", str(cache_stats.get("hits", 0)))
        table.add_row("Cache Misses", str(cache_stats.get("misses", 0)))
        table.add_row("Hit Rate", f"{cache_stats.get('hit_rate_percent', 0):.1f}%")
        table.add_row("Cache Size", str(cache_stats.get("size", 0)))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@cli.command()
def info():
    """Show system and model information."""
    try:
        engine = get_engine()
        health = engine.health_check()
        
        table = Table(title="SentimentPulse System Info", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", health["status"])
        table.add_row("Model Loaded", "✓" if health["model_loaded"] else "✗")
        table.add_row("Model Name", health["model_name"])
        table.add_row("Device", health["device"])
        table.add_row("API Version", "2.0.0")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


# ============================================================================
# Helper Functions
# ============================================================================

def _display_result(result: dict):
    """Display a single sentiment result beautifully."""
    sentiment = result.get("sentiment", "unknown")
    score = result.get("score", 0)
    confidence = result.get("confidence", 0)
    
    # Color based on sentiment
    color = {
        "positive": "green",
        "negative": "red",
        "neutral": "yellow"
    }.get(sentiment, "white")
    
    emoji = {
        "positive": "😊",
        "negative": "😔",
        "neutral": "😐"
    }.get(sentiment, "❓")
    
    confidence_pct = confidence * 100
    
    # Create result panel
    content = f"""
[bold]Sentiment:[/bold] [{color}]{sentiment.capitalize()} {emoji}[/{color}]
[bold]Score:[/bold] {score:+.4f}
[bold]Confidence:[/bold] {confidence_pct:.1f}%
[bold]Model:[/bold] {result.get('model', 'N/A')}
"""
    
    if result.get("raw_scores"):
        raw = result["raw_scores"]
        content += f"\n[dim]Raw Score:[/dim] {raw.get('raw_score', 'N/A')} ({raw.get('label', 'N/A')})"
    
    console.print(Panel(
        content.strip(),
        title="[bold cyan]Analysis Result[/bold cyan]",
        box=box.ROUNDED,
        expand=False
    ))


def _display_batch_results(results: List[dict], texts: List[str], verbose: bool):
    """Display batch results in a table."""
    if len(results) > 50:
        console.print(f"[dim]Showing first 50 of {len(results)} results[/dim]")
        results = results[:50]
        texts = texts[:50]
    
    table = Table(title="Batch Results", box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Text Preview", style="cyan", max_width=40)
    table.add_column("Sentiment", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")
    
    for i, (text, result) in enumerate(zip(texts, results), 1):
        sentiment = result.get("sentiment", "unknown")
        color = {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow"
        }.get(sentiment, "white")
        
        text_preview = text[:37] + "..." if len(text) > 40 else text
        
        table.add_row(
            str(i),
            text_preview,
            f"[{color}]{sentiment}[/{color}]",
            f"{result.get('score', 0):.3f}",
            f"{result.get('confidence', 0)*100:.1f}%"
        )
    
    console.print(table)


def _display_summary(results: List[dict]):
    """Display summary statistics."""
    sentiments = [r.get("sentiment") for r in results]
    
    positive = sentiments.count("positive")
    negative = sentiments.count("negative")
    neutral = sentiments.count("neutral")
    total = len(results)
    
    table = Table(title="Summary", box=box.ROUNDED)
    table.add_column("Sentiment", style="bold")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="cyan")
    
    table.add_row("😊 Positive", str(positive), f"{positive/total*100:.1f}%")
    table.add_row("😔 Negative", str(negative), f"{negative/total*100:.1f}%")
    table.add_row("😐 Neutral", str(neutral), f"{neutral/total*100:.1f}%")
    
    console.print(table)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    cli()

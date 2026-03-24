"""
SentimentPulse - CLI formatters for output
Built by Himal Badu, AI Founder

Output formatters for CLI display.
"""

from typing import Dict, Any, List
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.syntax import Syntax
from rich.json import JSON
from rich import box


console = Console()


class OutputFormatter:
    """Base class for output formatters."""
    
    @staticmethod
    def format_text(text: str, max_length: int = 50) -> str:
        """Format text for display."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."


class TableFormatter(OutputFormatter):
    """Format output as tables."""
    
    @staticmethod
    def format_results_table(results: List[Dict[str, Any]], limit: int = 50):
        """Format results as a table."""
        if not results:
            console.print("[yellow]No results to display[/yellow]")
            return
        
        # Limit results
        display_results = results[:limit]
        
        table = Table(
            title=f"Analysis Results ({len(results)} total)",
            box=box.ROUNDED
        )
        
        table.add_column("#", style="dim", width=4)
        table.add_column("Text", style="cyan", max_width=40)
        table.add_column("Sentiment", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Confidence", justify="right")
        
        for i, result in enumerate(display_results, 1):
            sentiment = result.get("sentiment", "unknown")
            color = {
                "positive": "green",
                "negative": "red",
                "neutral": "yellow"
            }.get(sentiment, "white")
            
            table.add_row(
                str(i),
                TableFormatter.format_text(result.get("text", "")),
                f"[{color}]{sentiment}[/{color}]",
                f"{result.get('score', 0):.3f}",
                f"{result.get('confidence', 0)*100:.1f}%"
            )
        
        console.print(table)
        
        if len(results) > limit:
            console.print(f"[dim]Showing {limit} of {len(results)} results[/dim]")
    
    @staticmethod
    def format_cache_table(stats: Dict[str, Any]):
        """Format cache statistics as a table."""
        table = Table(title="Cache Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Hits", str(stats.get("hits", 0)))
        table.add_row("Misses", str(stats.get("misses", 0)))
        table.add_row("Size", str(stats.get("size", 0)))
        table.add_row("Hit Rate", f"{stats.get('hit_rate_percent', 0):.1f}%")
        
        console.print(table)
    
    @staticmethod
    def format_model_info(info: Dict[str, Any]):
        """Format model information as a table."""
        table = Table(title="Model Information", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in info.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)


class PanelFormatter(OutputFormatter):
    """Format output as panels."""
    
    @staticmethod
    def format_single_result(result: Dict[str, Any], verbose: bool = False):
        """Format a single analysis result as a panel."""
        sentiment = result.get("sentiment", "unknown")
        score = result.get("score", 0)
        confidence = result.get("confidence", 0)
        
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
        
        content = f"""
[bold]Sentiment:[/bold] [{color}]{sentiment.capitalize()} {emoji}[/{color}]
[bold]Score:[/bold] {score:+.4f}
[bold]Confidence:[/bold] {confidence*100:.1f}%
[bold]Model:[/bold] {result.get('model', 'N/A')}
"""
        
        if verbose and result.get("raw_scores"):
            content += f"\n[dim]Raw:[/dim] {result['raw_scores']}"
        
        console.print(Panel(
            content.strip(),
            title="[bold cyan]Analysis Result[/bold cyan]",
            box=box.ROUNDED,
            expand=False
        ))
    
    @staticmethod
    def format_error(error: str, title: str = "Error"):
        """Format an error message as a panel."""
        console.print(Panel(
            f"[red]{error}[/red]",
            title=f"[bold red]{title}[/bold red]",
            box=box.ROUNDED,
            border_style="red"
        ))
    
    @staticmethod
    def format_success(message: str, title: str = "Success"):
        """Format a success message as a panel."""
        console.print(Panel(
            f"[green]{message}[/green]",
            title=f"[bold green]{title}[/bold green]",
            box=box.ROUNDED,
            border_style="green"
        ))


class JSONFormatter:
    """Format output as JSON."""
    
    @staticmethod
    def print_json(data: Any, indent: int = 2):
        """Print data as JSON."""
        console.print(JSON.from_data(data, indent=indent))
    
    @staticmethod
    def format_results_json(results: List[Dict[str, Any]]) -> str:
        """Format results as JSON string."""
        import json
        return json.dumps(results, indent=2)


class ProgressFormatter:
    """Format progress indicators."""
    
    @staticmethod
    def create_progress() -> Progress:
        """Create a progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        )
    
    @staticmethod
    def with_progress(coro, description: str = "Processing..."):
        """Execute a coroutine with progress bar."""
        with ProgressFormatter.create_progress() as progress:
            task = progress.add_task(description, total=None)
            # Note: This is simplified - actual implementation would handle async
            return coro


class MarkdownFormatter:
    """Format output as markdown."""
    
    @staticmethod
    def print_markdown(markdown_text: str):
        """Print markdown-formatted text."""
        console.print(Markdown(markdown_text))
    
    @staticmethod
    def format_results_markdown(results: List[Dict[str, Any]]) -> str:
        """Format results as markdown table."""
        lines = [
            "# Analysis Results",
            "",
            "| # | Text | Sentiment | Score | Confidence |",
            "|---|------|-----------|-------|------------|"
        ]
        
        for i, result in enumerate(results, 1):
            lines.append(
                f"| {i} | "
                f"{TableFormatter.format_text(result.get('text', ''), 20)} | "
                f"{result.get('sentiment', 'N/A')} | "
                f"{result.get('score', 0):.3f} | "
                f"{result.get('confidence', 0)*100:.1f}% |"
            )
        
        return "\n".join(lines)

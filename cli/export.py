"""
SentimentPulse - CLI export commands
Built by Himal Badu, AI Founder

Commands for exporting analysis results.
"""

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from sentimentpulse import analyze_batch
from sentimentpulse.export import ExportManager


console = Console()


@click.group()
def export():
    """Export analysis results."""
    pass


@export.command("json")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True)
def export_json(input_file: Path, output: Path):
    """Export analysis results as JSON."""
    # Read texts
    texts = _read_texts(input_file)
    
    # Analyze
    console.print(f"[bold]Analyzing {len(texts)} texts...[/bold]")
    results = analyze_batch(texts)
    
    # Add texts to results
    for text, result in zip(texts, results):
        result["text"] = text
    
    # Export
    ExportManager.export(results, str(output), "json")
    console.print(f"[green]Exported to {output}[/green]")


@export.command("csv")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True)
def export_csv(input_file: Path, output: Path):
    """Export analysis results as CSV."""
    texts = _read_texts(input_file)
    
    console.print(f"[bold]Analyzing {len(texts)} texts...[/bold]")
    results = analyze_batch(texts)
    
    for text, result in zip(texts, results):
        result["text"] = text
    
    ExportManager.export(results, str(output), "csv")
    console.print(f"[green]Exported to {output}[/green]")


@export.command("markdown")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True)
def export_markdown(input_file: Path, output: Path):
    """Export analysis results as Markdown."""
    texts = _read_texts(input_file)
    
    console.print(f"[bold]Analyzing {len(texts)} texts...[/bold]")
    results = analyze_batch(texts)
    
    for text, result in zip(texts, results):
        result["text"] = text
    
    ExportManager.export(results, str(output), "md")
    console.print(f"[green]Exported to {output}[/green]")


@export.command("xml")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True)
def export_xml(input_file: Path, output: Path):
    """Export analysis results as XML."""
    texts = _read_texts(input_file)
    
    console.print(f"[bold]Analyzing {len(texts)} texts...[/bold]")
    results = analyze_batch(texts)
    
    for text, result in zip(texts, results):
        result["text"] = text
    
    ExportManager.export(results, str(output), "xml")
    console.print(f"[green]Exported to {output}[/green]")


@export.command("formats")
def show_formats():
    """Show supported export formats."""
    formats = ExportManager.get_supported_formats()
    
    console.print("[bold]Supported export formats:[/bold]")
    for fmt in formats:
        console.print(f"  - {fmt}")


def _read_texts(input_file: Path) -> list:
    """Read texts from input file."""
    content = input_file.read_text(encoding="utf-8").strip()
    
    # Try JSON first
    try:
        texts = json.loads(content)
        if isinstance(texts, str):
            return [texts]
        elif isinstance(texts, dict):
            return list(texts.values())
        return texts
    except json.JSONDecodeError:
        pass
    
    # Fallback to line-by-line
    return [line.strip() for line in content.split("\n") if line.strip()]

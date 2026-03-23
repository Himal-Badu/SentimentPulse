"""
SentimentPulse CLI
Built by Himal Badu, AI Founder
"""

import click
from sentimentpulse import analyze_sentiment, analyze_batch
import json
import sys


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """SentimentPulse - Real-time sentiment analysis tool."""
    pass


@cli.command()
@click.argument("text", required=False)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed scores")
def analyze(text: str, verbose: bool):
    """Analyze sentiment of text."""
    if not text:
        text = click.prompt("Enter text to analyze", type=str)
    
    result = analyze_sentiment(text, verbose)
    _display_result(result)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed scores")
def batch(input_file: str, output: str, verbose: bool):
    """Analyze batch of texts from file (one per line or JSON array)."""
    # Read input
    with open(input_file, 'r') as f:
        content = f.read().strip()
    
    # Try JSON first, then fallback to line-by-line
    try:
        texts = json.loads(content)
        if isinstance(texts, str):
            texts = [texts]
    except json.JSONDecodeError:
        texts = [line.strip() for line in content.split('\n') if line.strip()]
    
    if not texts:
        click.echo("No texts found in input file", err=True)
        sys.exit(1)
    
    click.echo(f"Analyzing {len(texts)} texts...")
    results = analyze_batch(texts, verbose)
    
    # Display results
    for i, result in enumerate(results, 1):
        click.echo(f"\n[{i}] {texts[i-1][:50]}...")
        _display_result(result)
    
    if output:
        with open(output, 'w') as f:
            json.dump({"results": results, "total": len(results)}, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@cli.command()
def shell():
    """Interactive shell mode."""
    click.echo("SentimentPulse Interactive Shell")
    click.echo("Type 'exit' or 'quit' to stop\n")
    
    while True:
        try:
            text = click.prompt("Enter text", default="", show_default=False)
            
            if text.lower() in ('exit', 'quit', 'q'):
                click.echo("Goodbye!")
                break
            
            if not text.strip():
                continue
            
            result = analyze_sentiment(text)
            _display_result(result)
            click.echo()
            
        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            break


def _display_result(result: dict):
    """Display sentiment result nicely."""
    emoji = {
        "positive": "✅",
        "negative": "❌", 
        "neutral": "➖"
    }
    
    sent = result.get("sentiment", "unknown")
    score = result.get("score", 0)
    conf = result.get("confidence", 0)
    
    emoji_char = emoji.get(sent, "❓")
    conf_pct = conf * 100
    
    click.echo(f"{emoji_char} Sentiment: {sent.capitalize()} ({conf_pct:.1f}% confidence)")
    click.echo(f"   Score: {score:.4f}")
    
    if result.get("raw_scores"):
        raw = result["raw_scores"]
        click.echo(f"   Raw: pos={raw['pos']:.2f}, neg={raw['neg']:.2f}, neu={raw['neu']:.2f}")


if __name__ == "__main__":
    cli()

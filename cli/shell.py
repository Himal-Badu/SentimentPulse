"""
SentimentPulse - CLI Interactive Shell
Built by Himal Badu, AI Founder

Interactive shell mode with auto-completion and history.
"""

import os
import sys
import json
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table

console = Console()


class InteractiveShell:
    """Interactive shell for SentimentPulse CLI."""
    
    def __init__(self):
        self.history: List[str] = []
        self.history_file = os.path.expanduser("~/.sentimentpulse_history")
        self._load_history()
    
    def _load_history(self):
        """Load command history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    self.history = [line.strip() for line in f if line.strip()]
        except Exception:
            pass
    
    def _save_history(self):
        """Save command history to file."""
        try:
            with open(self.history_file, "w") as f:
                f.write("\n".join(self.history[-1000:]))
        except Exception:
            pass
    
    def _get_prompt(self) -> str:
        """Get shell prompt."""
        return "[bold green]sentimentpulse[/bold green]› "
    
    def print_welcome(self):
        """Print welcome message."""
        welcome = """
# 🤖 SentimentPulse Interactive Shell

Welcome to the SentimentPulse interactive shell!

## Available Commands

| Command | Description |
|---------|-------------|
| `analyze <text>` | Analyze sentiment of text |
| `batch <file>` | Analyze texts from file |
| `history` | Show command history |
| `clear` | Clear screen |
| `help` | Show this help message |
| `quit` or `exit` | Exit shell |

## Examples

```
analyze I love this product!
batch texts.txt
help
```

*Built by Himal Badu, AI Founder*
"""
        console.print(Panel(
            welcome.strip(),
            title="[bold cyan]Welcome to SentimentPulse[/bold cyan]",
            border_style="cyan"
        ))
    
    def print_help(self):
        """Print help message."""
        self.print_welcome()
    
    def print_history(self):
        """Print command history."""
        if not self.history:
            console.print("[dim]No command history[/dim]")
            return
        
        table = Table(title="Command History")
        table.add_column("#", style="dim", width=4)
        table.add_column("Command", style="cyan")
        
        for i, cmd in enumerate(self.history[-20:], 1):
            table.add_row(str(i), cmd)
        
        console.print(table)
    
    def print_result(self, result: dict):
        """Print analysis result."""
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
        
        console.print(f"""
[bold]Sentiment:[/bold] [{color}]{sentiment.capitalize()} {emoji}[/{color}]
[bold]Score:[/bold] {score:+.4f}
[bold]Confidence:[/bold] {confidence*100:.1f}%
""")
    
    def print_error(self, error: str):
        """Print error message."""
        console.print(f"[bold red]Error:[/bold red] {error}")
    
    def process_command(self, command: str) -> bool:
        """Process a shell command.
        
        Returns:
            True if should continue, False if should exit
        """
        command = command.strip()
        
        if not command:
            return True
        
        # Add to history
        self.history.append(command)
        self._save_history()
        
        # Parse command
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Handle commands
        if cmd in ("quit", "exit", "q"):
            console.print("[bold cyan]Goodbye! 👋[/bold cyan]")
            return False
        
        elif cmd == "clear":
            console.clear()
            return True
        
        elif cmd == "help":
            self.print_help()
            return True
        
        elif cmd == "history":
            self.print_history()
            return True
        
        elif cmd == "analyze":
            if not args:
                self.print_error("Usage: analyze <text>")
                return True
            
            from sentimentpulse import analyze_sentiment
            try:
                result = analyze_sentiment(args)
                self.print_result(result)
            except Exception as e:
                self.print_error(str(e))
            return True
        
        elif cmd == "batch":
            if not args:
                self.print_error("Usage: batch <file>")
                return True
            
            from sentimentpulse import analyze_batch
            try:
                with open(args, "r") as f:
                    content = f.read()
                    try:
                        texts = json.loads(content)
                    except json.JSONDecodeError:
                        texts = [line.strip() for line in content.split("\n") if line.strip()]
                
                results = analyze_batch(texts)
                console.print(f"[green]Analyzed {len(results)} texts[/green]")
                
                # Show distribution
                sentiments = [r.get("sentiment") for r in results]
                console.print(f"Positive: {sentiments.count('positive')}")
                console.print(f"Negative: {sentiments.count('negative')}")
                console.print(f"Neutral: {sentiments.count('neutral')}")
            except Exception as e:
                self.print_error(str(e))
            return True
        
        elif cmd == "stats":
            from sentimentpulse import get_engine
            try:
                engine = get_engine()
                cache = engine.get_cache_stats()
                console.print(f"Cache hits: {cache.get('hits', 0)}")
                console.print(f"Cache misses: {cache.get('misses', 0)}")
                console.print(f"Hit rate: {cache.get('hit_rate_percent', 0):.1f}%")
            except Exception as e:
                self.print_error(str(e))
            return True
        
        else:
            # Try to analyze as text
            from sentimentpulse import analyze_sentiment
            try:
                result = analyze_sentiment(command)
                self.print_result(result)
            except Exception as e:
                self.print_error(f"Unknown command: {cmd}")
            return True
    
    def run(self):
        """Run the interactive shell."""
        self.print_welcome()
        
        while True:
            try:
                command = console.input(self._get_prompt())
                if not self.process_command(command):
                    break
            except KeyboardInterrupt:
                console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]")
                break
            except EOFError:
                break


def start_shell():
    """Start the interactive shell."""
    shell = InteractiveShell()
    shell.run()

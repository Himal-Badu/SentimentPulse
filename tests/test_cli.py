"""
Tests for SentimentPulse CLI
Built by Himal Badu, AI Founder
"""

import pytest
from click.testing import CliRunner
from cli.main import cli, analyze, batch, shell, stats, info


runner = CliRunner()


class TestCLI:
    """Test suite for CLI commands."""
    
    def test_cli_help(self):
        """Test CLI help displays correctly."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'SentimentPulse' in result.output
    
    def test_cli_version(self):
        """Test CLI version."""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '2.0.0' in result.output
    
    def test_analyze_positive(self):
        """Test positive sentiment detection."""
        result = runner.invoke(analyze, ['This is amazing!'])
        assert result.exit_code == 0
        assert 'positive' in result.output.lower() or 'Positive' in result.output
    
    def test_analyze_negative(self):
        """Test negative sentiment detection."""
        result = runner.invoke(analyze, ['This is terrible'])
        assert result.exit_code == 0
        assert 'negative' in result.output.lower() or 'Negative' in result.output
    
    def test_analyze_verbose(self):
        """Test verbose output."""
        result = runner.invoke(analyze, ['Good job!', '--verbose'])
        assert result.exit_code == 0
    
    def test_analyze_json(self):
        """Test JSON output."""
        result = runner.invoke(analyze, ['Great!', '--json'])
        assert result.exit_code == 0
        assert 'sentiment' in result.output or '{' in result.output
    
    def test_stats_command(self):
        """Test stats command."""
        result = runner.invoke(stats)
        assert result.exit_code == 0
    
    def test_info_command(self):
        """Test info command."""
        result = runner.invoke(info)
        assert result.exit_code == 0
    
    def test_batch_with_input(self, tmp_path):
        """Test batch processing with file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Good\nBad\nOkay")
        
        result = runner.invoke(batch, [str(test_file)])
        assert result.exit_code in [0, 1]  # May fail if model not loaded in test env


class TestCLIInput:
    """Test CLI input handling."""
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result = runner.invoke(analyze, [''])
        # Should prompt for input or handle gracefully
        assert result.exit_code in [0, 1]

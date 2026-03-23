"""
Tests for SentimentPulse CLI
Built by Himal Badu, AI Founder
"""

import pytest
from click.testing import CliRunner
from cli.main import cli, analyze, batch, shell


runner = CliRunner()


def test_cli_help():
    """Test CLI help."""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'SentimentPulse' in result.output


def test_analyze_positive():
    """Test positive sentiment."""
    result = runner.invoke(analyze, ['This is amazing!'])
    assert result.exit_code == 0
    assert 'Positive' in result.output


def test_analyze_negative():
    """Test negative sentiment."""
    result = runner.invoke(analyze, ['This is terrible'])
    assert result.exit_code == 0
    assert 'Negative' in result.output


def test_analyze_verbose():
    """Test verbose output."""
    result = runner.invoke(analyze, ['Good job!', '--verbose'])
    assert result.exit_code == 0
    assert 'Raw' in result.output

"""Tests for CLI help functionality."""

import subprocess
import sys


def test_cli_help_module_invocation():
    """Test that CLI help works via module invocation."""
    result = subprocess.run(
        [sys.executable, "-m", "oncotarget_lite", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "Usage" in output or "oncotarget-lite" in output


def test_cli_help_commands_available():
    """Test that expected commands are available in help output."""
    result = subprocess.run(
        [sys.executable, "-m", "oncotarget_lite", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    
    # Check for key commands
    expected_commands = ["prepare", "train", "eval", "explain", "all"]
    for cmd in expected_commands:
        assert cmd in output


def test_cli_prepare_help():
    """Test that prepare command help works."""
    result = subprocess.run(
        [sys.executable, "-m", "oncotarget_lite", "prepare", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "processed features" in output or "Create processed" in output


def test_cli_generate_data_help():
    """Test that generate-data command help works."""
    result = subprocess.run(
        [sys.executable, "-m", "oncotarget_lite", "generate-data", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "synthetic" in output or "Generate" in output
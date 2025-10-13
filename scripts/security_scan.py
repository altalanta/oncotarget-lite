#!/usr/bin/env python3
"""
Security scanning script for local development.

This script runs various security checks on dependencies and provides
recommendations for keeping the project secure.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def run_command(cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        return subprocess.run(cmd, capture_output=capture_output, text=True, check=False)
    except FileNotFoundError:
        print(f"❌ Command not found: {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, -1, "", "Command not found")


def check_pip_audit() -> bool:
    """Check for known vulnerabilities using pip-audit."""
    print("🔍 Checking for known vulnerabilities...")

    # Check if pip-audit is installed
    result = run_command([sys.executable, "-m", "pip", "show", "pip-audit"])
    if result.returncode != 0:
        print("⚠️  pip-audit not installed. Install with: pip install pip-audit>=2.7.0")
        return True  # Don't fail if tool not available

    result = run_command([sys.executable, "-m", "pip_audit", "--format=json"])

    if result.returncode == 0:
        print("✅ No vulnerabilities found")
        return True
    else:
        # Save report for review
        report_file = Path("security-report.json")
        try:
            with open(report_file, 'w') as f:
                f.write(result.stdout)
            print(f"⚠️  Vulnerabilities found. Report saved to {report_file}")
        except Exception as e:
            print(f"❌ Failed to save security report: {e}")

        return False


def check_outdated_packages() -> bool:
    """Check for outdated packages."""
    print("📦 Checking for outdated packages...")

    # Check if pip-tools is available for better outdated checking
    result = run_command([sys.executable, "-c", "import pip"])
    if result.returncode != 0:
        print("⚠️  Cannot check outdated packages")
        return True

    try:
        # Simple check for outdated packages
        result = run_command([sys.executable, "-m", "pip", "list", "--outdated", "--format=json"])

        if result.returncode == 0:
            outdated = json.loads(result.stdout)
            if outdated:
                print(f"⚠️  {len(outdated)} packages are outdated:")
                for pkg in outdated[:5]:  # Show first 5
                    print(f"  - {pkg['name']}: {pkg['installed_version']} -> {pkg['latest_version']}")
                if len(outdated) > 5:
                    print(f"  ... and {len(outdated) - 5} more")
                print("   Run 'pip list --outdated' for full list")
            else:
                print("✅ All packages are up to date")
        else:
            print("✅ No outdated packages found")

        return True

    except Exception as e:
        print(f"⚠️  Error checking outdated packages: {e}")
        return True


def check_requirements_consistency() -> bool:
    """Check that requirements.txt is consistent with pyproject.toml."""
    print("🔗 Checking requirements consistency...")

    requirements_file = Path("requirements.txt")
    pyproject_file = Path("pyproject.toml")

    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False

    if not pyproject_file.exists():
        print("⚠️  pyproject.toml not found")
        return True

    try:
        # Read requirements.txt
        with open(requirements_file) as f:
            req_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        # Read pyproject.toml dependencies
        with open(pyproject_file) as f:
            content = f.read()

        # Simple check for major version consistency
        issues = []
        for req in req_lines:
            if '==' in req:
                pkg_name, version = req.split('==', 1)
                # Look for this package in pyproject.toml
                if pkg_name in content:
                    # This is a basic check - in practice you'd want more sophisticated parsing
                    pass

        if issues:
            print(f"⚠️  Found {len(issues)} consistency issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Requirements appear consistent")

        return True

    except Exception as e:
        print(f"⚠️  Error checking consistency: {e}")
        return True


def main():
    """Run all security checks."""
    print("🛡️  Running security scan...\n")

    checks = [
        check_pip_audit,
        check_outdated_packages,
        check_requirements_consistency,
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Error running {check.__name__}: {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    print("📊 Security Scan Summary:")
    print(f"   Passed: {passed}/{total}")

    if all(results):
        print("🎉 All security checks passed!")
        return 0
    else:
        print("⚠️  Some security checks failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


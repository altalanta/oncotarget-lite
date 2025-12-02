#!/usr/bin/env python
"""Local security scanning script for dependency vulnerabilities.

This script provides a comprehensive security audit of Python dependencies
and code, suitable for running locally before commits or in CI/CD pipelines.

Usage:
    python scripts/security_scan.py                    # Run all scans
    python scripts/security_scan.py --dependencies    # Only dependency scan
    python scripts/security_scan.py --code            # Only code scan
    python scripts/security_scan.py --fix             # Show fix suggestions
    python scripts/security_scan.py --json            # Output JSON report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Vulnerability:
    """Represents a security vulnerability."""

    package: str
    installed_version: str
    vulnerability_id: str
    severity: str
    description: str
    fix_version: str | None = None
    source: str = "unknown"


@dataclass
class ScanResult:
    """Aggregated scan results."""

    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    code_issues: list[dict[str, Any]] = field(default_factory=list)
    license_warnings: list[str] = field(default_factory=list)
    scan_time: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True


class SecurityScanner:
    """Comprehensive security scanner for Python projects."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.results = ScanResult()

    def run_pip_audit(self) -> list[Vulnerability]:
        """Run pip-audit to check for known vulnerabilities."""
        print("🔍 Running pip-audit...")
        vulnerabilities = []

        try:
            result = subprocess.run(
                ["pip-audit", "--format=json"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                print("  ✅ No vulnerabilities found by pip-audit")
                return vulnerabilities

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                for dep in data.get("dependencies", []):
                    for vuln in dep.get("vulns", []):
                        vulnerabilities.append(
                            Vulnerability(
                                package=dep.get("name", "unknown"),
                                installed_version=dep.get("version", "unknown"),
                                vulnerability_id=vuln.get("id", "unknown"),
                                severity=vuln.get("severity", "unknown"),
                                description=vuln.get("description", "")[:200],
                                fix_version=vuln.get("fix_versions", [None])[0]
                                if vuln.get("fix_versions")
                                else None,
                                source="pip-audit",
                            )
                        )
            except json.JSONDecodeError:
                # Fallback to text parsing
                if "No known vulnerabilities found" not in result.stdout:
                    print(f"  ⚠️ pip-audit output: {result.stdout[:500]}")

        except FileNotFoundError:
            print("  ⚠️ pip-audit not installed. Install with: pip install pip-audit")
        except subprocess.TimeoutExpired:
            print("  ⚠️ pip-audit timed out")
        except Exception as e:
            print(f"  ❌ pip-audit error: {e}")

        if vulnerabilities:
            print(f"  ❌ Found {len(vulnerabilities)} vulnerabilities")
        return vulnerabilities

    def run_safety(self) -> list[Vulnerability]:
        """Run Safety to check against the Safety vulnerability database."""
        print("🔍 Running Safety check...")
        vulnerabilities = []

        try:
            result = subprocess.run(
                ["safety", "check", "--output=json"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            try:
                data = json.loads(result.stdout)
                # Safety returns a list of vulnerabilities
                if isinstance(data, list):
                    for vuln in data:
                        if isinstance(vuln, dict):
                            vulnerabilities.append(
                                Vulnerability(
                                    package=vuln.get("package_name", "unknown"),
                                    installed_version=vuln.get(
                                        "installed_version", "unknown"
                                    ),
                                    vulnerability_id=vuln.get("vulnerability_id", "unknown"),
                                    severity=vuln.get("severity", "unknown"),
                                    description=vuln.get("advisory", "")[:200],
                                    fix_version=vuln.get("safe_versions", [None])[0]
                                    if vuln.get("safe_versions")
                                    else None,
                                    source="safety",
                                )
                            )
            except json.JSONDecodeError:
                if "No known security vulnerabilities found" in result.stdout:
                    print("  ✅ No vulnerabilities found by Safety")
                else:
                    print(f"  ⚠️ Safety output: {result.stdout[:500]}")

        except FileNotFoundError:
            print("  ⚠️ Safety not installed. Install with: pip install safety")
        except subprocess.TimeoutExpired:
            print("  ⚠️ Safety check timed out")
        except Exception as e:
            print(f"  ❌ Safety error: {e}")

        if vulnerabilities:
            print(f"  ❌ Found {len(vulnerabilities)} vulnerabilities")
        elif not vulnerabilities:
            print("  ✅ No vulnerabilities found by Safety")
        return vulnerabilities

    def run_bandit(self) -> list[dict[str, Any]]:
        """Run Bandit for Python-specific security issues."""
        print("🔍 Running Bandit code scan...")
        issues = []

        source_dir = self.project_root / "oncotarget_lite"
        if not source_dir.exists():
            print(f"  ⚠️ Source directory not found: {source_dir}")
            return issues

        try:
            result = subprocess.run(
                ["bandit", "-r", str(source_dir), "-f", "json", "-q"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            try:
                data = json.loads(result.stdout)
                issues = data.get("results", [])
                if issues:
                    print(f"  ⚠️ Found {len(issues)} code security issues")
                else:
                    print("  ✅ No code security issues found")
            except json.JSONDecodeError:
                if result.stdout.strip():
                    print(f"  ⚠️ Bandit output: {result.stdout[:500]}")

        except FileNotFoundError:
            print("  ⚠️ Bandit not installed. Install with: pip install bandit")
        except subprocess.TimeoutExpired:
            print("  ⚠️ Bandit scan timed out")
        except Exception as e:
            print(f"  ❌ Bandit error: {e}")

        return issues

    def check_licenses(self) -> list[str]:
        """Check for problematic licenses in dependencies."""
        print("🔍 Checking dependency licenses...")
        warnings = []

        # Licenses that may cause compliance issues
        problematic_licenses = ["GPL", "AGPL", "LGPL", "SSPL", "Commons Clause"]

        try:
            result = subprocess.run(
                ["pip-licenses", "--format=json"],
                capture_output=True,
                text=True,
                timeout=120,
            )

            try:
                licenses = json.loads(result.stdout)
                for pkg in licenses:
                    license_name = pkg.get("License", "").upper()
                    for prob in problematic_licenses:
                        if prob.upper() in license_name:
                            warnings.append(
                                f"{pkg.get('Name', 'unknown')}: {pkg.get('License', 'unknown')}"
                            )
                            break

                if warnings:
                    print(f"  ⚠️ Found {len(warnings)} packages with copyleft licenses")
                else:
                    print("  ✅ No problematic licenses found")

            except json.JSONDecodeError:
                print(f"  ⚠️ Could not parse license info")

        except FileNotFoundError:
            print("  ⚠️ pip-licenses not installed. Install with: pip install pip-licenses")
        except Exception as e:
            print(f"  ❌ License check error: {e}")

        return warnings

    def scan_dependencies(self) -> None:
        """Run all dependency vulnerability scans."""
        pip_vulns = self.run_pip_audit()
        safety_vulns = self.run_safety()

        # Deduplicate vulnerabilities by package + vulnerability_id
        seen = set()
        for vuln in pip_vulns + safety_vulns:
            key = (vuln.package, vuln.vulnerability_id)
            if key not in seen:
                seen.add(key)
                self.results.vulnerabilities.append(vuln)

    def scan_code(self) -> None:
        """Run code security scans."""
        self.results.code_issues = self.run_bandit()

    def scan_licenses(self) -> None:
        """Run license compliance check."""
        self.results.license_warnings = self.check_licenses()

    def scan_all(self) -> ScanResult:
        """Run all security scans."""
        print("\n" + "=" * 60)
        print("🔒 SECURITY SCAN REPORT")
        print("=" * 60 + "\n")

        self.scan_dependencies()
        print()
        self.scan_code()
        print()
        self.scan_licenses()

        # Determine overall success
        self.results.success = (
            len(self.results.vulnerabilities) == 0
            and len(
                [i for i in self.results.code_issues if i.get("issue_severity") == "HIGH"]
            )
            == 0
        )

        return self.results

    def print_summary(self, show_fixes: bool = False) -> None:
        """Print a summary of scan results."""
        print("\n" + "=" * 60)
        print("📊 SCAN SUMMARY")
        print("=" * 60)

        # Vulnerabilities
        if self.results.vulnerabilities:
            print(f"\n❌ Dependency Vulnerabilities: {len(self.results.vulnerabilities)}")
            for vuln in self.results.vulnerabilities:
                print(f"   • {vuln.package}=={vuln.installed_version}")
                print(f"     ID: {vuln.vulnerability_id} | Severity: {vuln.severity}")
                if show_fixes and vuln.fix_version:
                    print(f"     Fix: upgrade to {vuln.fix_version}")
        else:
            print("\n✅ No dependency vulnerabilities found")

        # Code issues
        high_severity = [
            i for i in self.results.code_issues if i.get("issue_severity") == "HIGH"
        ]
        if high_severity:
            print(f"\n❌ High-severity code issues: {len(high_severity)}")
            for issue in high_severity[:5]:  # Show first 5
                print(f"   • {issue.get('filename', 'unknown')}:{issue.get('line_number', '?')}")
                print(f"     {issue.get('issue_text', 'No description')[:80]}")
        elif self.results.code_issues:
            print(f"\n⚠️ Code issues (non-critical): {len(self.results.code_issues)}")
        else:
            print("\n✅ No code security issues found")

        # Licenses
        if self.results.license_warnings:
            print(f"\n⚠️ License warnings: {len(self.results.license_warnings)}")
            for warning in self.results.license_warnings[:5]:
                print(f"   • {warning}")
        else:
            print("\n✅ No license compliance issues")

        # Overall status
        print("\n" + "-" * 60)
        if self.results.success:
            print("✅ OVERALL: PASSED")
        else:
            print("❌ OVERALL: FAILED - Action required")
        print("-" * 60 + "\n")

    def to_json(self) -> str:
        """Export results as JSON."""
        return json.dumps(
            {
                "scan_time": self.results.scan_time,
                "success": self.results.success,
                "vulnerabilities": [
                    {
                        "package": v.package,
                        "installed_version": v.installed_version,
                        "vulnerability_id": v.vulnerability_id,
                        "severity": v.severity,
                        "description": v.description,
                        "fix_version": v.fix_version,
                        "source": v.source,
                    }
                    for v in self.results.vulnerabilities
                ],
                "code_issues": self.results.code_issues,
                "license_warnings": self.results.license_warnings,
            },
            indent=2,
        )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Security scan for Python dependencies and code"
    )
    parser.add_argument(
        "--dependencies",
        "-d",
        action="store_true",
        help="Only scan dependencies for vulnerabilities",
    )
    parser.add_argument(
        "--code",
        "-c",
        action="store_true",
        help="Only scan code for security issues",
    )
    parser.add_argument(
        "--licenses",
        "-l",
        action="store_true",
        help="Only check license compliance",
    )
    parser.add_argument(
        "--fix",
        "-f",
        action="store_true",
        help="Show fix suggestions for vulnerabilities",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write JSON results to file",
    )

    args = parser.parse_args()

    scanner = SecurityScanner()

    # Run specific scans or all
    if args.dependencies:
        scanner.scan_dependencies()
    elif args.code:
        scanner.scan_code()
    elif args.licenses:
        scanner.scan_licenses()
    else:
        scanner.scan_all()

    # Output results
    if args.json:
        print(scanner.to_json())
    else:
        scanner.print_summary(show_fixes=args.fix)

    if args.output:
        args.output.write_text(scanner.to_json())
        print(f"Results written to {args.output}")

    return 0 if scanner.results.success else 1


if __name__ == "__main__":
    sys.exit(main())

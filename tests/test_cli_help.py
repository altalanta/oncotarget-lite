import subprocess, sys

def test_cli_help_module_invocation():
    out = subprocess.run([sys.executable, "-m", "oncotarget_lite", "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "Usage" in (out.stdout + out.stderr)
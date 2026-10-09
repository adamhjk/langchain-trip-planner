import json
import os
import subprocess
import sys


def test_cli_reports_missing_model_credentials():
    process = subprocess.run(
        [sys.executable, "-m", "trip_planner", "--demo"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "ANTHROPIC_API_KEY": ""},
    )
    assert process.returncode == 2
    assert "ANTHROPIC_API_KEY" in process.stdout
    assert "# Trip itinerary report" in process.stdout


def test_json_errors_remain_machine_readable():
    process = subprocess.run(
        [sys.executable, "-m", "trip_planner", "--demo", "--format", "json"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "ANTHROPIC_API_KEY": ""},
    )
    assert process.returncode == 2
    assert json.loads(process.stdout)["status"] == "error"

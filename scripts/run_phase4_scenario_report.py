"""Run Phase 4 scenario assertions, parse results, and write a detailed text report."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from bot_training.config import PROJECT_ROOT, REPORTS_DIR

SCENARIO_LINE_RE = re.compile(r"^\[(PASS|FAIL)\]\s+(.+?):\s*(.*)$")
COMPLETED_LINE_RE = re.compile(r"^Completed\s+(\d+)\s+scenario checks with\s+(\d+)\s+failure\(s\)\.$")
SLOT_RMB_RE = re.compile(r"slot=(\d+),\s*expected=(\d+),\s*rmb=([0-9]*\.?[0-9]+)")

POTION_SCENARIO_NAME = "Step 6 - Drinking Potion"
FOOD_SCENARIO_NAME = "Step 9 - Food Eating"


@dataclass(slots=True)
class ParsedScenarioResult:
    name: str
    passed: bool
    details: str


@dataclass(slots=True)
class ParsedRunOutput:
    scenarios: list[ParsedScenarioResult]
    completed_total: int | None
    completed_failures: int | None
    warnings: list[str]


@dataclass(slots=True)
class InventoryActionSummary:
    scenario_name: str
    found: bool
    passed: bool | None
    actual_slot: int | None
    expected_slot: int | None
    rmb_probability: float | None
    detail: str


@dataclass(slots=True)
class SubprocessResult:
    command: list[str]
    return_code: int
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run assert_phase4_scenarios.py, parse outputs, and write a text summary report."
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=REPORTS_DIR / "phase4_scenario_summary.txt",
        help="Path to the generated text report.",
    )
    parser.add_argument(
        "--scenario-script",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "assert_phase4_scenarios.py",
        help="Path to the scenario assertion script to execute.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch the scenario script.",
    )
    parser.add_argument(
        "--pass-through-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments forwarded to assert_phase4_scenarios.py.",
    )
    return parser.parse_args()


def run_scenarios(python_bin: str, scenario_script: Path, pass_through_args: list[str]) -> SubprocessResult:
    command = [python_bin, str(scenario_script), *pass_through_args]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return SubprocessResult(
        command=command,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def parse_run_output(stdout: str) -> ParsedRunOutput:
    scenarios: list[ParsedScenarioResult] = []
    warnings: list[str] = []
    completed_total: int | None = None
    completed_failures: int | None = None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        scenario_match = SCENARIO_LINE_RE.match(line)
        if scenario_match is not None:
            status, name, details = scenario_match.groups()
            scenarios.append(
                ParsedScenarioResult(
                    name=name,
                    passed=(status == "PASS"),
                    details=details,
                )
            )
            continue

        completed_match = COMPLETED_LINE_RE.match(line)
        if completed_match is not None:
            completed_total = int(completed_match.group(1))
            completed_failures = int(completed_match.group(2))
            continue

        if line.startswith("[WARN]"):
            warnings.append(line)

    return ParsedRunOutput(
        scenarios=scenarios,
        completed_total=completed_total,
        completed_failures=completed_failures,
        warnings=warnings,
    )


def _extract_slot_rmb(details: str) -> tuple[int | None, int | None, float | None]:
    match = SLOT_RMB_RE.search(details)
    if match is None:
        return None, None, None
    slot = int(match.group(1))
    expected = int(match.group(2))
    rmb_prob = float(match.group(3))
    return slot, expected, rmb_prob


def summarize_inventory_action(parsed: ParsedRunOutput, scenario_name: str) -> InventoryActionSummary:
    target = next((scenario for scenario in parsed.scenarios if scenario.name == scenario_name), None)
    if target is None:
        return InventoryActionSummary(
            scenario_name=scenario_name,
            found=False,
            passed=None,
            actual_slot=None,
            expected_slot=None,
            rmb_probability=None,
            detail="Scenario line was not found in captured output.",
        )

    actual_slot, expected_slot, rmb_probability = _extract_slot_rmb(target.details)
    return InventoryActionSummary(
        scenario_name=scenario_name,
        found=True,
        passed=target.passed,
        actual_slot=actual_slot,
        expected_slot=expected_slot,
        rmb_probability=rmb_probability,
        detail=target.details,
    )


def _inventory_line(summary: InventoryActionSummary, intent: str) -> str:
    if not summary.found:
        return f"- {summary.scenario_name}: NOT FOUND in output. {summary.detail}"

    status = "PASS" if summary.passed else "FAIL"
    if summary.actual_slot is None or summary.expected_slot is None:
        return f"- {summary.scenario_name}: {status} (could not parse slot details) | {summary.detail}"

    switch_ok = summary.actual_slot == summary.expected_slot
    rmb_text = "n/a" if summary.rmb_probability is None else f"{summary.rmb_probability:.3f}"
    behavior = "correct" if switch_ok else "incorrect"
    return (
        f"- {summary.scenario_name}: {status} | {intent}: {behavior} "
        f"(actual_slot={summary.actual_slot}, expected_slot={summary.expected_slot}, rmb={rmb_text})"
    )


def build_report_text(result: SubprocessResult, parsed: ParsedRunOutput) -> str:
    total = len(parsed.scenarios)
    passed = sum(1 for scenario in parsed.scenarios if scenario.passed)
    failed = total - passed

    completed_total_text = "n/a" if parsed.completed_total is None else str(parsed.completed_total)
    completed_failures_text = "n/a" if parsed.completed_failures is None else str(parsed.completed_failures)

    potion_summary = summarize_inventory_action(parsed, POTION_SCENARIO_NAME)
    food_summary = summarize_inventory_action(parsed, FOOD_SCENARIO_NAME)

    lines: list[str] = [
        "Phase 4 Scenario Automation Report",
        "=" * 33,
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Command: {' '.join(result.command)}",
        f"Subprocess exit code: {result.return_code}",
        "",
        "Overview",
        "-" * 8,
        f"Parsed scenarios: {total}",
        f"Parsed passes: {passed}",
        f"Parsed failures: {failed}",
        f"Scenario script completion line total: {completed_total_text}",
        f"Scenario script completion line failures: {completed_failures_text}",
        "",
        "Complex Inventory Action Checks",
        "-" * 31,
        _inventory_line(
            potion_summary,
            "switch to potion slot when health is critically low",
        ),
        _inventory_line(
            food_summary,
            "switch to food slot when sustained resources are low",
        ),
        "",
        "Per-Scenario Results",
        "-" * 20,
    ]

    for scenario in parsed.scenarios:
        status = "PASS" if scenario.passed else "FAIL"
        lines.append(f"- [{status}] {scenario.name}: {scenario.details}")

    if parsed.warnings:
        lines.extend(["", "Warnings", "-" * 8])
        for warning in parsed.warnings:
            lines.append(f"- {warning}")

    if result.stderr.strip():
        lines.extend(["", "Captured stderr", "-" * 15, result.stderr.rstrip()])

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not args.scenario_script.exists():
        raise FileNotFoundError(f"Scenario script not found: {args.scenario_script}")

    subprocess_result = run_scenarios(
        python_bin=args.python_bin,
        scenario_script=args.scenario_script,
        pass_through_args=args.pass_through_args,
    )
    parsed = parse_run_output(subprocess_result.stdout)

    report_text = build_report_text(subprocess_result, parsed)
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(report_text, encoding="utf-8")

    print(f"Saved scenario report: {args.report_file.resolve()}")
    print(f"Scenario process exit code: {subprocess_result.return_code}")
    return subprocess_result.return_code


if __name__ == "__main__":
    raise SystemExit(main())


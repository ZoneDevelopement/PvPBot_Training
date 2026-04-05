from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_phase4_scenario_report.py"
SCRIPTS_DIR = ROOT / "scripts"


def _load_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("run_phase4_scenario_report", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_phase4_scenario_report module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_run_output_and_inventory_focus() -> None:
    module = _load_module()
    sample_output = "\n".join(
        [
            "[PASS] Step 2 - Chasing Enemy: forward=0.900, sprint=0.920, threshold=0.80",
            "[PASS] Step 6 - Drinking Potion: slot=1, expected=1, rmb=0.991",
            "[FAIL] Step 9 - Food Eating: slot=2, expected=3, rmb=0.455",
            "Completed 12 scenario checks with 1 failure(s).",
        ]
    )

    parsed = module.parse_run_output(sample_output)
    assert len(parsed.scenarios) == 3
    assert parsed.completed_total == 12
    assert parsed.completed_failures == 1

    potion = module.summarize_inventory_action(parsed, module.POTION_SCENARIO_NAME)
    assert potion.found is True
    assert potion.passed is True
    assert potion.actual_slot == 1
    assert potion.expected_slot == 1

    food = module.summarize_inventory_action(parsed, module.FOOD_SCENARIO_NAME)
    assert food.found is True
    assert food.passed is False
    assert food.actual_slot == 2
    assert food.expected_slot == 3


def test_build_report_text_contains_inventory_section() -> None:
    module = _load_module()
    parsed = module.parse_run_output(
        "\n".join(
            [
                "[PASS] Step 6 - Drinking Potion: slot=1, expected=1, rmb=0.991",
                "[PASS] Step 9 - Food Eating: slot=3, expected=3, rmb=0.881",
                "Completed 12 scenario checks with 0 failure(s).",
            ]
        )
    )

    result = module.SubprocessResult(
        command=["python", "scripts/assert_phase4_scenarios.py", "--allow-failures"],
        return_code=0,
        stdout="",
        stderr="",
    )
    report = module.build_report_text(result, parsed)

    assert "Complex Inventory Action Checks" in report
    assert "Step 6 - Drinking Potion" in report
    assert "Step 9 - Food Eating" in report
    assert "actual_slot=1, expected_slot=1" in report
    assert "actual_slot=3, expected_slot=3" in report




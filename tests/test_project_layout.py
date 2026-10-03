from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_uses_responsibility_based_layout():
    for relative in (
        "run.py",
        "src/driving_lab",
        "tests",
        "scripts",
        "experiments",
        "assets",
        "legacy",
        "docs/reports",
    ):
        assert (ROOT / relative).exists(), f"missing target path: {relative}"

    assert not (ROOT / "auto").exists()

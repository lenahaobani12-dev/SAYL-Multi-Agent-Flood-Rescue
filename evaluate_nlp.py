"""Report descriptive severity-label agreement on the bundled Arabic dataset.

These examples also train the optional classifier, so this is an in-sample
agreement check, NOT an independent generalization/accuracy estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

from sayl.severity import analyze


def main() -> None:
    dataset = Path(__file__).resolve().parent / "data" / "distress_messages.json"
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    results = [(int(r["expected_severity"]), int(analyze(r["text"])["severity"])) for r in rows]
    correct = sum(expected == observed for expected, observed in results)
    print(f"Labeled messages: {len(results)}")
    print(f"In-sample agreement: {correct}/{len(results)} = {correct/len(results):.1%}")
    print("Note: This is NOT an independent evaluation; a held-out test set is required for that.")
    print("Expected / Predicted distribution:")
    for level in range(1, 6):
        actual = sum(expected == level for expected, _ in results)
        matched = sum(expected == observed == level for expected, observed in results)
        print(f"  Severity {level}: matched {matched}/{actual}")


if __name__ == "__main__":
    main()

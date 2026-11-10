"""Run SAYL's scripted scenarios without a graphical interface.

Examples:
    python run_headless.py
    python run_headless.py 3 100
    python run_headless.py 2 --json scenario2_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sayl.model import SaylModel
from sayl.scenarios import SCENARIOS


def run_scenario(scenario_number: int, max_ticks: int) -> dict:
    name = list(SCENARIOS)[scenario_number - 1]
    params, events = SCENARIOS[name]()
    model = SaylModel(params, events)
    for _ in range(max_ticks):
        if model.scenario_complete():
            break
        model.step()
    return {
        "scenario": name,
        "complete": model.scenario_complete(),
        "statistics": model.stats(),
        "incidents": [
            {"id": inc.incident_id, "status": inc.status, "initial_people": inc.total_people,
             "awaiting_pickup": inc.people, "rescued": inc.rescued_people,
             "sheltered": inc.sheltered_people, "at_staging": inc.staged_people}
            for inc in model.env.incidents.values()
        ],
        "shelters": [
            {"name": sh.name, "state": sh.state, "occupied": sh.occupied,
             "reserved": sh.reserved, "capacity": sh.capacity}
            for sh in model.shelters
        ],
        "message_performatives": model.bus.performative_counts(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SAYL multi-agent flood rescue simulation.")
    parser.add_argument("scenario", type=int, choices=(1, 2, 3), nargs="?",
                        help="Scenario number (omit to run all three)")
    parser.add_argument("ticks", type=int, nargs="?", default=160,
                        help="Maximum simulated minutes per scenario (default: 160)")
    parser.add_argument("--json", type=Path, dest="output",
                        help="Optional JSON results file")
    args = parser.parse_args()
    if args.ticks <= 0:
        parser.error("ticks must be a positive integer")
    selected = [args.scenario] if args.scenario else [1, 2, 3]
    results = [run_scenario(n, args.ticks) for n in selected]
    for result in results:
        status = "COMPLETE" if result["complete"] else "INCOMPLETE"
        stats = result["statistics"]
        print(f"[{status}] {result['scenario']} | tick={stats['tick']} | rescued={stats['people_rescued']} "
              f"| sheltered={stats['people_sheltered']} | staging={stats['people_staged']} "
              f"| messages={stats['messages']}")
    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Results saved: {args.output}")
    if not all(r["complete"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
The three testing scenarios required by the project (section 4.7 of the report).

1. Normal operation       - agents working individually and together.
2. Many reports at once   - conflict: more incidents than drones/boats, small shelters.
3. Agent failure          - a drone fails mid-flight and a boat gets stuck;
                            the Command Center reassigns the tasks.

Each scenario returns (SimParams, events). An event is (tick, function(model)).

Scenario parameters and seeded events support repeatable evaluation.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .model import Event, SaylModel, SimParams


def _msg(text: str, pos=None) -> Callable[[SaylModel], None]:
    def submit_distress(model: SaylModel) -> None:
        model.submit_message(text, pos)
    return submit_distress


def scenario_normal() -> Tuple[SimParams, List[Event]]:
    params = SimParams(n_drones=3, n_boats=2, shelter_capacity=20, rain_intensity=0.4,
                       failure_prob=0.0, distress_rate=0.0)
    events: List[Event] = [
        (1, _msg("الموية دخلت البيت وعندنا كبار سن", (12, 10))),
        (4, _msg("احنا ثلاثة فوق السطح والموية ترتفع", (20, 12))),
        (10, _msg("الموية بدت تدخل الحوش", (8, 9))),
    ]
    return params, events


def scenario_many_reports() -> Tuple[SimParams, List[Event]]:
    params = SimParams(n_drones=2, n_boats=2, shelter_capacity=8, rain_intensity=0.7,
                       failure_prob=0.0, distress_rate=0.0)
    texts = [
        "الموية دخلت المحل بس احنا بخير",
        "ساعدونا البيت يغرق ومعنا رضيع",
        "الشارع مقفول بالموية",
        "فيه شخص مصاب وينزف والموية حولنا",
        "احنا ثمانية في الاستراحة والموية حاوطتنا",
        "جدتي على كرسي متحرك وما نقدر نطلع من البيت",
        "الموية دخلت الكراج بس",
        "عشرة طلاب محاصرين في المدرسة والموية ترتفع",
    ]
    events: List[Event] = [(1 + i // 3, _msg(t)) for i, t in enumerate(texts)]
    return params, events


def scenario_failure() -> Tuple[SimParams, List[Event]]:
    params = SimParams(n_drones=2, n_boats=2, shelter_capacity=20, rain_intensity=0.5,
                       failure_prob=0.0, distress_rate=0.0)

    def fail_flying_drone(m: SaylModel) -> bool:
        for d in m.drones:
            if d.state == "FLYING":
                d.fail("technical fault (scenario)")
                return True
        return False                                   # no drone flying yet: retry next tick

    def stall_boat(m: SaylModel) -> bool:
        for b in m.boats:
            if b.state == "TO_INCIDENT":
                b.get_stuck("engine failure (scenario)")
                return True
        return False                                   # no boat on its way yet: retry next tick

    events: List[Event] = [
        (1, _msg("الحقونا الموية دخلت البيت وعندنا اطفال", (18, 11))),
        (2, _msg("ثلاث اشخاص فوق السيارة في الوادي الموية تزيد", (24, 9))),
        (6, fail_flying_drone),
        (18, stall_boat),
    ]
    return params, events


SCENARIO_PURPOSES = {
    "1 - Normal operation": "Verify reports, coordinate drone and boat assignments, and deliver rescued residents to available shelters.",
    "2 - Many reports (conflict)": "Handle eight concurrent reports with limited response units and shelter capacity; demonstrate REFUSE and staging when shelters are full.",
    "3 - Agent failure": "Force one drone and one rescue boat to fail mid-task, then confirm FAILURE messages and reassignment to different agents.",
}

SCENARIOS: Dict[str, Callable[[], Tuple[SimParams, List[Event]]]] = {
    "1 - Normal operation": scenario_normal,
    "2 - Many reports (conflict)": scenario_many_reports,
    "3 - Agent failure": scenario_failure,
}

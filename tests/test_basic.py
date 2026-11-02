"""
Quick checks that the whole system still works after someone changes a file.

Run:  python -m pytest tests      (or simply: python tests/test_basic.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sayl.acl import Performative
from sayl.agents.distress_agent import analyze, normalize
from sayl.model import SaylModel, SimParams
from sayl.scenarios import SCENARIOS


def test_arabic_analysis():
    r = analyze("ساعدونا البيت يغرق ومعنا رضيع")
    assert r["severity"] >= 4
    assert "infant" in r["tags"]
    assert analyze("احنا ثمانية في الاستراحة")["people"] == 8
    assert normalize("الْمَوِيَّة") == "المويه"


def test_negation_and_safety():
    from sayl.severity import rule_analysis
    assert "injured" not in rule_analysis("ما فيه مصابين بس الموية دخلت البيت")["tags"]
    assert rule_analysis("الموية دخلت المحل بس احنا بخير")["severity"] == 1
    assert rule_analysis("شفت شخص يطفو في الموية وما يتحرك")["severity"] == 5


def test_every_scenario_rescues_people():
    for key, build in SCENARIOS.items():
        params, events = build()
        model = SaylModel(params, events)
        model.run(80)
        assert model.stats()["people_rescued"] > 0, key


def test_contract_net_messages_exist():
    params, events = SCENARIOS["1 - Normal operation"]()
    model = SaylModel(params, events)
    model.run(40)
    used = {m.performative for m in model.bus.log}
    for p in (Performative.INFORM, Performative.CFP, Performative.PROPOSE,
              Performative.ACCEPT_PROPOSAL, Performative.REQUEST, Performative.AGREE):
        assert p in used, p


def test_failure_is_reassigned():
    params, events = SCENARIOS["3 - Agent failure"]()
    model = SaylModel(params, events)
    model.run(80)
    assert any(m.performative == Performative.FAILURE for m in model.bus.log)
    assert all(not i.open for i in model.env.incidents.values())


def test_visitor_message():
    model = SaylModel(SimParams(distress_rate=0.0))
    model.submit_message("الحقونا الموية دخلت البيت وعندنا كبار سن", (12, 10))
    model.run(60)
    assert len(model.env.incidents) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)

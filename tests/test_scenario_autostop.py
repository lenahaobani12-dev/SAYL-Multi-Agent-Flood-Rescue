"""Regression tests for finite scenario completion."""

import pytest
from sayl.model import SaylModel, SimParams
from sayl.scenarios import SCENARIOS


def make_scenario(name):
    params, events = SCENARIOS[name]()
    return SaylModel(params, events)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scripted_scenario_finishes_automatically_after_real_resolution(scenario):
    m = make_scenario(scenario)
    assert not m.scenario_complete(), "No incidents yet; cannot stop at tick zero"
    for _ in range(160):
        m.step()
        if m.scenario_complete():
            break
    assert m.scenario_complete(), f"{scenario} did not finish within 160 ticks"
    assert not m.events
    assert not m.stats()["open"]
    assert all(b.state == "IDLE" and b.on_board == 0 for b in m.boats)
    assert not any(s.reserved for s in m.shelters)


def test_autostop_waits_for_future_reports():
    m = make_scenario("1 - Normal operation")
    for _ in range(3):
        m.step()
    assert m.events, "More reports scheduled at ticks 4 and 10"
    assert not m.scenario_complete()


def test_autostop_waits_for_delivery_not_just_pickup():
    m = make_scenario("1 - Normal operation")
    for _ in range(150):
        m.step()
        if m.env.incidents and all(not incident.open for incident in m.env.incidents.values()):
            # Incidents are marked RESCUED at pickup before delivery; do not
            # auto-pause until the boat returns to IDLE and shelter is resolved.
            if any(boat.on_board or boat.state != "IDLE" for boat in m.boats):
                assert not m.scenario_complete()
                return
    pytest.fail("Expected at least one picked-up-but-not-yet-delivered tick")


def test_autostop_waits_for_pending_distress_queue():
    m = make_scenario("1 - Normal operation")
    for _ in range(150):
        m.step()
        if m.scenario_complete():
            break
    assert m.scenario_complete()
    m.submit_message("عالقين في السطح والموية ترتفع", (12, 10))
    assert not m.scenario_complete()


def test_random_distress_stream_is_not_finite_scenario():
    m = make_scenario("1 - Normal operation")
    for _ in range(150):
        m.step()
        if m.scenario_complete():
            break
    assert m.scenario_complete()
    m.params.distress_rate = 0.10
    assert not m.scenario_complete()


def test_free_play_not_auto_stopped_by_ui_contract():
    # This method can be true in a manually submitted finite run, but the UI
    # limits auto-pause to SCENARIOS, never FREE_FIELD.
    assert "Free play (random messages)" not in SCENARIOS

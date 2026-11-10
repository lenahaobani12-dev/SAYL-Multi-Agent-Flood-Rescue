"""Safety regression tests for real shelter admission and multi-trip accounting."""

from sayl.acl import Ontology, Performative
from sayl.model import SaylModel, SimParams
from sayl.scenarios import SCENARIOS


def advance_until_complete(model, limit=200):
    for _ in range(limit):
        if model.scenario_complete():
            break
        model.step()
    assert model.scenario_complete(), "rescue simulation did not finish"
    assert model.stats()["undelivered_messages"] == 0
    assert not model.command.waiting_shelter
    assert not model.command.reservations
    assert all(site.reserved == 0 for site in model.shelters)
    assert all(boat.state == "IDLE" and boat.on_board == 0 for boat in model.boats)


def test_closed_shelter_is_not_used_and_passengers_are_rerouted():
    model = SaylModel(SimParams(distress_rate=0, failure_prob=0))
    model.submit_message("الحقونا الموية دخلت البيت وفيه أطفال", (18, 11))
    closed = None
    for _ in range(100):
        transit = next((b for b in model.boats if b.state == "TO_SHELTER"
                        and b.shelter != "CommandCenter"), None)
        if transit:
            closed = model.agent(transit.shelter)
            # The closure travels through the actual ACL bus, just like Flood Monitor events.
            model.flood.send(Performative.INFORM, closed.name,
                             {"event": "close", "depth": 0.3}, Ontology.FLOOD)
            break
        model.step()
    assert closed is not None, "expected a boat assigned to a physical shelter"
    advance_until_complete(model)
    case = next(iter(model.env.incidents.values()))
    assert closed.state == "CLOSED"
    assert closed.occupied == 0
    assert case.rescued_people == case.sheltered_people + case.staged_people
    assert any(msg.content.get("event") == "reroute" for msg in model.bus.log)
    assert any(msg.content.get("event") == "shelter_closed" for msg in model.bus.log)
    assert case.status == "SHELTERED" if case.staged_people == 0 else case.status == "RESCUED"


def test_multiple_boat_trips_and_temporary_staging_are_counted_separately():
    model = SaylModel(SimParams(n_drones=1, n_boats=2, shelter_capacity=5,
                                distress_rate=0, failure_prob=0))
    model.submit_message("عشرة أشخاص محاصرين في البيت والموية ترتفع", (12, 10))
    advance_until_complete(model)
    incident = next(iter(model.env.incidents.values()))
    assert incident.total_people == 10
    assert incident.people == 0
    assert incident.rescued_people == 10
    assert incident.sheltered_people == 4
    assert incident.staged_people == 6
    assert incident.status == "RESCUED"  # At staging is NOT sheltered.
    assert model.stats()["people_rescued"] == 10
    assert model.stats()["people_sheltered"] == 4
    assert model.stats()["people_staged"] == 6
    assert sum(s.occupied for s in model.shelters) == 4
    assert all(s.occupied <= s.capacity for s in model.shelters)
    trip_ids = {m.content.get("trip") for m in model.bus.log
                if m.content.get("event") == "picked_up"}
    assert len(trip_ids) == 2


def test_all_scripted_scenarios_are_fully_accounted_for_before_autostop():
    for builder in SCENARIOS.values():
        params, events = builder()
        model = SaylModel(params, events)
        advance_until_complete(model)
        for inc in model.env.incidents.values():
            assert inc.total_people == inc.rescued_people
            assert inc.rescued_people == inc.sheltered_people + inc.staged_people
        assert model.stats()["people_rescued"] == (
            model.stats()["people_sheltered"] + model.stats()["people_staged"])

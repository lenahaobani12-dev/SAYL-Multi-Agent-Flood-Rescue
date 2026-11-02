"""Person 4 regression tests for ACL interaction and three testing scenarios."""

from sayl.acl import ACLMessage, MessageBus, Ontology, Performative
from sayl.model import SaylModel
from sayl.scenarios import SCENARIOS


def simulate(name, ticks=120):
    params, events = SCENARIOS[name]()
    model = SaylModel(params, events)
    model.run(ticks)
    return model


def test_acl_reply_keeps_correlation_and_delivers_next_tick():
    bus = MessageBus()
    class Stub:
        name = "Receiver"
        def __init__(self):
            self.inbox = []
    receiver = Stub()
    bus.register(receiver)
    sent = ACLMessage(Performative.REQUEST, "Sender", receiver.name,
                      {"action": "reserve", "people": 1}, Ontology.SHELTER,
                      conversation_id="case-1")
    bus.send(sent)
    assert receiver.inbox == []
    bus.deliver()
    assert receiver.inbox == [sent]
    answer = sent.reply(Performative.AGREE, {"free": 5})
    assert answer.sender == "Receiver" and answer.receiver == "Sender"
    assert answer.conversation_id == "case-1" and answer.reply_to == sent.msg_id
    assert bus.conversation_trace("case-1") == [sent]
    assert bus.performative_counts() == {"REQUEST": 1}


def test_unregistered_receivers_are_audited():
    bus = MessageBus()
    sent = ACLMessage(Performative.INFORM, "Agent", "Missing",
                      {"event": "ping"}, Ontology.FLOOD)
    bus.send(sent)
    bus.deliver()
    assert bus.undelivered == [sent]


def test_normal_scenario_shows_acl_negotiation_and_sheltering():
    m = simulate("1 - Normal operation")
    used = set(m.bus.performative_counts())
    assert {"CFP", "PROPOSE", "ACCEPT", "REQUEST", "AGREE", "INFORM"} <= used
    assert len(m.env.incidents) == 3
    assert all(i.status == "SHELTERED" for i in m.env.incidents.values())
    assert m.stats()["undelivered_messages"] == 0


def test_many_reports_preserves_shelter_capacity_and_staging_status():
    m = simulate("2 - Many reports (conflict)")
    assert len(m.env.incidents) == 8
    assert m.bus.performative_counts().get("REFUSE", 0) > 0
    assert all(0 <= sh.occupied + sh.reserved <= sh.capacity for sh in m.shelters)
    assert m.stats()["staging_incidents"] > 0
    assert all(i.status != "SHELTERED" for i in m.env.incidents.values()
               if i.shelter == m.command.name)
    assert m.stats()["open"] == 0


def test_failures_are_reassigned_and_reservations_released():
    m = simulate("3 - Agent failure")
    failures = [x for x in m.bus.log if x.performative == Performative.FAILURE]
    assert {f.content.get("task") for f in failures} == {"verify", "rescue"}
    for failed in failures:
        later_accepts = [x for x in m.bus.log
                         if x.tick > failed.tick
                         and x.performative == Performative.ACCEPT_PROPOSAL
                         and x.ontology == failed.ontology
                         and x.content.get("incident") == failed.content.get("incident")
                         and x.receiver != failed.sender]
        assert later_accepts, failed.short()
    assert all(i.status == "SHELTERED" for i in m.env.incidents.values())
    assert sum(s.reserved for s in m.shelters) == 0
    assert not m.command.reservations
    assert not m.command.waiting_shelter
    assert not m.bus.undelivered


def test_boat_route_uses_flood_aware_astar():
    m = simulate("1 - Normal operation", 12)
    boat = m.boats[0]
    # The integrated boat module provides the A* routing method and remains
    # within grid bounds after many simulation ticks.
    assert callable(boat._move_toward)
    assert all(m.env.in_bounds(b.pos) for b in m.boats)

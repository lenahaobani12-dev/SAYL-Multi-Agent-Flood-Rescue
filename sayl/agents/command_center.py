"""Command Center Agent: coordinates verification, rescue and shelter allocation.

Rescue tasks are awarded through Contract Net (CFP, PROPOSE, ACCEPT/REJECT).
Shelter reservations are associated with individual rescue trips, not incident
IDs, so capacity accounting remains correct across multiple boat journeys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from langsmith import trace

from ..acl import ACLMessage, Ontology, Performative
from .base import BaseAgent
from .flood_agent import SHELTER_FLOOD

CRITICAL = 4
CFP_WAIT = 2


@dataclass
class OpenCall:
    conv: str
    task: str                   # verify | rescue | shelter
    incident: int
    deadline: int
    proposals: List[Tuple[str, Dict]] = field(default_factory=list)
    refusals: List[str] = field(default_factory=list)
    trip: str = ""


class CommandCenterAgent(BaseAgent):
    kind = "Command Center"
    agent_type = "Deliberative"

    def __init__(self, name, bus, env, drones: List[str], boats: List[str], shelters: List[str]) -> None:
        super().__init__(name, bus, env)
        self.drones = drones
        self.boats = boats
        self.shelters = shelters
        self.to_verify: List[int] = []
        self.to_rescue: List[int] = []
        self.calls: Dict[str, OpenCall] = {}
        # trip -> (boat, number of people, incident ID)
        self.waiting_shelter: Dict[str, tuple] = {}
        # trip -> (shelter, number of people, incident ID, boat)
        self.reservations: Dict[str, tuple] = {}
        self._conv_counter = 0
        self.decisions = 0

    def _conv(self, task: str, incident: int) -> str:
        self._conv_counter += 1
        return f"{task}-{incident}-{self._conv_counter}"

    def priority(self, incident_id: int) -> float:
        inc = self.env.incidents[incident_id]
        waiting = self.env.tick - inc.created_tick
        depth = min(self.env.depth(inc.pos), 2.0)
        return inc.severity * 10 + waiting * 0.5 + depth * 5

    def _busy_with(self, task: str, incident: int) -> bool:
        return any(c.task == task and c.incident == incident for c in self.calls.values())

    def _cancel_offers(self, trip: str, incident: int, selected: str = "") -> None:
        for shelter in self.shelters:
            if shelter != selected:
                self.send(Performative.INFORM, shelter,
                          {"event": "cancel_reservation", "trip": trip, "incident": incident},
                          Ontology.SHELTER, trip)

    def _release_reservation(self, trip: str) -> tuple | None:
        reservation = self.reservations.pop(trip, None)
        if reservation:
            self.send(Performative.INFORM, reservation[0],
                      {"event": "cancel_reservation", "trip": trip, "incident": reservation[2]},
                      Ontology.SHELTER, trip)
        return reservation

    def _reroute(self, trip: str, incident_id: int, boat: str, people: int) -> None:
        """Reassign the passengers of one interrupted trip without losing them."""
        if trip in self.waiting_shelter or any(c.task == "shelter" and c.trip == trip for c in self.calls.values()):
            return
        self._release_reservation(trip)
        self.send(Performative.INFORM, boat,
                  {"event": "reroute", "trip": trip, "incident": incident_id},
                  Ontology.SHELTER, trip)
        self._ask_shelters(incident_id, boat, people, trip)

    def handle(self, msg: ACLMessage) -> None:
        c = msg.content
        p = msg.performative
        event = c.get("event")
        if p == Performative.INFORM and event == "new_incident":
            inc = self.env.incidents[c["incident"]]
            (self.to_rescue if inc.severity >= CRITICAL else self.to_verify).append(inc.incident_id)

        elif p in (Performative.PROPOSE, Performative.REFUSE, Performative.AGREE) and msg.conversation_id in self.calls:
            call = self.calls[msg.conversation_id]
            if p == Performative.REFUSE:
                call.refusals.append(msg.sender)
            else:
                call.proposals.append((msg.sender, c))

        elif p == Performative.INFORM and event == "verified":
            inc = self.env.incidents[c["incident"]]
            if inc.rescued_people == 0:
                inc.people = max(inc.people, c.get("people_seen", inc.people))
                inc.total_people = max(inc.total_people, inc.people)
            if inc.status == "VERIFYING":
                inc.status = "VERIFIED"
                self.to_rescue.append(inc.incident_id)

        elif p == Performative.INFORM and event == "picked_up":
            inc = self.env.incidents[c["incident"]]
            taken = max(0, int(c.get("taken", 0)))
            inc.people = max(0, int(c["remaining"]))
            inc.rescued_people += taken
            if inc.people > 0:
                inc.status = "VERIFIED"
                inc.boat = None
                self.to_rescue.append(inc.incident_id)
            else:
                inc.status = "RESCUED"
                inc.rescued_tick = self.env.tick

        elif p == Performative.INFORM and event == "admitted":
            trip = str(c["trip"])
            booked = self.reservations.get(trip)
            if booked and booked[0] == msg.sender and booked[2] == c["incident"]:
                self.reservations.pop(trip, None)
                inc = self.env.incidents[c["incident"]]
                inc.sheltered_people += min(int(c["people"]), booked[1])
                self._update_incident_status(inc)

        elif p == Performative.INFORM and event == "delivered":
            inc = self.env.incidents[c["incident"]]
            if c.get("shelter") == self.name:
                inc.staged_people += max(0, int(c.get("people", 0)))
            self._update_incident_status(inc)

        elif p == Performative.INFORM and event == "shelter_unavailable":
            trip = str(c.get("trip", msg.conversation_id))
            inc_id = c["incident"]
            boat = msg.sender
            if inc_id in self.env.incidents and c.get("people", 0) > 0:
                self._reroute(trip, inc_id, boat, int(c["people"]))

        elif p == Performative.INFORM and event == "shelter_closed":
            name = c.get("shelter", msg.sender)
            for trip in c.get("affected_trips", []):
                booked = self.reservations.get(str(trip))
                if booked and booked[0] == name:
                    self._reroute(str(trip), booked[2], booked[3], booked[1])

        elif p == Performative.REFUSE and event in ("admission_refused", "reservation_unavailable"):
            # The boat retains its passengers and independently requests a reroute.
            # Release only the invalid hold here to prevent a capacity leak.
            trip = str(c.get("trip", msg.conversation_id))
            booked = self.reservations.get(trip)
            if booked and booked[0] == msg.sender:
                self._release_reservation(trip)

        elif p == Performative.FAILURE:
            inc = self.env.incidents[c["incident"]]
            if c.get("task") == "verify":
                inc.status, inc.drone = "NEW", None
                self.to_verify.insert(0, inc.incident_id)
            else:
                inc.status, inc.boat = "VERIFIED", None
                self.to_rescue.insert(0, inc.incident_id)
                trip = str(c.get("trip", msg.conversation_id))
                self.waiting_shelter.pop(trip, None)
                for conv, call in list(self.calls.items()):
                    if call.task == "shelter" and call.trip == trip:
                        del self.calls[conv]
                self._release_reservation(trip)
                self._cancel_offers(trip, inc.incident_id)
                inc.shelter = None

        elif p == Performative.INFORM and event == "severity_update":
            inc = self.env.incidents.get(c["incident"])
            if inc and inc.open:
                inc.severity = c["severity"]
                if inc.severity >= CRITICAL and inc.incident_id in self.to_verify:
                    self.to_verify.remove(inc.incident_id)
                    self.to_rescue.append(inc.incident_id)

    @staticmethod
    def _update_incident_status(inc) -> None:
        if inc.people > 0:
            return
        if inc.total_people > 0 and inc.sheltered_people >= inc.total_people:
            inc.status = "SHELTERED"
        elif inc.rescued_people >= inc.total_people:
            inc.status = "RESCUED"

    def _close_due_calls(self) -> None:
        for conv, call in list(self.calls.items()):
            if self.env.tick < call.deadline:
                continue
            del self.calls[conv]
            inc = self.env.incidents[call.incident]
            if call.task == "shelter":
                self._choose_shelter(call, inc)
                continue
            still_needed = (inc.status == "NEW") if call.task == "verify" else inc.status in ("NEW", "VERIFYING", "VERIFIED")
            if not still_needed:
                onto = Ontology.VERIFICATION if call.task == "verify" else Ontology.RESCUE
                for name, _ in call.proposals:
                    self.send(Performative.REJECT_PROPOSAL, name, {"incident": inc.incident_id}, onto, conv)
                continue
            if not call.proposals:
                (self.to_verify if call.task == "verify" else self.to_rescue).append(inc.incident_id)
                continue
            best_name, best_offer = min(call.proposals, key=lambda pr: (pr[1]["eta"], -pr[1].get("battery", 0)))

            with trace(
                "SAYL_Contract_Net_Decision",
                run_type="chain",
                inputs={
                    "tick": self.env.tick,
                    "task": call.task,
                    "incident_id": inc.incident_id,
                    "proposals": [
                        {
                            "agent": name,
                            "eta": offer["eta"],
                            "battery": offer.get("battery", 0),
                        }
                        for name, offer in call.proposals
                    ],
                },
            ) as run:
                run.end(outputs={
                    "selected_agent": best_name,
                    "selected_eta": best_offer["eta"],
                })

            onto = Ontology.VERIFICATION if call.task == "verify" else Ontology.RESCUE
            for name, _ in call.proposals:
                perf = Performative.ACCEPT_PROPOSAL if name == best_name else Performative.REJECT_PROPOSAL
                self.send(perf, name, {"incident": inc.incident_id, "pos": inc.pos, "people": inc.people},
                          onto, conv)
            self.decisions += 1
            if call.task == "verify":
                inc.status, inc.drone = "VERIFYING", best_name
            else:
                inc.status, inc.boat = "RESCUING", best_name
                self._ask_shelters(inc.incident_id, best_name,
                                   min(inc.people, best_offer.get("capacity", inc.people)), conv)

    def _ask_shelters(self, incident_id: int, boat: str, people: int, trip: str) -> None:
        conv = self._conv("shelter", incident_id)
        self.calls[conv] = OpenCall(conv, "shelter", incident_id, self.env.tick + CFP_WAIT, trip=trip)
        self.waiting_shelter[trip] = (boat, people, incident_id)
        for shelter in self.shelters:
            self.send(Performative.REQUEST, shelter,
                      {"action": "reserve", "people": people, "incident": incident_id, "trip": trip},
                      Ontology.SHELTER, conv)

    def _choose_shelter(self, call: OpenCall, inc) -> None:
        waiting = self.waiting_shelter.pop(call.trip, None)
        if waiting is None:
            self._cancel_offers(call.trip, inc.incident_id)
            return
        boat, people, incident_id = waiting
        safe = []
        for name, offer in call.proposals:
            shelter = self.bus.agents.get(name)
            if (shelter is not None and shelter.state != "CLOSED"
                    and call.trip in getattr(shelter, "_holds", {})
                    and self.env.depth(shelter.pos) < SHELTER_FLOOD):
                safe.append((name, offer))
        if safe:
            name, offer = min(safe, key=lambda pr: self.env.distance(tuple(pr[1]["pos"]), inc.pos))
            self.send(Performative.INFORM, name,
                      {"event": "confirm_reservation", "people": people,
                       "incident": incident_id, "trip": call.trip}, Ontology.SHELTER, call.conv)
            self.reservations[call.trip] = (name, people, incident_id, boat)
            target, label = tuple(offer["pos"]), name
            self._cancel_offers(call.trip, incident_id, selected=name)
        else:
            target, label = self.env.command_center_pos, self.name
            self._cancel_offers(call.trip, incident_id)
        inc.shelter = label
        self.send(Performative.INFORM, boat,
                  {"event": "go_to_shelter", "shelter": label, "pos": target,
                   "incident": incident_id, "trip": call.trip}, Ontology.SHELTER, call.conv)

    def _open_call(self, queue: List[int], task: str, receivers: List[str]) -> None:
        if not queue or any(c.task == task for c in self.calls.values()):
            return
        queue.sort(key=self.priority, reverse=True)
        incident_id = queue.pop(0)
        inc = self.env.incidents[incident_id]
        if task == "verify" and inc.status != "NEW":
            return
        if task == "rescue" and inc.status not in ("NEW", "VERIFYING", "VERIFIED"):
            return
        conv = self._conv(task, incident_id)
        self.calls[conv] = OpenCall(conv, task, incident_id, self.env.tick + CFP_WAIT)
        onto = Ontology.VERIFICATION if task == "verify" else Ontology.RESCUE
        for r in receivers:
            self.send(Performative.CFP, r, {"task": task, "incident": incident_id, "pos": inc.pos,
                                           "severity": inc.severity, "people": inc.people}, onto, conv)

    def step(self) -> None:
        super().step()
        self._close_due_calls()
        self._open_call(self.to_rescue, "rescue", self.boats)
        self._open_call(self.to_verify, "verify", self.drones)
        self.state = "COORDINATING" if (self.calls or self.to_verify or self.to_rescue) else "MONITORING"

    def status(self):
        s = super().status()
        s.update({"to_verify": len(self.to_verify), "to_rescue": len(self.to_rescue),
                  "open_calls": len(self.calls), "decisions": self.decisions})
        return s

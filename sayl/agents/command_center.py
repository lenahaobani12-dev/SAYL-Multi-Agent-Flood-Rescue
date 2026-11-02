"""
Command Center Agent: ranks incidents and assigns drones, boats and shelters.

Owner: Person 2 (Distress Agent + Command Center)
Agent type: Deliberative (it keeps a model of all incidents and resources and
plans who should do what, using the Contract Net Protocol).

Protocols used
--------------
* Contract Net (CFP -> PROPOSE / REFUSE -> ACCEPT / REJECT) to pick a drone
  for verification and a boat for the rescue.
* Request (REQUEST -> AGREE / REFUSE) to reserve shelter places.
* INFORM / FAILURE from drones and boats to follow the progress of each task.

Priority rule (simple version)
------------------------------
priority = severity * 10 + waiting_ticks * 0.5 + water_depth_m * 5
(severity matters most; waiting time prevents starvation; deeper water = more urgent)
Critical cases (severity >= 4) skip the drone verification and go straight to
the boat CFP, because waiting could cost lives.

TODO for Person 2
-----------------
* Write this agent's description, PEAS, type and behaviours in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..acl import ACLMessage, Ontology, Performative
from .base import BaseAgent

CRITICAL = 4
CFP_WAIT = 2          # ticks to wait for proposals (send at t, answers processed at t+2)


@dataclass
class OpenCall:
    conv: str
    task: str                     # "verify" | "rescue" | "shelter"
    incident: int
    deadline: int
    proposals: List[Tuple[str, Dict]] = field(default_factory=list)
    refusals: List[str] = field(default_factory=list)


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
        self.waiting_shelter: Dict[int, tuple] = {}   # incident -> (boat, people) waiting for a shelter
        self.reservations: Dict[int, tuple] = {}      # incident -> (shelter, people)
        self._conv_counter = 0
        self.decisions = 0

    # ------------------------------------------------------------------ helpers
    def _conv(self, task: str, incident: int) -> str:
        self._conv_counter += 1
        return f"{task}-{incident}-{self._conv_counter}"

    def priority(self, incident_id: int) -> float:
        inc = self.env.incidents[incident_id]
        waiting = self.env.tick - inc.created_tick
        depth = min(self.env.depth(inc.pos), 2.0)      # metres of water at the incident (capped)
        return inc.severity * 10 + waiting * 0.5 + depth * 5

    def _busy_with(self, task: str, incident: int) -> bool:
        return any(c.task == task and c.incident == incident for c in self.calls.values())

    # ------------------------------------------------------------------ messages
    def handle(self, msg: ACLMessage) -> None:
        c = msg.content
        p = msg.performative
        if p == Performative.INFORM and c.get("event") == "new_incident":
            inc = self.env.incidents[c["incident"]]
            (self.to_rescue if inc.severity >= CRITICAL else self.to_verify).append(inc.incident_id)

        elif p in (Performative.PROPOSE, Performative.REFUSE, Performative.AGREE) and msg.conversation_id in self.calls:
            call = self.calls[msg.conversation_id]
            if p == Performative.REFUSE:
                call.refusals.append(msg.sender)
            else:
                call.proposals.append((msg.sender, c))

        elif p == Performative.INFORM and c.get("event") == "verified":
            inc = self.env.incidents[c["incident"]]
            inc.people = max(inc.people, c.get("people_seen", inc.people))
            if inc.status == "VERIFYING":              # a boat may already be on its way (escalated case)
                inc.status = "VERIFIED"
                self.to_rescue.append(inc.incident_id)

        elif p == Performative.INFORM and c.get("event") == "picked_up":
            inc = self.env.incidents[c["incident"]]
            if c["remaining"] > 0:                     # boat was full: rescue the rest later
                inc.people = c["remaining"]
                inc.status = "VERIFIED"
                inc.boat = None
                self.to_rescue.append(inc.incident_id)
            else:
                inc.status = "RESCUED"
                inc.rescued_tick = self.env.tick

        elif p == Performative.INFORM and c.get("event") == "delivered":
            inc = self.env.incidents[c["incident"]]
            booked = self.reservations.pop(inc.incident_id, None)
            # The command centre is an emergency staging location, not a shelter.
            # Do not claim a person was sheltered unless a real shelter received them.
            if inc.status == "RESCUED" and booked and c.get("shelter") == booked[0]:
                inc.status = "SHELTERED"

        elif p == Performative.FAILURE:
            inc = self.env.incidents[c["incident"]]
            if c.get("task") == "verify":
                inc.status, inc.drone = "NEW", None
                self.to_verify.insert(0, inc.incident_id)      # retry first
            else:
                inc.status, inc.boat = "VERIFIED", None
                self.to_rescue.insert(0, inc.incident_id)
                # Invalidate shelter offers still pending for the failed boat.
                # Otherwise an offer can be confirmed *after* the failure and
                # reserve seats forever for a boat that no longer has this task.
                self.waiting_shelter.pop(inc.incident_id, None)
                for conv, call in list(self.calls.items()):
                    if call.task == "shelter" and call.incident == inc.incident_id:
                        del self.calls[conv]
                res = self.reservations.pop(inc.incident_id, None)
                if res:                                # free the shelter places of the failed trip
                    self.send(Performative.INFORM, res[0], {"event": "cancel_reservation", "people": res[1],
                                                           "incident": inc.incident_id}, Ontology.SHELTER)
                inc.shelter = None

        elif p == Performative.INFORM and c.get("event") == "severity_update":
            inc = self.env.incidents.get(c["incident"])
            if inc and inc.open:
                inc.severity = c["severity"]
                if inc.severity >= CRITICAL and inc.incident_id in self.to_verify:
                    self.to_verify.remove(inc.incident_id)      # escalate: skip verification
                    self.to_rescue.append(inc.incident_id)

    # ------------------------------------------------------------------ decisions
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
            if not still_needed:                       # situation changed while waiting: cancel
                onto = Ontology.VERIFICATION if call.task == "verify" else Ontology.RESCUE
                for name, _ in call.proposals:
                    self.send(Performative.REJECT_PROPOSAL, name, {"incident": inc.incident_id}, onto, conv)
                continue
            if not call.proposals:                     # nobody can do it now: retry later
                (self.to_verify if call.task == "verify" else self.to_rescue).append(inc.incident_id)
                continue
            best_name, best_offer = min(call.proposals, key=lambda pr: (pr[1]["eta"], -pr[1].get("battery", 0)))
            onto = Ontology.VERIFICATION if call.task == "verify" else Ontology.RESCUE
            for name, offer in call.proposals:
                perf = Performative.ACCEPT_PROPOSAL if name == best_name else Performative.REJECT_PROPOSAL
                self.send(perf, name, {"incident": inc.incident_id, "pos": inc.pos, "people": inc.people},
                          onto, conv)
            self.decisions += 1
            if call.task == "verify":
                inc.status, inc.drone = "VERIFYING", best_name
            else:
                inc.status, inc.boat = "RESCUING", best_name
                self._ask_shelters(inc.incident_id, best_name, min(inc.people, best_offer.get("capacity", inc.people)))

    def _ask_shelters(self, incident_id: int, boat: str, people: int) -> None:
        conv = self._conv("shelter", incident_id)
        self.calls[conv] = OpenCall(conv, "shelter", incident_id, self.env.tick + CFP_WAIT)
        self.waiting_shelter[incident_id] = (boat, people)
        for s in self.shelters:
            self.send(Performative.REQUEST, s, {"action": "reserve", "people": people,
                                               "incident": incident_id}, Ontology.SHELTER, conv)

    def _choose_shelter(self, call: OpenCall, inc) -> None:
        waiting = self.waiting_shelter.pop(inc.incident_id, None)
        if waiting is None:
            return
        boat, people = waiting
        if call.proposals:
            name, offer = min(call.proposals, key=lambda pr: self.env.distance(tuple(pr[1]["pos"]), inc.pos))
            self.send(Performative.INFORM, name, {"event": "confirm_reservation", "people": people,
                                                 "incident": inc.incident_id}, Ontology.SHELTER, call.conv)
            target, label = tuple(offer["pos"]), name
            self.reservations[inc.incident_id] = (name, people)
        else:                                          # every shelter is full: use the staging area
            target, label = self.env.command_center_pos, self.name
        inc.shelter = label
        self.send(Performative.INFORM, boat, {"event": "go_to_shelter", "shelter": label, "pos": target,
                                             "incident": inc.incident_id}, Ontology.SHELTER, call.conv)

    def _open_call(self, queue: List[int], task: str, receivers: List[str]) -> None:
        if not queue or any(c.task == task for c in self.calls.values()):
            return                                     # one open call per task type at a time
        queue.sort(key=self.priority, reverse=True)
        incident_id = queue.pop(0)
        inc = self.env.incidents[incident_id]
        if task == "verify" and inc.status != "NEW":
            return                                     # already verified / escalated
        if task == "rescue" and inc.status not in ("NEW", "VERIFYING", "VERIFIED"):
            return                                     # already being rescued or done
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

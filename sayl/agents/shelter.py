"""Shelter Agent: manages capacity and acknowledges safe arrivals.

The ACL reservation protocol uses a unique rescue-trip identifier, so multiple
boats serving the same incident never overwrite one another's reservations.
Requests create temporary capacity holds; the command centre confirms the
chosen shelter and cancels the others. A closed shelter cannot accept arrivals.
"""

from __future__ import annotations

from ..acl import ACLMessage, Ontology, Performative
from .base import BaseAgent


class ShelterAgent(BaseAgent):
    kind = "Shelter"
    agent_type = "Reactive"

    def __init__(self, name, bus, env, pos, capacity: int):
        super().__init__(name, bus, env)
        self.pos = pos
        self.capacity = capacity
        self.occupied = 0
        self.reserved = 0
        self._holds: dict[str, int] = {}
        self._confirmed: set[str] = set()
        self.state = "OPEN"

    @property
    def free(self) -> int:
        return max(0, self.capacity - self.occupied - self.reserved)

    def _release(self, trip: str) -> int:
        count = self._holds.pop(trip, 0)
        self._confirmed.discard(trip)
        self.reserved = max(0, self.reserved - count)
        return count

    def handle(self, msg: ACLMessage) -> None:
        c = msg.content
        p = msg.performative
        event = c.get("event")
        trip = str(c.get("trip", msg.conversation_id))
        if p == Performative.REQUEST and c.get("action") == "reserve":
            people = c.get("people", 0)
            if not isinstance(people, int) or isinstance(people, bool) or people <= 0:
                self.reply(msg, Performative.REFUSE, {"reason": "invalid number of people", "trip": trip})
            elif self.state == "CLOSED":
                self.reply(msg, Performative.REFUSE, {"reason": "shelter closed", "trip": trip})
            elif self.free < people and trip not in self._holds:
                self.reply(msg, Performative.REFUSE, {"reason": f"not enough capacity ({self.free} free)", "trip": trip})
            else:
                if trip not in self._holds:
                    self._holds[trip] = people
                    self.reserved += people
                self.reply(msg, Performative.AGREE, {"pos": self.pos, "free": self.free, "trip": trip})
        elif p == Performative.INFORM and event == "confirm_reservation":
            if self.state == "CLOSED" or trip not in self._holds:
                self._release(trip)
                self.send(Performative.REFUSE, msg.sender,
                          {"event": "reservation_unavailable", "trip": trip, "incident": c.get("incident")},
                          Ontology.SHELTER, msg.conversation_id)
            else:
                self._confirmed.add(trip)
        elif p == Performative.INFORM and event == "cancel_reservation":
            self._release(trip)
        elif p == Performative.INFORM and event == "arrived":
            people = max(0, int(c.get("people", 0)))
            hold = self._holds.get(trip, 0)
            # Admission is an acknowledged event, not an assumption by the boat.
            safe = self.state != "CLOSED" and people > 0 and trip in self._confirmed
            safe = safe and people <= hold and self.occupied + people <= self.capacity
            if safe:
                self._release(trip)
                self.occupied += people
                receipt = {"event": "admitted", "incident": c.get("incident"),
                           "trip": trip, "people": people, "shelter": self.name}
                self.send(Performative.INFORM, "CommandCenter", receipt,
                          Ontology.SHELTER, msg.conversation_id)
                self.send(Performative.INFORM, msg.sender, receipt,
                          Ontology.SHELTER, msg.conversation_id)
            else:
                self._release(trip)
                receipt = {"event": "admission_refused", "incident": c.get("incident"),
                           "trip": trip, "people": people, "shelter": self.name}
                self.send(Performative.REFUSE, "CommandCenter", receipt,
                          Ontology.SHELTER, msg.conversation_id)
                self.send(Performative.REFUSE, msg.sender, receipt,
                          Ontology.SHELTER, msg.conversation_id)
        elif p == Performative.INFORM and event == "close":
            if self.state != "CLOSED":
                self.state = "CLOSED"
                affected = list(self._holds)
                for held_trip in affected:
                    self._release(held_trip)
                self.send(Performative.INFORM, "CommandCenter",
                          {"event": "shelter_closed", "shelter": self.name,
                           "affected_trips": affected}, Ontology.SHELTER)

    def step(self) -> None:
        super().step()
        if self.state == "OPEN" and self.free <= 0:
            self.state = "FULL"
        elif self.state == "FULL" and self.free > 0:
            self.state = "OPEN"

    def status(self):
        s = super().status()
        s.update({"pos": self.pos, "occupied": f"{self.occupied}/{self.capacity}",
                  "reserved": self.reserved, "free": self.free})
        return s

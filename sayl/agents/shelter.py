"""
Shelter Agent: manages the free places in one shelter.

Owner: Person 3 (Drones + Boats + Shelters)
Agent type: Reactive (answers requests based on its current capacity).

Protocol: REQUEST(reserve n) -> AGREE / REFUSE, then INFORM(confirm_reservation)
from the Command Center and INFORM(arrived) from the boat.

TODO for Person 3
-----------------
* Write this agent's description, PEAS, type and behaviours in the report.
"""

from __future__ import annotations

from ..acl import ACLMessage, Performative
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
        self.state = "OPEN"

    @property
    def free(self) -> int:
        return self.capacity - self.occupied - self.reserved

    def handle(self, msg: ACLMessage) -> None:
        c = msg.content
        p = msg.performative
        if p == Performative.REQUEST and c.get("action") == "reserve":
            people = c.get("people", 0)
            if not isinstance(people, int) or isinstance(people, bool) or people <= 0:
                self.reply(msg, Performative.REFUSE, {"reason": "invalid number of people"})
            elif self.state == "CLOSED":
                self.reply(msg, Performative.REFUSE, {"reason": "shelter closed"})
            elif self.free < people:
                self.reply(msg, Performative.REFUSE, {"reason": f"not enough capacity ({self.free} free)"})
            else:
                self.reply(msg, Performative.AGREE, {"pos": self.pos, "free": self.free})
        elif p == Performative.INFORM and c.get("event") == "confirm_reservation":
            people = max(0, int(c.get("people", 0)))
            self.reserved = min(self.reserved + people, max(0, self.capacity - self.occupied))
        elif p == Performative.INFORM and c.get("event") == "cancel_reservation":
            people = max(0, int(c.get("people", 0)))
            self.reserved = max(0, self.reserved - people)
        elif p == Performative.INFORM and c.get("event") == "arrived":
            people = max(0, int(c.get("people", 0)))
            self.reserved = max(0, self.reserved - people)
            self.occupied = min(self.capacity, self.occupied + people)
        elif p == Performative.INFORM and c.get("event") == "close":
            self.state = "CLOSED"

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

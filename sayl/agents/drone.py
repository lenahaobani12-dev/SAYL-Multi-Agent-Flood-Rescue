"""
Drone Agent: flies to a reported location and confirms what is happening.

Agent type: Hybrid (reacts to battery and failures, but plans whether it has
enough battery for a round trip before it proposes).

States: IDLE -> FLYING -> RETURNING -> CHARGING -> IDLE   (or FAILED)

"""

from __future__ import annotations

from typing import Optional
from math import ceil

from ..acl import ACLMessage, Ontology, Performative
from .base import BaseAgent


class DroneAgent(BaseAgent):
    kind = "Drone"
    agent_type = "Hybrid"
    SPEED = 2                 # cells per tick
    DRAIN = 2.5               # battery % per tick of flight
    CHARGE = 12.0             # battery % per tick at base
    SAFETY_RESERVE = 10.0     # keep a reserve for unexpected movement
    REPAIR_TICKS = 5          # simulated ticks before a failed drone returns to service

    def __init__(self, name, bus, env, base, failure_prob: float = 0.0, command_center: str = "CommandCenter"):
        super().__init__(name, bus, env)
        self.pos = base
        self.base = base
        self.battery = 100.0
        self.failure_prob = failure_prob
        self.command_center = command_center
        self.target: Optional[tuple] = None
        self.incident: Optional[int] = None
        self.conv = ""
        self.verified = 0
        self.repair_left = 0

    def _travel_ticks(self, start, target) -> int:
        # step_towards moves in both axes in the same tick, so the longer
        # axis determines the number of movement ticks.
        dx = abs(start[0] - target[0])
        dy = abs(start[1] - target[1])
        return ceil(max(dx, dy) / self.SPEED)

    def _needed_battery(self, target) -> float:
        outbound = self._travel_ticks(self.pos, target)
        return_ticks = self._travel_ticks(target, self.base)
        return (outbound + return_ticks) * self.DRAIN + self.SAFETY_RESERVE

    def handle(self, msg: ACLMessage) -> None:
        c = msg.content
        if msg.performative == Performative.CFP:
            if self.state != "IDLE":
                self.reply(msg, Performative.REFUSE, {"reason": f"busy ({self.state.lower()})"})
            elif self.battery < self._needed_battery(tuple(c["pos"])):
                self.reply(msg, Performative.REFUSE, {"reason": f"low battery {self.battery:.0f}%"})
            else:
                eta = round(self.env.distance(self.pos, tuple(c["pos"])) / self.SPEED, 1)
                self.reply(msg, Performative.PROPOSE, {"eta": eta, "battery": round(self.battery)})
        elif msg.performative == Performative.ACCEPT_PROPOSAL and self.state == "IDLE":
            self.state, self.target, self.incident, self.conv = "FLYING", tuple(c["pos"]), c["incident"], msg.conversation_id

    def fail(self, reason: str) -> None:
        """Can also be called by a scenario to force a failure."""
        if self.state == "FAILED":
            return
        if self.incident is not None and self.state == "FLYING":
            self.send(Performative.FAILURE, self.command_center,
                      {"task": "verify", "incident": self.incident, "reason": reason},
                      Ontology.VERIFICATION, self.conv)
        self.state, self.target, self.incident = "FAILED", None, None
        self.repair_left = self.REPAIR_TICKS

    def step(self) -> None:
        super().step()
        if self.state == "FAILED":
            self.repair_left -= 1
            if self.repair_left <= 0:
                self.pos = self.base
                self.battery = 100.0
                self.state = "IDLE"
            return
        if self.state in ("FLYING", "RETURNING"):
            self.battery -= self.DRAIN
            if self.battery <= 0:
                self.battery = 0
                self.fail("battery empty")
                return
            if self.env.rng.random() < self.failure_prob:
                self.fail("technical fault")
                return
        if self.state == "FLYING":
            self.pos = self.env.step_towards(self.pos, self.target, self.SPEED)
            if self.pos == self.target:
                inc = self.env.incidents[self.incident]
                self.send(Performative.INFORM, self.command_center,
                          {"event": "verified", "incident": self.incident, "people_seen": inc.people,
                           "water_depth": round(self.env.depth(self.pos), 2)},
                          Ontology.VERIFICATION, self.conv)
                self.verified += 1
                self.state, self.target, self.incident = "RETURNING", self.base, None
        elif self.state == "RETURNING":
            self.pos = self.env.step_towards(self.pos, self.base, self.SPEED)
            if self.pos == self.base:
                self.state = "CHARGING"
        elif self.state == "CHARGING":
            self.battery = min(100.0, self.battery + self.CHARGE)
            if self.battery >= 100:
                self.state = "IDLE"

    def status(self):
        s = super().status()
        s.update({"pos": self.pos, "battery": f"{self.battery:.0f}%", "task": self.incident or "-",
                  "verified": self.verified, "repair_left": self.repair_left})
        return s

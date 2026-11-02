"""
Rescue Boat Agent: bids for rescue tasks, picks people up, takes them to a shelter.

Owner: Person 3 (Drones + Boats + Shelters)
Agent type: Hybrid (plans its trip, but reacts to getting stuck).

States: IDLE -> TO_INCIDENT -> (WAITING_SHELTER) -> TO_SHELTER -> IDLE   (or STUCK)

Integration note (Person 4)
---------------------------
* Movement now uses Person 1's flood-aware A* path function where available.
* Person 3's remaining report description, PEAS and behaviours are kept separate.
"""

from __future__ import annotations

from typing import Optional

from ..acl import ACLMessage, Ontology, Performative
from .base import BaseAgent


class BoatAgent(BaseAgent):
    kind = "Rescue Boat"
    agent_type = "Hybrid"
    SPEED = 2
    CAPACITY = 6
    STUCK_TICKS = 8

    def __init__(self, name, bus, env, base, failure_prob: float = 0.0, command_center: str = "CommandCenter"):
        super().__init__(name, bus, env)
        self.pos = base
        self.failure_prob = failure_prob
        self.command_center = command_center
        self.target: Optional[tuple] = None
        self.incident: Optional[int] = None
        self.shelter: Optional[str] = None
        self.shelter_pos: Optional[tuple] = None
        self.on_board = 0
        self.conv = ""
        self.stuck_left = 0
        self.rescued = 0

    def _move_toward(self, goal) -> None:
        """Follow the updated flooded route, at most SPEED adjacent cells per tick."""
        if self.pos == goal:
            return
        if hasattr(self.env, "find_path"):
            route = self.env.find_path(self.pos, goal, mode="boat")
            if route:
                self.pos = route[min(self.SPEED, len(route)) - 1]
                return
        self.pos = self.env.step_towards(self.pos, goal, self.SPEED)

    def handle(self, msg: ACLMessage) -> None:
        c = msg.content
        p = msg.performative
        if p == Performative.CFP:
            if self.state != "IDLE":
                self.reply(msg, Performative.REFUSE, {"reason": f"busy ({self.state.lower()})"})
            elif self.CAPACITY <= 0:
                self.reply(msg, Performative.REFUSE, {"reason": "no rescue capacity"})
            else:
                route = self.env.find_path(self.pos, tuple(c["pos"]), mode="boat")
                eta = round(len(route) / self.SPEED, 1) if route else round(
                    self.env.distance(self.pos, tuple(c["pos"])) / self.SPEED, 1)
                self.reply(msg, Performative.PROPOSE, {"eta": eta, "capacity": self.CAPACITY})
        elif p == Performative.ACCEPT_PROPOSAL and self.state == "IDLE":
            self.state, self.target, self.incident, self.conv = "TO_INCIDENT", tuple(c["pos"]), c["incident"], msg.conversation_id
            self.shelter = self.shelter_pos = None
        elif p == Performative.INFORM and c.get("event") == "go_to_shelter" and c["incident"] == self.incident:
            self.shelter, self.shelter_pos = c["shelter"], tuple(c["pos"])
            if self.state == "WAITING_SHELTER":
                self.state, self.target = "TO_SHELTER", self.shelter_pos

    def get_stuck(self, reason: str = "engine failure") -> None:
        """Can also be called by a scenario to force a failure."""
        if self.state != "TO_INCIDENT":
            return
        self.send(Performative.FAILURE, self.command_center,
                  {"task": "rescue", "incident": self.incident, "reason": reason}, Ontology.RESCUE, self.conv)
        self.state, self.stuck_left, self.incident, self.target = "STUCK", self.STUCK_TICKS, None, None

    def step(self) -> None:
        super().step()
        if self.state == "STUCK":
            self.stuck_left -= 1
            if self.stuck_left <= 0:
                self.state = "IDLE"
            return
        if self.state == "TO_INCIDENT":
            if self.env.rng.random() < self.failure_prob:
                self.get_stuck()
                return
            self._move_toward(self.target)
            if self.pos == self.target:
                inc = self.env.incidents[self.incident]
                # Never take more people than the boat can safely carry.
                self.on_board = max(0, min(int(inc.people), self.CAPACITY))
                remaining = max(0, inc.people - self.on_board)
                self.send(Performative.INFORM, self.command_center,
                          {"event": "picked_up", "incident": self.incident, "taken": self.on_board,
                           "remaining": remaining}, Ontology.RESCUE, self.conv)
                if self.shelter_pos:
                    self.state, self.target = "TO_SHELTER", self.shelter_pos
                else:
                    self.state = "WAITING_SHELTER"
        elif self.state == "TO_SHELTER":
            self._move_toward(self.target)
            if self.pos == self.target:
                self.send(Performative.INFORM, self.command_center,
                          {"event": "delivered", "incident": self.incident, "people": self.on_board,
                           "shelter": self.shelter}, Ontology.RESCUE, self.conv)
                if self.shelter != self.command_center:
                    self.send(Performative.INFORM, self.shelter,
                              {"event": "arrived", "people": self.on_board, "incident": self.incident},
                              Ontology.SHELTER, self.conv)
                self.rescued += self.on_board
                self.on_board, self.incident, self.target = 0, None, None
                self.state = "IDLE"

    def status(self):
        s = super().status()
        s.update({"pos": self.pos, "on_board": self.on_board, "capacity": self.CAPACITY,
                  "task": self.incident or "-", "rescued": self.rescued,
                  "stuck_left": self.stuck_left})
        return s

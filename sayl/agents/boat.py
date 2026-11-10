"""Rescue Boat Agent: negotiates rescues and transports people safely.

Each accepted rescue has a unique trip identifier. Before approaching a shelter,
the boat verifies its destination remains open and safe. If it becomes unsafe,
passengers remain aboard while the command centre selects a new destination.
"""

from __future__ import annotations

from typing import Optional

from ..acl import ACLMessage, Ontology, Performative
from .base import BaseAgent
from .flood_agent import SHELTER_FLOOD


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
        """Follow the flood-aware route, at most SPEED cells per tick."""
        if self.pos == goal:
            return
        if hasattr(self.env, "find_path"):
            route = self.env.find_path(self.pos, goal, mode="boat")
            if route:
                self.pos = route[min(self.SPEED, len(route)) - 1]
                return
        self.pos = self.env.step_towards(self.pos, goal, self.SPEED)

    def _destination_safe(self) -> bool:
        if self.shelter == self.command_center:
            return True  # Explicitly designated emergency staging site.
        site = self.bus.agents.get(self.shelter or "")
        return (site is not None and site.state != "CLOSED"
                and self.shelter_pos is not None
                and self.env.depth(self.shelter_pos) < SHELTER_FLOOD)

    def _request_reroute(self) -> None:
        self.send(Performative.INFORM, self.command_center,
                  {"event": "shelter_unavailable", "incident": self.incident,
                   "trip": self.conv, "boat": self.name, "shelter": self.shelter,
                   "people": self.on_board}, Ontology.SHELTER, self.conv)
        self.state = "WAITING_SHELTER"
        self.shelter = self.shelter_pos = self.target = None

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
            self.state, self.target, self.incident, self.conv = (
                "TO_INCIDENT", tuple(c["pos"]), c["incident"], msg.conversation_id)
            self.shelter = self.shelter_pos = None
        elif p == Performative.INFORM and c.get("event") == "reroute" and c.get("trip") == self.conv:
            if self.state in ("TO_SHELTER", "AWAITING_ADMISSION") and self.on_board:
                self.state = "WAITING_SHELTER"
                self.shelter = self.shelter_pos = self.target = None
        elif (p == Performative.INFORM and c.get("event") == "admitted"
              and c.get("trip") == self.conv and self.state == "AWAITING_ADMISSION"):
            self._complete_delivery()
        elif (p == Performative.REFUSE and c.get("event") == "admission_refused"
              and c.get("trip") == self.conv and self.state == "AWAITING_ADMISSION"):
            self._request_reroute()
        elif (p == Performative.INFORM and c.get("event") == "go_to_shelter"
              and c.get("trip") == self.conv and c.get("incident") == self.incident):
            self.shelter, self.shelter_pos = c["shelter"], tuple(c["pos"])
            if self.state == "WAITING_SHELTER":
                self.state, self.target = "TO_SHELTER", self.shelter_pos

    def _complete_delivery(self) -> None:
        self.send(Performative.INFORM, self.command_center,
                  {"event": "delivered", "incident": self.incident, "trip": self.conv,
                   "people": self.on_board, "shelter": self.shelter},
                  Ontology.RESCUE, self.conv)
        self.rescued += self.on_board
        self.on_board, self.incident, self.target = 0, None, None
        self.shelter = self.shelter_pos = None
        self.state = "IDLE"

    def get_stuck(self, reason: str = "engine failure") -> None:
        """Can also be called by a scenario to force a failure before pickup."""
        if self.state != "TO_INCIDENT":
            return
        self.send(Performative.FAILURE, self.command_center,
                  {"task": "rescue", "incident": self.incident, "trip": self.conv,
                   "reason": reason}, Ontology.RESCUE, self.conv)
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
                self.on_board = max(0, min(int(inc.people), self.CAPACITY))
                remaining = max(0, inc.people - self.on_board)
                self.send(Performative.INFORM, self.command_center,
                          {"event": "picked_up", "incident": self.incident, "trip": self.conv,
                           "taken": self.on_board, "remaining": remaining}, Ontology.RESCUE, self.conv)
                if self.shelter_pos:
                    self.state, self.target = "TO_SHELTER", self.shelter_pos
                else:
                    self.state = "WAITING_SHELTER"
        elif self.state == "TO_SHELTER":
            if not self._destination_safe():
                self._request_reroute()
                return
            self._move_toward(self.target)
            if self.pos == self.target:
                if not self._destination_safe():
                    self._request_reroute()
                    return
                if self.shelter == self.command_center:
                    self._complete_delivery()
                else:
                    self.send(Performative.INFORM, self.shelter,
                              {"event": "arrived", "people": self.on_board,
                               "incident": self.incident, "trip": self.conv},
                              Ontology.SHELTER, self.conv)
                    self.state = "AWAITING_ADMISSION"

    def status(self):
        s = super().status()
        s.update({"pos": self.pos, "on_board": self.on_board, "capacity": self.CAPACITY,
                  "task": self.incident if self.incident is not None else "—",
                  "transported": self.rescued, "stuck_left": self.stuck_left})
        return s

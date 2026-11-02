"""
SaylModel: builds the environment and all agents, and runs the simulation.

Owner: Person 4 (Communication, Scenarios, Integration)

One tick = 1 simulated minute:
  1. the environment updates the water,
  2. the bus delivers the messages sent in the previous tick,
  3. every agent runs step() (reads its inbox, decides, sends messages),
  4. scheduled scenario events are applied.

The UI (app.py) and the terminal runner (run_headless.py) only talk to this class.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .acl import MessageBus
from .agents.boat import BoatAgent
from .agents.command_center import CommandCenterAgent
from .agents.distress_agent import DistressAgent
from .agents.drone import DroneAgent
from .agents.flood_agent import FloodAgent
from .agents.shelter import ShelterAgent
from .environment import FloodEnvironment

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "distress_messages.json")


@dataclass
class SimParams:
    """Every value here can be changed from a slider in the UI."""

    n_drones: int = 3
    n_boats: int = 2
    shelter_capacity: int = 20
    rain_intensity: float = 0.5        # 0..1
    failure_prob: float = 0.0          # chance per tick that a moving drone/boat fails
    distress_rate: float = 0.15        # chance per tick that a new random message arrives
    seed: int = 7


# An event is (tick, function(model)). Scenarios use them to script what happens.
# If the function returns False, it is tried again on the next tick.
Event = Tuple[int, Callable[["SaylModel"], None]]


class SaylModel:
    def __init__(self, params: Optional[SimParams] = None, events: Optional[List[Event]] = None) -> None:
        self.params = params or SimParams()
        p = self.params
        self.env = FloodEnvironment(rain_intensity=p.rain_intensity, seed=p.seed)
        self.bus = MessageBus()
        self.events: List[Event] = sorted(events or [], key=lambda e: e[0])
        # Successful scripted scenario actions, including actions retried on
        # later ticks when the required drone/boat state was not yet reached.
        self.scenario_events_log: List[Dict[str, Any]] = []
        self.sample_messages = self._load_messages()

        base = self.env.command_center_pos
        drone_names = [f"Drone-{i + 1}" for i in range(p.n_drones)]
        boat_names = [f"Boat-{i + 1}" for i in range(p.n_boats)]
        shelter_spots = self._shelter_positions()
        shelter_names = [f"Shelter-{chr(65 + i)}" for i in range(len(shelter_spots))]

        self.distress = DistressAgent("DistressAgent", self.bus, self.env)
        self.command = CommandCenterAgent("CommandCenter", self.bus, self.env, drone_names, boat_names, shelter_names)
        self.flood = FloodAgent("FloodMonitor", self.bus, self.env, dict(zip(shelter_names, shelter_spots)))
        self.drones = [DroneAgent(n, self.bus, self.env, base, p.failure_prob) for n in drone_names]
        self.boats = [BoatAgent(n, self.bus, self.env, (base[0] + 1, base[1] + 1), p.failure_prob) for n in boat_names]
        self.shelters = [ShelterAgent(n, self.bus, self.env, pos, p.shelter_capacity)
                         for n, pos in zip(shelter_names, shelter_spots)]

    # ------------------------------------------------------------------ setup
    def _load_messages(self) -> List[str]:
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                return [m["text"] for m in json.load(f)]
        except (OSError, ValueError, KeyError):
            return ["الحقونا الموية دخلت البيت"]

    def _shelter_positions(self) -> List[Tuple[int, int]]:
        """Three shelters on the highest ground, far from each other."""
        env = self.env
        cells = sorted(((float(env.elevation[y, x]), (x, y)) for y in range(2, env.height - 2)
                        for x in range(2, env.width - 2)), reverse=True)
        chosen: List[Tuple[int, int]] = []
        for _, pos in cells:
            if all(env.distance(pos, c) > 8 for c in chosen) and env.distance(pos, env.command_center_pos) > 4:
                chosen.append(pos)
            if len(chosen) == 3:
                break
        return chosen

    # ------------------------------------------------------------------ API used by UI / scenarios
    @property
    def agents(self) -> List[Any]:
        return [self.distress, self.command, self.flood, *self.drones, *self.boats, *self.shelters]

    @property
    def tick(self) -> int:
        return self.env.tick

    def submit_message(self, text: str, pos: Optional[Tuple[int, int]] = None) -> None:
        """A resident (or an exhibition visitor) sends a distress message."""
        self.distress.submit(text, pos)

    def agent(self, name: str) -> Any:
        return self.bus.agents[name]

    def step(self) -> None:
        self.env.step()
        self.bus.tick = self.env.tick
        self.bus.deliver()
        if self.env.rng.random() < self.params.distress_rate:
            self.submit_message(self.env.rng.choice(self.sample_messages))
        for agent in self.agents:
            agent.step()
        retry: List[Event] = []
        while self.events and self.events[0][0] <= self.env.tick:
            _, action = self.events.pop(0)
            if action(self) is False:                  # the event could not happen yet: try next tick
                retry.append((self.env.tick + 1, action))
            else:
                self.scenario_events_log.append({"tick": self.env.tick, "action": action.__name__})
        self.events = sorted(self.events + retry, key=lambda e: e[0])

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    def scenario_complete(self) -> bool:
        """True once a *finite* scripted exercise has finished safely.

        This is used by the Streamlit interface to auto-pause scenarios, not by
        the free-play simulation. An empty incident list or a temporarily empty
        queue is never enough to declare victory: later scripted reports,
        in-flight rescues and shelter reservations must also be finished.
        """
        if self.params.distress_rate > 0:
            # A scenario configured to receive ongoing random reports is open-ended.
            return False
        if not self.env.incidents or self.events or self.distress.queue:
            return False
        if any(incident.open for incident in self.env.incidents.values()):
            return False
        if self.command.to_verify or self.command.to_rescue or self.command.calls:
            return False
        if self.command.waiting_shelter or self.command.reservations:
            return False
        # A case can be marked RESCUED when a boat picks people up, while
        # they are still travelling to a shelter/staging area. Wait for arrival.
        if any(boat.state != "IDLE" or boat.on_board for boat in self.boats):
            return False
        if any(shelter.reserved for shelter in self.shelters):
            return False
        return True

    # ------------------------------------------------------------------ statistics
    def stats(self) -> Dict[str, Any]:
        incs = list(self.env.incidents.values())
        done = [i for i in incs if i.rescued_tick is not None]
        times = [i.rescued_tick - i.created_tick for i in done]
        return {
            "tick": self.env.tick,
            "water_level_m": round(self.env.water_surface, 2),
            "incidents": len(incs),
            "open": sum(1 for i in incs if i.open),
            "rescued_incidents": len(done),
            "sheltered_incidents": sum(1 for i in incs if i.status == "SHELTERED"),
            "staging_incidents": sum(1 for i in incs if i.status == "RESCUED" and i.shelter == self.command.name),
            "people_rescued": sum(b.rescued for b in self.boats),
            "avg_response_ticks": round(sum(times) / len(times), 1) if times else None,
            "messages": len(self.bus.log),
            "undelivered_messages": len(self.bus.undelivered),
        }

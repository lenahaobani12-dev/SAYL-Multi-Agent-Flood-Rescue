"""
Flood Monitoring Agent: watches the water and warns the other agents.

Agent type: Reactive (perceives water depth and sends warnings by fixed rules).

Rules
-----
Every ``CHECK_EVERY`` ticks the agent reads the environment and applies four rules:

1. ESCALATE  - if the water at an open incident is deep (>= 0.8 m) the severity goes up
               by one and the Command Center is informed (INFORM severity_update).
2. CLOSE     - if water reaches a shelter (>= 0.3 m) the shelter is told to close
               (INFORM close).
3. FORECAST  - it asks the environment what the water will look like in the next
               ``FORECAST_HORIZON`` ticks. If an open incident or a shelter is going to
               reach the dangerous depth soon, the Command Center gets an early warning
               (INFORM flood_forecast with the number of ticks left). This is information
               only: the Command Center can use it, but nothing depends on it.
4. ROADS     - when the share of closed roads has changed by 5 percentage points or more
               since the last report, the Command Center is informed (INFORM roads_closed).

Nothing here uses memory of earlier states except the "already warned" sets, so the
agent stays reactive: the same water situation always gives the same warnings.
"""

from __future__ import annotations

from typing import Dict, List

from ..acl import Ontology, Performative
from .base import BaseAgent

DEEP = 0.8                 # metres: deep water at an incident -> severity + 1
SHELTER_FLOOD = 0.3        # metres: water at a shelter -> close it
CHECK_EVERY = 3            # ticks between two checks
FORECAST_HORIZON = 12      # ticks looked ahead
ROAD_REPORT_STEP = 0.05    # report when the closed-road share changes by this much


class FloodAgent(BaseAgent):
    kind = "Flood Monitor"
    agent_type = "Reactive"

    def __init__(self, name, bus, env, shelters: Dict[str, tuple], command_center: str = "CommandCenter"):
        super().__init__(name, bus, env)
        self.shelters = shelters
        self.command_center = command_center
        self.escalated: set = set()
        self.closed: List[str] = []
        self.warnings = 0
        self.forecast_warned: set = set()        # ("incident", id) / ("shelter", name) already warned
        self.forecasts = 0
        self.last_road_report = 0.0
        self.road_reports = 0

    # ------------------------------------------------------------------ rules
    def _escalate_and_close(self) -> None:
        for inc in self.env.open_incidents():
            if inc.incident_id in self.escalated or inc.severity >= 5:
                continue
            if self.env.depth(inc.pos) >= DEEP:
                self.escalated.add(inc.incident_id)
                self.warnings += 1
                self.send(Performative.INFORM, self.command_center,
                          {"event": "severity_update", "incident": inc.incident_id,
                           "severity": inc.severity + 1, "depth": round(self.env.depth(inc.pos), 2)},
                          Ontology.FLOOD)
        for name, pos in self.shelters.items():
            if name not in self.closed and self.env.depth(pos) >= SHELTER_FLOOD:
                self.closed.append(name)
                self.warnings += 1
                self.send(Performative.INFORM, name, {"event": "close", "depth": round(self.env.depth(pos), 2)},
                          Ontology.FLOOD)

    def _forecast(self) -> None:
        """Warn early about places that will become dangerous within FORECAST_HORIZON ticks."""
        targets = []                                              # (key, label, pos, limit)
        for inc in self.env.open_incidents():
            if inc.incident_id not in self.escalated and inc.severity < 5:
                targets.append((("incident", inc.incident_id), f"incident {inc.incident_id}", inc.pos, DEEP))
        for name, pos in self.shelters.items():
            if name not in self.closed:
                targets.append((("shelter", name), name, pos, SHELTER_FLOOD))
        targets = [t for t in targets if t[0] not in self.forecast_warned]
        if not targets:
            return
        depth = self.env.water.copy()
        pending = {t[0]: t for t in targets if self.env.depth(t[2]) < t[3]}
        for k in range(1, FORECAST_HORIZON + 1):
            depth = self.env._advance(depth, self.env.rain_intensity)
            for key, (_, label, pos, limit) in list(pending.items()):
                if depth[pos[1], pos[0]] >= limit:
                    del pending[key]
                    self.forecast_warned.add(key)
                    self.forecasts += 1
                    self.send(Performative.INFORM, self.command_center,
                              {"event": "flood_forecast", "target": label, "ticks": k,
                               "depth_now": round(self.env.depth(pos), 2),
                               "depth_then": round(float(depth[pos[1], pos[0]]), 2)},
                              Ontology.FLOOD)

    def _roads(self) -> None:
        rate = self.env.road_closure_rate()
        if abs(rate - self.last_road_report) >= ROAD_REPORT_STEP:
            self.last_road_report = rate
            self.road_reports += 1
            self.send(Performative.INFORM, self.command_center,
                      {"event": "roads_closed", "closed_cells": len(self.env.closed_roads()),
                       "closed_share": f"{rate:.0%}"}, Ontology.FLOOD)

    # ------------------------------------------------------------------ agent loop
    def step(self) -> None:
        super().step()
        self.state = "MONITORING"
        if self.env.tick % CHECK_EVERY:
            return
        before = self.warnings + self.forecasts + self.road_reports
        self._escalate_and_close()
        self._forecast()
        self._roads()
        if self.warnings + self.forecasts + self.road_reports > before:
            self.state = "WARNING"

    def status(self):
        s = super().status()
        s.update({"water_level": f"{self.env.water_surface:.2f} m",
                  "closed_roads": f"{self.env.road_closure_rate():.0%}",
                  "warnings": self.warnings, "forecasts": self.forecasts})
        return s

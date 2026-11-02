"""
Distress Agent: reads Arabic distress messages and turns them into incidents.

Owner: Person 2 (Distress Agent + Command Center)
Agent type: Reactive. Each message is mapped straight to a severity (rules + a
trained classifier); the agent keeps no plan and no model of the other incidents.

The language analysis itself lives in sayl/severity.py:
* normalisation, keyword levels, negation ("ما فيه مصابين"), safety phrases,
* people count, and an optional TF-IDF + Logistic Regression classifier.
Run `python evaluate_nlp.py` to see its accuracy on data/distress_messages.json.

TODO for Person 2
-----------------
* Add more labelled messages to data/distress_messages.json (the ML part improves with data).
* Write this agent's description, PEAS, type and behaviours in the report.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

from ..acl import Ontology, Performative
from ..environment import Cell
from ..severity import analyze, normalize  # noqa: F401  (normalize re-exported for tests)
from .base import BaseAgent


class DistressAgent(BaseAgent):
    kind = "Distress (NLP)"
    agent_type = "Reactive"

    def __init__(self, name, bus, env, command_center: str = "CommandCenter") -> None:
        super().__init__(name, bus, env)
        self.command_center = command_center
        self.queue: Deque[Tuple[str, Optional[Cell]]] = deque()
        self.processed = 0
        self.per_tick = 3            # how many messages it can read per tick
        self.last_result: dict = {}

    def submit(self, text: str, pos: Optional[Cell] = None) -> None:
        """Called by the UI / scenarios when a resident sends a message."""
        self.queue.append((text, pos))

    def step(self) -> None:
        super().step()
        self.state = "ANALYZING" if self.queue else "IDLE"
        for _ in range(min(self.per_tick, len(self.queue))):
            text, pos = self.queue.popleft()
            result = analyze(text)
            inc = self.env.add_incident(text, pos)
            inc.severity = int(result["severity"])
            inc.people = int(result["people"])
            inc.tags = list(result["tags"])
            self.processed += 1
            self.last_result = {"incident": inc.incident_id, **result}
            self.send(Performative.INFORM, self.command_center,
                      {"event": "new_incident", "incident": inc.incident_id, "pos": inc.pos,
                       "severity": inc.severity, "people": inc.people},
                      Ontology.DISTRESS, conversation_id=f"inc-{inc.incident_id}")

    def status(self):
        s = super().status()
        s.update({"queue": len(self.queue), "processed": self.processed})
        if self.last_result:
            s["last"] = f"#{self.last_result['incident']} sev {self.last_result['severity']}"
        return s

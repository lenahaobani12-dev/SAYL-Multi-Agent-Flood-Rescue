"""
Base class for every SAYL agent.

Shared file (all members). Please do not change the method names without
telling the group, because the model and the UI call them.

Every agent:
* has a unique ``name`` and a ``kind`` (e.g. "Drone"),
* reads its ``inbox`` (ACL messages delivered by the bus),
* runs ``step()`` once per tick,
* exposes ``status()`` so the UI can show its live state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from ..acl import ACLMessage, MessageBus, Ontology, Performative

if TYPE_CHECKING:
    from ..environment import FloodEnvironment


class BaseAgent:
    kind = "Agent"
    agent_type = "Reactive"   # Reactive / Deliberative / Hybrid (for the report and the UI)

    def __init__(self, name: str, bus: MessageBus, env: "FloodEnvironment") -> None:
        self.name = name
        self.bus = bus
        self.env = env
        self.inbox: List[ACLMessage] = []
        self.state = "IDLE"
        self.alive = True
        bus.register(self)

    # ---------------------------------------------------------------- messaging
    def send(self, performative: Performative, receiver: str, content: Dict[str, Any],
             ontology: Ontology, conversation_id: str = "") -> ACLMessage:
        msg = ACLMessage(performative, self.name, receiver, content, ontology, conversation_id)
        self.bus.send(msg)
        return msg

    def reply(self, original: ACLMessage, performative: Performative, content: Dict[str, Any]) -> None:
        self.bus.send(original.reply(performative, content))

    def take_inbox(self) -> List[ACLMessage]:
        msgs, self.inbox = self.inbox, []
        return msgs

    # ---------------------------------------------------------------- lifecycle
    def step(self) -> None:
        """Called once per tick. Override in each agent."""
        for msg in self.take_inbox():
            self.handle(msg)

    def handle(self, msg: ACLMessage) -> None:
        """Override to react to one message."""

    def status(self) -> Dict[str, Any]:
        """Live status shown in the UI table. Agents can add more fields."""
        return {"agent": self.name, "kind": self.kind, "type": self.agent_type, "state": self.state}

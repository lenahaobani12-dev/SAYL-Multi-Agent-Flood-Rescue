"""
ACL-like messaging for SAYL.

Agents communicate through the shared ACLMessage and MessageBus interfaces.

How it works
------------
* Agents never call each other directly. They put an ACLMessage on the bus.
* Messages sent during tick t are delivered at the start of tick t + 1.
  This makes the simulation predictable and easy to show in the message log.
* Every message is also stored in ``bus.log`` so the UI can display it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Any, Dict, List, Optional


class Performative(str, Enum):
    """FIPA-ACL performatives used in SAYL."""

    REQUEST = "REQUEST"            # ask another agent to do something
    INFORM = "INFORM"              # share a fact (new incident, task done, failure, ...)
    CFP = "CFP"                    # call for proposals: who can do this task?
    PROPOSE = "PROPOSE"            # answer to a CFP with an offer (ETA, battery, ...)
    ACCEPT_PROPOSAL = "ACCEPT"     # the best proposal wins the task
    REJECT_PROPOSAL = "REJECT"     # the other proposals lose
    REFUSE = "REFUSE"              # "I cannot do this" (busy, full, low battery)
    AGREE = "AGREE"                # "I accept your request"
    FAILURE = "FAILURE"            # "I started the task but could not finish it"


class Ontology(str, Enum):
    """What the message is about. Used in the report table (4.6)."""

    DISTRESS = "flood-distress"
    VERIFICATION = "incident-verification"
    RESCUE = "rescue-task"
    SHELTER = "shelter-allocation"
    FLOOD = "flood-conditions"


_msg_ids = count(1)


@dataclass
class ACLMessage:
    performative: Performative
    sender: str
    receiver: str
    content: Dict[str, Any]
    ontology: Ontology
    conversation_id: str = ""
    reply_to: Optional[int] = None
    tick: int = 0
    msg_id: int = field(default_factory=lambda: next(_msg_ids))

    def reply(self, performative: Performative, content: Dict[str, Any]) -> "ACLMessage":
        """Create an answer to this message (same ontology and conversation)."""
        return ACLMessage(
            performative=performative,
            sender=self.receiver,
            receiver=self.sender,
            content=content,
            ontology=self.ontology,
            conversation_id=self.conversation_id,
            reply_to=self.msg_id,
        )

    def short(self) -> str:
        """One-line text for the message log."""
        info = ", ".join(f"{k}={v}" for k, v in self.content.items())
        return f"[t={self.tick}] {self.performative.value:<7} {self.sender} -> {self.receiver} ({self.ontology.value}) {info}"


class MessageBus:
    """Delivers messages between agents and keeps the message log."""

    def __init__(self) -> None:
        self.agents: Dict[str, Any] = {}
        self._outbox: List[ACLMessage] = []
        self.log: List[ACLMessage] = []
        # Messages to an unregistered receiver are audited rather than crashing
        # the simulation. This makes communication failures observable during testing.
        self.undelivered: List[ACLMessage] = []
        self.tick = 0

    def register(self, agent: Any) -> None:
        self.agents[agent.name] = agent

    def unregister(self, name: str) -> None:
        self.agents.pop(name, None)

    def send(self, msg: ACLMessage) -> None:
        msg.tick = self.tick
        self._outbox.append(msg)
        self.log.append(msg)

    def deliver(self) -> None:
        """Move all queued messages into the receivers' inboxes."""
        pending, self._outbox = self._outbox, []
        for msg in pending:
            agent = self.agents.get(msg.receiver)
            if agent is not None:
                agent.inbox.append(msg)
            else:
                self.undelivered.append(msg)

    def conversation_trace(self, conversation_id: str) -> List[ACLMessage]:
        """Ordered ACL messages for one task/incident conversation."""
        return [m for m in self.log if m.conversation_id == conversation_id]

    def performative_counts(self) -> Dict[str, int]:
        """Count exchanged message types for tests and report section 4.6."""
        counts: Dict[str, int] = {}
        for message in self.log:
            key = message.performative.value
            counts[key] = counts.get(key, 0) + 1
        return counts

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentFact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    identity: str
    role: str
    capabilities: List[str] = Field(default_factory=list)
    endpoint: str
    policies: Dict[str, Any] = Field(default_factory=dict)
    jurisdiction: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DiscoveryQuery(BaseModel):
    intent: Optional[str] = None
    role: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    materials: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    max_results: int = 5


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, AgentFact] = {}

    def add(self, fact: AgentFact) -> None:
        self._agents[fact.id] = fact

    def all(self) -> List[AgentFact]:
        return list(self._agents.values())

    def discover(self, query: DiscoveryQuery) -> List[Dict[str, Any]]:
        intent_terms = set(_tokenize(query.intent or ""))
        capability_terms = set(_tokenize(" ".join(query.capabilities)))
        material_terms = set(_tokenize(" ".join(query.materials)))

        results: List[Dict[str, Any]] = []
        for agent in self._agents.values():
            score = 0.0
            rationale: List[str] = []

            if query.role and agent.role.lower() == query.role.lower():
                score += 3.0
                rationale.append(f"role={agent.role}")

            agent_caps = set(_tokenize(" ".join(agent.capabilities)))
            cap_overlap = capability_terms & agent_caps
            if cap_overlap:
                score += 2.0 + 0.5 * len(cap_overlap)
                rationale.append(f"capabilities match: {', '.join(sorted(cap_overlap))}")

            agent_materials = set(_tokenize(" ".join(agent.metadata.get("materials", []))))
            mat_overlap = material_terms & agent_materials
            if mat_overlap:
                score += 2.0 + 0.5 * len(mat_overlap)
                rationale.append(f"materials match: {', '.join(sorted(mat_overlap))}")

            if query.jurisdiction:
                agent_j = str(agent.jurisdiction.get("jurisdiction", "")).lower()
                if agent_j == query.jurisdiction.lower():
                    score += 2.0
                    rationale.append(f"jurisdiction={agent_j}")

            if query.location:
                agent_country = str(agent.jurisdiction.get("country", "")).lower()
                if agent_country == query.location.lower():
                    score += 1.5
                    rationale.append(f"location={agent_country}")

            if intent_terms:
                agent_terms = set(
                    _tokenize(agent.identity) + _tokenize(" ".join(agent.capabilities))
                )
                overlap = intent_terms & agent_terms
                if overlap:
                    score += 1.0 + 0.25 * len(overlap)
                    rationale.append(f"intent overlap: {', '.join(sorted(overlap))}")

            if score > 0:
                results.append(
                    {
                        "agent": agent,
                        "score": round(score, 2),
                        "rationale": rationale,
                    }
                )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[: max(query.max_results, 1)]


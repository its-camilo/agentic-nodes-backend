from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from registry import AgentRegistry

logger = logging.getLogger(__name__)


MCP_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "planning": {
        "type": "object",
        "properties": {"plan": {"type": "object"}, "buyer_location": {"type": "string"}},
        "required": ["plan"],
    },
    "travel_dispatch": {
        "type": "object",
        "properties": {"travel_agents": {"type": "array"}},
        "required": ["travel_agents"],
    },
    "port_negotiation": {
        "type": "object",
        "properties": {"port_agents": {"type": "array"}},
        "required": ["port_agents"],
    },
    "instruction_update": {
        "type": "object",
        "properties": {"instructions": {"type": "string"}, "updates": {"type": "object"}},
        "required": ["instructions"],
    },
    "demand_signal": {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "product": {"type": "string"},
            "materials": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"material": {"type": "string"}, "qty": {"type": "number"}},
                    "required": ["material", "qty"],
                },
            },
        },
        "required": ["intent", "product", "materials"],
    },
    "supplier_validation": {
        "type": "object",
        "properties": {"validated_suppliers": {"type": "array"}, "trust_logic": {"type": "array"}},
        "required": ["validated_suppliers"],
    },
    "logistics_routing": {
        "type": "object",
        "properties": {"routes": {"type": "array"}, "disruptions": {"type": "array"}},
        "required": ["routes"],
    },
    "negotiation": {
        "type": "object",
        "properties": {"terms": {"type": "array"}, "total_cost_estimate": {"type": "number"}},
        "required": ["terms"],
    },
    "execution_plan": {
        "type": "object",
        "properties": {"steps": {"type": "array"}, "timeline_days": {"type": "number"}, "risk_score": {"type": "number"}},
        "required": ["steps"],
    },
}



class MCPMessage(BaseModel):
    schema_version: str = "mcp-0.1"
    type: str
    source: str
    target: str
    timestamp: float
    payload: Dict[str, Any]
    trace: Dict[str, Any] = Field(default_factory=dict)


def build_mcp_message(
    msg_type: str, source: str, target: str, payload: Dict[str, Any],
    trace: Optional[Dict[str, Any]] = None,
) -> MCPMessage:
    return MCPMessage(type=msg_type, source=source, target=target, timestamp=time.time(), payload=payload, trace=trace or {})


# ---------------------------------------------------------------------------
# LLM Client — optimized retry logic
# ---------------------------------------------------------------------------

class LLMClient:
    PROVIDER_OPENAI = "openai"
    PROVIDER_OPENROUTER = "openrouter"

    def __init__(self) -> None:
        self.openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        self.openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        self.openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        raw = (os.getenv("LLM_PROVIDER") or self.PROVIDER_OPENROUTER).strip().lower()
        self._provider = raw if raw in (self.PROVIDER_OPENAI, self.PROVIDER_OPENROUTER) else self.PROVIDER_OPENROUTER
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
        return self._http_client

    @property
    def enabled(self) -> bool:
        return bool(self.openrouter_key) if self._provider == self.PROVIDER_OPENROUTER else bool(self.openai_key)

    @property
    def provider(self) -> str:
        return self._provider

    def _strip_code_fences(self, text: str) -> str:
        if "```" not in text:
            return text
        m = re.search(r"```(?:\w+)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            lines = inner.split("\n", 1)
            return lines[1].strip() if len(lines) > 1 and lines[0].strip().isalpha() else inner.strip()
        return text

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text or not text.strip():
            return None
        cleaned = self._strip_code_fences(text).strip()
        if not cleaned:
            return None
        # Direct parse
        try:
            p = json.loads(cleaned)
            if isinstance(p, dict):
                return p
        except json.JSONDecodeError:
            pass
        # Extract { ... }
        i, j = cleaned.find("{"), cleaned.rfind("}")
        if i != -1 and j > i:
            candidate = cleaned[i:j + 1]
            try:
                p = json.loads(candidate)
                if isinstance(p, dict):
                    return p
            except json.JSONDecodeError:
                pass
            try:
                p = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
                if isinstance(p, dict):
                    return p
            except json.JSONDecodeError:
                pass
        logger.warning("[LLM] JSON parse failed (len=%d): %s", len(text), text[:200])
        return None

    async def _call_api(
        self, base_url: str, api_key: str, model: str, system: str, prompt: str,
        force_json: bool = False, timeout_seconds: float = 60.0, max_tokens: int = 3500,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Returns (parsed_json_or_None, failure_reason: 'ok'|'timeout'|'parse'|'error')."""
        headers = {"Authorization": f"Bearer {api_key}"}
        is_new = self._provider == self.PROVIDER_OPENAI and any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if is_new:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["temperature"] = 0.2
            payload["max_tokens"] = max_tokens
        if force_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            client = await self._get_http_client()
            logger.info("[LLM] %s model=%s json=%s t=%.0fs tok=%d", base_url, model, force_json, timeout_seconds, max_tokens)
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=httpx.Timeout(timeout_seconds, connect=10.0))
            if resp.status_code >= 300:
                logger.error("[LLM] %d: %s", resp.status_code, resp.text[:200])
                return None, "error"
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            finish = data.get("choices", [{}])[0].get("finish_reason", "?")
            if not content:
                return None, "error"
            if finish == "length":
                logger.warning("[LLM] Truncated len=%d", len(content))
            logger.info("[LLM] OK finish=%s len=%d", finish, len(content))
            result = self._try_parse_json(content)
            return (result, "ok") if result else (None, "parse")
        except httpx.TimeoutException:
            logger.error("[LLM] Timeout %.0fs", timeout_seconds)
            return None, "timeout"
        except httpx.HTTPError as exc:
            logger.error("[LLM] HTTP: %s", exc)
            return None, "error"

    async def reason_json(
        self, system: str, prompt: str, max_retries: int = 2, max_tokens: int = 3500,
    ) -> Optional[Dict[str, Any]]:
        """Optimized retry: on timeout, increase timeout (no point retrying without force_json).
        On parse failure, retry once without force_json. Saves 1 wasted API call per timeout."""
        base_url, api_key, model = self._resolve_provider()
        if not api_key:
            logger.warning("[LLM] No API key for %s", self._provider)
            return None

        for attempt in range(1, max_retries + 1):
            timeout_s = 60.0 + (attempt - 1) * 25.0
            logger.info("[LLM] attempt %d/%d", attempt, max_retries)

            result, reason = await self._call_api(
                base_url, api_key, model, system, prompt,
                force_json=True, timeout_seconds=timeout_s, max_tokens=max_tokens,
            )
            if result is not None:
                return result

            # On parse failure only: retry without force_json (model may work better freeform)
            if reason == "parse":
                logger.info("[LLM] Parse fail → retry freeform")
                result, _ = await self._call_api(
                    base_url, api_key, model, system,
                    prompt + "\nOutput valid JSON only.",
                    force_json=False, timeout_seconds=timeout_s, max_tokens=max_tokens,
                )
                if result is not None:
                    return result

            # On timeout: just increase timeout next attempt (no freeform retry — same call will timeout)
            if attempt < max_retries:
                await asyncio.sleep(1)

        logger.error("[LLM] All %d attempts failed", max_retries)
        return None

    def _resolve_provider(self) -> tuple:
        if self._provider == self.PROVIDER_OPENAI and self.openai_key:
            return self.openai_base, self.openai_key, self.openai_model
        if self._provider == self.PROVIDER_OPENROUTER and self.openrouter_key:
            return self.openrouter_base, self.openrouter_key, self.openrouter_model
        return "", "", ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text).strip()


def _match_product(intent: str, products: List[Dict[str, Any]]) -> Dict[str, Any]:
    intent_norm = _normalize(intent)
    for product in products:
        if _normalize(product["name"]) in intent_norm:
            return product
    tokens = set(intent_norm.split())
    best, best_score = products[0], 0
    for product in products:
        score = len(tokens & set(_normalize(product["name"]).split()))
        if score > best_score:
            best_score, best = score, product
    return best


def _latest_price_index(price_history: List[Dict[str, Any]]) -> float:
    if not price_history:
        return 1.0
    return max(float(price_history[-1].get("index", 100)) / 100.0, 0.1)


def _trust_score(supplier: Dict[str, Any], compliance_flags: List[str]) -> Tuple[float, List[str]]:
    score = 50.0
    rationale: List[str] = []
    certs = supplier.get("certifications", [])
    if "ISO9001" in certs:
        score += 10; rationale.append("ISO9001")
    if "IATF16949" in certs:
        score += 12; rationale.append("IATF16949")
    if supplier.get("policies", {}).get("sustainability") == "high":
        score += 6; rationale.append("sustainability")
    if compliance_flags:
        score -= 15 + 2 * len(compliance_flags)
        rationale.append("flags:" + ",".join(compliance_flags))
    return max(0.0, min(100.0, score)), rationale


@dataclass
class DemandSignal:
    intent: str
    product: Dict[str, Any]
    materials: List[Dict[str, Any]]


class ProcurementAgent:
    def __init__(self, registry: AgentRegistry, world: Dict[str, Any], llm: LLMClient):
        self.registry = registry
        self.world = world
        self.llm = llm
        self.identity = "procurement-agent"

    async def demand_signal(self, intent: str) -> DemandSignal:
        if not self.llm or not self.llm.enabled:
            raise ValueError("Demand parsing requires an enabled LLM")
        products = self.world.get("products", [])
        product = _match_product(intent, products) if products else {"name": "custom", "bom": []}
        system = "Parse procurement intent→product+BOM. JSON:{intent,product,materials:[{material,qty}]}"
        prod_names = [p["name"] for p in products] if products else []
        matched_bom = json.dumps(product.get("bom", []), separators=(",", ":"))
        prompt = f"Intent:{intent}\nProducts:{','.join(prod_names)}\nMatched:{product.get('name','custom')}\nBOM:{matched_bom}"
        llm_data = await self.llm.reason_json(system, prompt, max_tokens=800)
        if not llm_data or "materials" not in llm_data or not isinstance(llm_data.get("materials"), list):
            raise ValueError("LLM did not return valid demand materials")
        materials = llm_data["materials"]
        product = {"name": llm_data.get("product", product.get("name", "custom")), "bom": materials}
        return DemandSignal(intent=intent, product=product, materials=materials)


class SupplierAgent:
    def __init__(self, registry: AgentRegistry, world: Dict[str, Any]):
        self.registry = registry; self.world = world; self.identity = "supplier-agent"

    def validate_suppliers(self, demand: DemandSignal, jurisdiction: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        suppliers = self.world.get("suppliers", [])
        laws = self.world.get("laws", [])
        validated, trust_logic = [], []
        for item in demand.materials:
            material = item["material"]
            for supplier in [s for s in suppliers if material in s.get("materials", [])]:
                flags = [law.get("name") for law in laws if law.get("jurisdiction") == supplier.get("jurisdiction") and "Supplier" in law.get("restricted_roles", ["Supplier"]) and material in law.get("restricted_materials", []) and law.get("name") not in supplier.get("compliance", [])]
                trust, rationale = _trust_score(supplier, flags)
                trust_logic.append({"supplier_id": supplier["id"], "material": material, "trust_score": trust, "rationale": rationale, "flags": flags})
                validated.append({"material": material, "supplier": supplier, "trust_score": trust, "compliance_flags": flags})
        if jurisdiction:
            validated = [v for v in validated if v["supplier"].get("jurisdiction") == jurisdiction]
        return validated, trust_logic


ROUTE_SELECTION_SYSTEM = (
    "You are a logistics route optimizer. Given suppliers and their candidate routes, pick the BEST route per supplier (by risk, cost, transit time, disruptions). "
    "You MUST respond with a single JSON object containing exactly one key: \"routes\". "
    "The value of \"routes\" must be an array of objects. Each object MUST have: supplier_id (string), route_id (string, one of the candidate route ids provided). "
    "Optional fields per item: material, status (e.g. \"planned\"), rationale. "
    "Example format: {\"routes\":[{\"supplier_id\":\"sup-cn-01\",\"route_id\":\"route-cn-us-01\",\"material\":\"steel-hrcoil\",\"status\":\"planned\"}]}. "
    "Return only valid JSON with the \"routes\" key. One entry per supplier. Use the exact supplier_id and route_id from the input."
)


class LogisticsAgent:
    def __init__(self, registry: AgentRegistry, world: Dict[str, Any], llm: Optional[LLMClient] = None):
        self.registry = registry; self.world = world; self.llm = llm; self.identity = "logistics-agent"

    async def route_plan_ai(self, validated_suppliers: List[Dict[str, Any]], destination_country: str, disruptions: Optional[List[Dict[str, Any]]] = None, intent: str = "") -> List[Dict[str, Any]]:
        """Route selection is AI-only. Requires enabled LLM."""
        if not self.llm or not self.llm.enabled:
            raise ValueError("Route selection requires an enabled LLM")

        all_routes = self.world.get("routes", [])
        closed = {d.get("location") for d in (disruptions or []) if d.get("type") == "port_closure"}

        candidates_by_supplier: Dict[str, List[Dict[str, Any]]] = {}
        for item in validated_suppliers:
            s = item["supplier"]
            sid = s["id"]
            opts = [r for r in all_routes if r.get("from") == s.get("country") and r.get("to") == destination_country and not closed.intersection(set(r.get("ports", [])))]
            if opts:
                compact_opts = [{"id": r["id"], "p": r.get("ports", []), "td": r.get("transit_days"), "rs": r.get("risk_score"), "ci": r.get("cost_index")} for r in opts[:3]]
                candidates_by_supplier[sid] = compact_opts

        if not candidates_by_supplier:
            return [{"material": item["material"], "supplier_id": item["supplier"]["id"], "route": None, "status": "no_route"} for item in validated_suppliers]

        sup_list = [{"s": item["supplier"]["id"], "m": item["material"], "c": item["supplier"].get("country")} for item in validated_suppliers]
        prompt_parts = [f"Dest:{destination_country}", f"Intent:{intent[:80]}"]
        if disruptions:
            compact_dis = [{"t": d.get("type"), "l": d.get("location"), "sv": d.get("severity")} for d in disruptions[:5]]
            prompt_parts.append(f"Disruptions:{json.dumps(compact_dis, separators=(',', ':'))}")
        prompt_parts.append(f"Suppliers:{json.dumps(sup_list, separators=(',', ':'))}")
        prompt_parts.append(f"Candidates:{json.dumps(candidates_by_supplier, separators=(',', ':'))}")

        data = await self.llm.reason_json(ROUTE_SELECTION_SYSTEM, "\n".join(prompt_parts), max_tokens=1500)
        if not data:
            raise ValueError("LLM did not return valid route selection")

        # Tolerate alternative shapes: "routes" | "route_selection" | raw array
        routes_list = data.get("routes")
        if routes_list is None:
            routes_list = data.get("route_selection") or data.get("selected_routes")
        if routes_list is None and isinstance(data, list):
            routes_list = data
        if not isinstance(routes_list, list):
            logger.warning("[LogisticsAgent] LLM response missing routes array: keys=%s", list(data.keys()) if isinstance(data, dict) else "not-dict")
            raise ValueError("LLM did not return valid route selection")

        route_map = {r["id"]: r for r in all_routes}
        ai_plan = []
        ai_routes_by_sid = {r.get("supplier_id"): r for r in routes_list if r.get("supplier_id")}

        for item in validated_suppliers:
            s = item["supplier"]
            sid = s["id"]
            ai_r = ai_routes_by_sid.get(sid)
            if ai_r and ai_r.get("route_id") and ai_r["route_id"] in route_map:
                full_route = route_map[ai_r["route_id"]]
                ai_plan.append({"material": item["material"], "supplier_id": sid, "route": full_route, "status": ai_r.get("status", "planned")})
            else:
                ai_plan.append({"material": item["material"], "supplier_id": sid, "route": None, "status": "no_route"})

        logger.info("[LogisticsAgent] AI selected %d routes", len([r for r in ai_plan if r.get("route")]))
        return ai_plan


class NegotiationAgent:
    def __init__(self, world: Dict[str, Any], llm: LLMClient):
        self.world = world; self.llm = llm; self.identity = "negotiation-agent"

    async def negotiate(self, demand: DemandSignal, validated_suppliers: List[Dict[str, Any]], constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Negotiation is AI-only. Requires enabled LLM."""
        if not self.llm.enabled:
            raise ValueError("Negotiation requires an enabled LLM")

        constraints = constraints or {}
        price_indices = self.world.get("price_indices", {})
        base_terms = []
        for item in validated_suppliers:
            mat, s = item["material"], item["supplier"]
            qty = next((m["qty"] for m in demand.materials if m["material"] == mat), 0)
            pi = _latest_price_index(price_indices.get(mat, []))
            bp, mk = s.get("base_price", 100.0), s.get("markup", 1.1)
            base_terms.append({"m": mat, "s": s["id"], "q": qty, "p": round(bp * pi * mk, 2), "lt": s.get("lead_time_days", 30)})

        system = "Negotiator. Volume discounts+lead time. JSON:{terms:[{material,supplier_id,qty,unit_price_est,subtotal,currency,lead_time_days}],total_cost_estimate:N}"
        prompt = f"T:{json.dumps(base_terms, separators=(',', ':'))}\nI:{demand.intent[:80]}"
        if constraints:
            prompt = f"C:{json.dumps(constraints, separators=(',', ':'))}\n{prompt}"
        d = await self.llm.reason_json(system, prompt, max_tokens=1500)
        if not d or "terms" not in d:
            raise ValueError("LLM did not return valid negotiation terms")
        terms = d["terms"]
        total = float(d.get("total_cost_estimate", 0) or sum(t.get("subtotal", 0) for t in terms))
        return {"terms": terms, "total_cost_estimate": round(total, 2)}


class ExecutionPlanner:
    def __init__(self, world: Dict[str, Any]):
        self.world = world; self.identity = "execution-planner"

    def create_execution_plan(self, demand: DemandSignal, routes: List[Dict[str, Any]], negotiation: Dict[str, Any]) -> Dict[str, Any]:
        supplier_by_id = {s["id"]: s.get("name", s["id"]) for s in self.world.get("suppliers", [])}
        steps, total_days, risk = [], 0.0, 0.0
        for r in routes:
            sid = r.get("supplier_id", "")
            supplier_name = supplier_by_id.get(sid, sid)
            mat = r.get("material", "")
            if r["route"] is None:
                steps.append({
                    "material": mat, "supplier_id": sid, "supplier_name": supplier_name,
                    "action": "escalate_no_route",
                    "description": f"Escalate: no route for {mat} from {supplier_name}",
                })
            else:
                ri = r["route"]
                t = ri.get("transit_days", 0)
                total_days = max(total_days, t)
                risk += ri.get("risk_score", 0.0)
                ports = ri.get("ports", [])
                ports_str = " → ".join(ports) if ports else "—"
                steps.append({
                    "material": mat, "supplier_id": sid, "supplier_name": supplier_name,
                    "action": "ship",
                    "route_id": ri.get("id", f"route-{sid}-{mat}"),
                    "transit_days": t,
                    "description": f"Ship {mat} from {supplier_name} via {ports_str} ({t}d)",
                })
        risk = min(1.0, risk / max(len(routes), 1))
        terms_count = len(negotiation.get("terms", []))
        total_cost = negotiation.get("total_cost_estimate")
        steps.append({
            "action": "finalize_contracts",
            "terms_count": terms_count,
            "total_cost_estimate": total_cost,
            "description": f"Finalize contracts: {terms_count} terms, total ${total_cost:,.2f}" if total_cost is not None else f"Finalize contracts: {terms_count} terms",
        })
        return {"steps": steps, "timeline_days": total_days, "risk_score": round(risk, 2)}


# ---------------------------------------------------------------------------
# System prompts — schema hints merged in, maximally concise
# ---------------------------------------------------------------------------

INSTRUCTION_UPDATE_SYSTEM = (
    "Execution coordinator. Apply instructions to the plan. "
    "JSON:{routes:[...],agentic_process:{...},execution_plan:{...},progress_notes:[{agent,text}]}"
)



# ---------------------------------------------------------------------------
# Apply instructions
# ---------------------------------------------------------------------------

async def apply_instructions_ai(
    llm: LLMClient, instructions: str, simulation: Dict[str, Any],
    target_port: Optional[str] = None, material: Optional[str] = None,
) -> Dict[str, Any]:
    if not llm.enabled:
        raise ValueError("Instructions require an enabled LLM")
    parts = [f"I:{instructions}"]
    if target_port:
        parts.append(f"P:{target_port}")
    if material:
        parts.append(f"M:{material}")
    # Compact route summary: only material, supplier_id, route_id, transit_days, status
    compact_routes = [{"m": r.get("material"), "s": r.get("supplier_id"), "r": (r.get("route") or {}).get("id"), "td": (r.get("route") or {}).get("transit_days"), "st": r.get("status")} for r in simulation.get("routes", [])]
    parts.append(f"R:{json.dumps(compact_routes, separators=(',', ':'))}")
    ap = simulation.get("agentic_process", {})
    compact_ap = {"bl": ap.get("buyer_location"), "pa": ap.get("planning_agent", {}).get("agent_id"), "ta": len(ap.get("travel_agents", [])), "pa_n": len(ap.get("port_agents", []))}
    parts.append(f"A:{json.dumps(compact_ap, separators=(',', ':'))}")
    data = await llm.reason_json(INSTRUCTION_UPDATE_SYSTEM, "\n".join(parts), max_tokens=2000)
    if not data:
        raise ValueError("LLM failed instruction update")
    return data


# ---------------------------------------------------------------------------
# Dependency graph builder
# ---------------------------------------------------------------------------

def build_dependency_graph(
    demand: DemandSignal, validated_suppliers: List[Dict[str, Any]],
    routes: List[Dict[str, Any]], agentic_process: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    nodes = [
        {"id": "procurement-agent", "type": "agent", "label": "Procurement"},
        {"id": "supplier-agent", "type": "agent", "label": "Supplier"},
        {"id": "logistics-agent", "type": "agent", "label": "Logistics"},
        {"id": "negotiation-agent", "type": "agent", "label": "Negotiation"},
        {"id": "execution-planner", "type": "agent", "label": "Execution"},
    ]
    node_ids = {n["id"] for n in nodes}

    def add(nid: str, nt: str, lbl: str) -> None:
        if nid and nid not in node_ids:
            nodes.append({"id": nid, "type": nt, "label": lbl}); node_ids.add(nid)

    edges = []
    if agentic_process:
        pid = (agentic_process.get("planning_agent") or {}).get("agent_id", "planning-agent")
        add(pid, "agent", "Planning")
        edges.append({"source": "procurement-agent", "target": pid, "type": "planning"})
        for i, t in enumerate(agentic_process.get("travel_agents") or []):
            tid = t.get("agent_id") or f"travel-{i+1}"
            add(tid, "agent", "Travel"); edges.append({"source": pid, "target": tid, "type": "dispatch"})
        for i, p in enumerate(agentic_process.get("port_agents") or []):
            paid = p.get("agent_id") or f"port-{i+1}"
            add(paid, "agent", "Port"); edges.append({"source": pid, "target": paid, "type": "handoff"})
            edges.append({"source": paid, "target": "negotiation-agent", "type": "negotiation"})
    for item in validated_suppliers:
        s = item["supplier"]
        add(s["id"], "supplier", s.get("name", s["id"]))
        edges.append({"source": "procurement-agent", "target": s["id"], "type": "demand"})
        edges.append({"source": s["id"], "target": "logistics-agent", "type": "route_request"})
    for r in routes:
        if r.get("route"):
            edges.append({"source": "logistics-agent", "target": "execution-planner", "type": "routing"}); break
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# AI-powered summary generation
# ---------------------------------------------------------------------------

def _normalize_summary_text(text: str) -> str:
    """Fix common mojibake and formatting so the summary is well-formed UTF-8."""
    if not text or not isinstance(text, str):
        return text
    # UTF-8 interpreted as Latin-1 / Windows-1252 (use ASCII/escapes for portability)
    em_dash = "\u2014"
    replacements = (
        ("\u00e2\u20ac\u2014", em_dash),   # em dash
        ("\u00e2\u20ac\u2019", "\u2019"),  # right single quote
        ("\u00e2\u20ac\u201c", '"'),       # left double quote
        ("\u00e2\u20ac\u201d", '"'),       # right double quote
        ("\u00e2\u20ac\u2026", "..."),     # ellipsis
        ("\u00e2\u20ac", em_dash),        # truncated em dash
    )
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    # Single \u00e2 (â) between word chars often is corrupted em-dash
    out = re.sub(r"(\w)\u00e2(\w)", r"\1" + em_dash + r"\2", out)
    # Normalize line breaks and trim trailing whitespace per line
    lines = [line.rstrip() for line in out.splitlines()]
    return "\n".join(lines).strip()


async def generate_summary_ai(llm: LLMClient, report: Dict[str, Any]) -> str:
    """Summary is AI-only. Requires enabled LLM."""
    if not llm.enabled:
        raise ValueError("Summary generation requires an enabled LLM")

    parts = []
    parts.append(f"Request: {report.get('intent', '')[:120]}")
    parts.append(f"Buyer: {report.get('buyer_location', 'Unknown')}")
    mfg = report.get("manufacturer", {})
    if mfg.get("name"):
        parts.append(f"Manufacturer: {mfg['name']} ({mfg.get('country', '?')})")
    discovery = report.get("discovery_paths", [])
    trust_logic = report.get("trust_logic", [])
    sup_lines = []
    for path in discovery:
        mat = path.get("material", "?")
        cands = path.get("candidates", [])
        if cands:
            best = cands[0]
            trust_entry = next((t for t in trust_logic if t.get("supplier_id") == best.get("agent_id") and t.get("material") == mat), {})
            score = trust_entry.get("trust_score", best.get("score", "?"))
            sup_lines.append(f"{mat}:{best.get('identity','?')}(trust:{score})")
    if sup_lines:
        parts.append(f"Suppliers: {'; '.join(sup_lines)}")
    routes = report.get("routes", [])
    active = [r for r in routes if r.get("route")]
    route_lines = [f"{r.get('material','?')}:{r['route'].get('from','?')}->{','.join(r['route'].get('ports',[]))}->{r['route'].get('to','?')} {r['route'].get('transit_days','?')}d risk:{r['route'].get('risk_score',0)}" for r in active]
    if route_lines:
        parts.append(f"Routes: {'; '.join(route_lines)}")
    neg = report.get("negotiation", {})
    terms = neg.get("terms", [])
    term_lines = [f"{t.get('material','?')}:{t.get('qty',0)}@${t.get('unit_price_est',0)}" for t in terms]
    parts.append(f"Negotiation: {'; '.join(term_lines)} Total:${neg.get('total_cost_estimate', 0)}")
    ep = report.get("execution_plan", {})
    parts.append(f"Timeline:{ep.get('timeline_days','?')}d Risk:{ep.get('risk_score','?')}")

    prompt = "\n".join(parts)
    summary_instruction = (
        "Context: This simulation uses LOCAL world data, AI-powered route selection, and AI negotiation agents. "
        "Write a detailed English summary of the results. You MUST include specific figures and percentages throughout. "
        "Include: "
        "(1) overview and buyer/manufacturer; "
        "(2) suppliers and trust (cite trust scores or counts where relevant); "
        "(3) shipping routes (chosen by AI) with concrete numbers: transit days, risk as a percentage, and port names; "
        "(4) negotiation terms and total cost with explicit amounts: per-material unit prices, subtotals, and total in currency; "
        "(5) a dedicated paragraph 'Negotiation benefits' with explicit numbers and percentages: e.g. 'X% volume discount', '$Y saved vs list', 'Z% lower risk', 'N days lead time reduction'; "
        "(6) execution timeline (in days) and risk as a percentage. "
        "The full summary, including negotiation benefits, must be generated by you (AI). Use real numbers from the data; do not use placeholders. "
        "Respond with JSON only: {\"summary\": \"...your full text...\"}"
    )
    data = await llm.reason_json(summary_instruction, prompt, max_tokens=1200)
    if not data:
        raise ValueError("LLM did not return valid summary")
    summary = data.get("summary")
    if summary is None:
        summary = data.get("summary_text") or data.get("text")
    if not summary or not isinstance(summary, str):
        raise ValueError("LLM did not return valid summary")
    return _normalize_summary_text(summary)

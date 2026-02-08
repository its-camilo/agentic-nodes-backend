from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from agents import (
    MCP_SCHEMAS,
    ExecutionPlanner,
    LLMClient,
    LogisticsAgent,
    NegotiationAgent,
    ProcurementAgent,
    SupplierAgent,
    apply_instructions_ai,
    build_dependency_graph,
    build_mcp_message,
    generate_summary_ai,
)
from registry import AgentFact, AgentRegistry, DiscoveryQuery
from world_loader import generate_world_with_ai, load_world, lookup_buyer_coordinates

# Maximum number of simulations kept in memory before cleanup
MAX_SIMULATIONS = 200
# Maximum age of a simulation in seconds (2 hours)
MAX_SIMULATION_AGE_SECONDS = 7200
# Maximum length for intent text
MAX_INTENT_LENGTH = 2000


def _load_env_file(path: Optional[str] = None) -> None:
    """Load .env file and ALWAYS overwrite os.environ so changes take effect on reload."""
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / ".env"
    if not env_path.is_file():
        env_path = Path(path or ".env").resolve()
        if not env_path.is_file():
            logger.warning("[Env] .env file not found at %s or %s", script_dir / ".env", env_path)
            return
    logger.info("[Env] Loading .env from %s", env_path)
    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] in ("'", '"') and value[-1:] == value[0]:
            value = value[1:-1]
        else:
            if "#" in value:
                value = value.split("#", 1)[0].rstrip()
        if key:
            os.environ[key] = value
            loaded += 1
    logger.info("[Env] Loaded %d variables", loaded)


class Disruption(BaseModel):
    type: str
    location: str
    severity: Optional[str] = None
    description: Optional[str] = None


class IntentRequest(BaseModel):
    intent: str = Field(..., max_length=MAX_INTENT_LENGTH)
    jurisdiction: Optional[str] = None
    buyer_location: Optional[str] = None
    disruptions: List[Disruption] = Field(default_factory=list)
    simulate_disruptions: bool = False
    constraints: Dict[str, Any] = Field(default_factory=dict)


class IntentResponse(BaseModel):
    trace_id: str
    report: Dict[str, Any]
    summary: Optional[str] = None


class InstructionRequest(BaseModel):
    text: str = Field(..., max_length=MAX_INTENT_LENGTH)
    target_port: Optional[str] = None
    material: Optional[str] = None


class EventBus:
    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue] = []

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def publish(self, event: Dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass


event_bus = EventBus()


def _cleanup_old_simulations(simulations: Dict[str, Any]) -> None:
    """Remove expired simulations to prevent memory leaks."""
    now = time.time()
    expired = [
        tid for tid, sim in simulations.items()
        if now - sim.get("updated_at", 0) > MAX_SIMULATION_AGE_SECONDS
    ]
    for tid in expired:
        del simulations[tid]
    if len(simulations) > MAX_SIMULATIONS:
        sorted_sims = sorted(simulations.items(), key=lambda x: x[1].get("updated_at", 0))
        for tid, _ in sorted_sims[:len(simulations) - MAX_SIMULATIONS]:
            del simulations[tid]


def _build_agentic_process_from_routes(
    buyer_location: str, validated_suppliers: List[Dict[str, Any]], routes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build an agentic_process structure from local agent route data (for map/graph rendering)."""
    planning_agent = {"agent_id": "planning-agent", "plan_summary": "Route optimization based on trust, cost, and lead time", "chosen_ports": []}
    travel_agents: List[Dict[str, Any]] = []
    port_agents: List[Dict[str, Any]] = []
    port_conditions: List[Dict[str, Any]] = []
    instruction_slots: List[Dict[str, Any]] = []
    seen_ports: set = set()

    for i, r in enumerate(routes):
        ri = r.get("route")
        if not ri:
            continue
        sid = r.get("supplier_id", "")
        mat = r.get("material", "")
        ports = ri.get("ports", [])
        if ports:
            meeting_port = ports[0]
            travel_agents.append({
                "agent_id": f"travel-{i+1}", "supplier_id": sid,
                "meeting_port": meeting_port, "status": "arrived",
                "eta_days": ri.get("transit_days", 0),
            })
            for port_name in ports:
                port_agents.append({
                    "agent_id": f"port-{port_name.replace(' ', '-').lower()}-{i+1}",
                    "port": port_name, "supplier_id": sid, "status": "negotiated",
                })
                if port_name not in seen_ports:
                    seen_ports.add(port_name)
                    port_conditions.append({"port": port_name, "issues": [], "risk_level": "low"})
            planning_agent["chosen_ports"].append({"material": mat, "port": meeting_port})
            instruction_slots.append({"material": mat, "port": meeting_port, "agent_id": f"travel-{i+1}", "status": "confirmed"})

    return {
        "buyer_location": buyer_location,
        "default_country": "US",
        "planning_agent": planning_agent,
        "travel_agents": travel_agents,
        "port_agents": port_agents,
        "port_conditions": port_conditions,
        "instruction_slots": instruction_slots,
    }



def _build_map_data(report: Dict[str, Any], sim: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-process all map elements (pins, routes, agents) for frontend rendering."""
    world_ctx = report.get("world_context", sim.get("world_context", {}))
    buyer_coords = world_ctx.get("buyer_coordinates", {})
    countries = world_ctx.get("countries", [])
    trust_logic = report.get("trust_logic", sim.get("trust_logic", []))
    routes = report.get("routes", sim.get("routes", []))
    validated_suppliers = sim.get("validated_suppliers", [])
    agentic = report.get("agentic_process", sim.get("agentic_process", {}))
    negotiation = report.get("negotiation", sim.get("negotiation", {}))
    buyer_location = report.get("buyer_location", sim.get("buyer_location", "Unknown"))

    # --- Buyer pin ---
    b_lat = float(buyer_coords.get("lat", 0) or 0)
    b_lng = float(buyer_coords.get("lng", 0) or 0)
    buyer_pin = {
        "id": "buyer", "type": "buyer", "label": buyer_location,
        "lat": b_lat, "lng": b_lng, "coordinates": [b_lng, b_lat],
    }

    # --- Port pins with agent info ---
    port_agents_list = agentic.get("port_agents", []) if isinstance(agentic, dict) else []
    travel_agents_list = agentic.get("travel_agents", []) if isinstance(agentic, dict) else []
    port_conditions_list = agentic.get("port_conditions", []) if isinstance(agentic, dict) else []

    port_agent_map: Dict[str, List[Dict[str, Any]]] = {}
    for pa in port_agents_list:
        pname = pa.get("port", "")
        if pname:
            port_agent_map.setdefault(pname, []).append({
                "agent_id": pa.get("agent_id", ""), "role": "Port Agent",
                "supplier_id": pa.get("supplier_id", ""), "status": pa.get("status", ""),
                "negotiation_summary": pa.get("negotiation_summary", ""),
            })
    for ta in travel_agents_list:
        pname = ta.get("meeting_port", "")
        if pname:
            port_agent_map.setdefault(pname, []).append({
                "agent_id": ta.get("agent_id", ""), "role": "Travel Agent",
                "supplier_id": ta.get("supplier_id", ""), "status": ta.get("status", ""),
                "eta_days": ta.get("eta_days"),
            })

    port_cond_map: Dict[str, Dict[str, Any]] = {}
    for pc in port_conditions_list:
        pname = pc.get("port", "")
        if pname:
            port_cond_map[pname] = {"issues": pc.get("issues", []), "risk_level": pc.get("risk_level", "unknown")}

    port_pins: List[Dict[str, Any]] = []
    seen_ports: set = set()
    for country in countries:
        country_name = country.get("name", "")
        for port in country.get("ports", []):
            pname = port.get("name", port) if isinstance(port, dict) else port
            if pname in seen_ports:
                continue
            seen_ports.add(pname)
            p_lat = float(port.get("lat", 0) or 0) if isinstance(port, dict) else 0.0
            p_lng = float(port.get("lng", 0) or 0) if isinstance(port, dict) else 0.0
            port_pins.append({
                "id": f"port-{pname}", "type": "port", "name": pname, "country": country_name,
                "lat": p_lat, "lng": p_lng, "coordinates": [p_lng, p_lat],
                "agents": port_agent_map.get(pname, []), "conditions": port_cond_map.get(pname, {}),
            })

    # --- Supplier pins with trust + terms ---
    trust_map = {(t.get("supplier_id"), t.get("material")): t for t in trust_logic}
    terms_map: Dict[str, Dict[str, Any]] = {}
    for t in negotiation.get("terms", []):
        terms_map[t.get("supplier_id", "")] = t

    supplier_pins: List[Dict[str, Any]] = []
    seen_suppliers: set = set()
    for vs in validated_suppliers:
        s = vs.get("supplier", {})
        sid = s.get("id", "")
        if sid in seen_suppliers:
            continue
        seen_suppliers.add(sid)
        trust_entry = trust_map.get((sid, vs.get("material", "")), {})
        term = terms_map.get(sid, {})
        supplier_pins.append({
            "id": sid, "type": "supplier", "name": s.get("name", sid),
            "country": s.get("country", ""), "material": vs.get("material", ""),
            "lat": 0.0, "lng": 0.0, "coordinates": [0.0, 0.0],
            "trust_score": trust_entry.get("trust_score", vs.get("trust_score", 50)),
            "trust_rationale": trust_entry.get("rationale", []),
            "compliance_flags": trust_entry.get("flags", []),
            "base_price": s.get("base_price", 0), "currency": s.get("currency", "USD"),
            "lead_time_days": s.get("lead_time_days", 0),
            "certifications": s.get("certifications", []),
            "negotiated_price": term.get("unit_price_est"),
            "negotiated_qty": term.get("qty"),
            "negotiated_subtotal": term.get("subtotal"),
        })

    # Fill lat/lng from suppliers_raw
    suppliers_raw = sim.get("suppliers_raw", [])
    raw_coords = {s.get("supplier_id", s.get("id", "")): (float(s.get("lat", 0) or 0), float(s.get("lng", 0) or 0)) for s in suppliers_raw}
    for sp in supplier_pins:
        coords = raw_coords.get(sp["id"], (0.0, 0.0))
        sp["lat"] = coords[0]
        sp["lng"] = coords[1]
        sp["coordinates"] = [coords[1], coords[0]]

    # --- Route lines ---
    route_lines: List[Dict[str, Any]] = []
    for r in routes:
        ri = r.get("route")
        if not ri:
            continue
        sid = r.get("supplier_id", "")
        from_coords = raw_coords.get(sid, (0, 0))
        waypoints = []
        for port_name in ri.get("ports", []):
            for pp in port_pins:
                if pp["name"] == port_name:
                    waypoints.append({"name": port_name, "lat": pp["lat"], "lng": pp["lng"], "coordinates": [pp["lng"], pp["lat"]]})
                    break

        route_agents: Dict[str, Any] = {}
        for ta in travel_agents_list:
            if ta.get("supplier_id") == sid:
                route_agents["travel_agent"] = {"agent_id": ta.get("agent_id", ""), "status": ta.get("status", ""), "eta_days": ta.get("eta_days"), "meeting_port": ta.get("meeting_port", "")}
                break
        rpa = [{"agent_id": pa.get("agent_id", ""), "port": pa.get("port", ""), "status": pa.get("status", ""), "negotiation_summary": pa.get("negotiation_summary", "")} for pa in port_agents_list if pa.get("supplier_id") == sid]
        if rpa:
            route_agents["port_agents"] = rpa

        risk_val = ri.get("risk_score", 0.2)
        polyline: List[List[float]] = [[from_coords[1], from_coords[0]]]
        for wp in waypoints:
            polyline.append(wp["coordinates"])
        polyline.append([buyer_pin["lng"], buyer_pin["lat"]])

        route_lines.append({
            "id": ri.get("id", f"route-{sid}-{r.get('material', '')}"),
            "material": r.get("material", ""), "supplier_id": sid,
            "supplier_name": next((sp["name"] for sp in supplier_pins if sp["id"] == sid), sid),
            "from_coords": {"lat": from_coords[0], "lng": from_coords[1], "coordinates": [from_coords[1], from_coords[0]]},
            "to_coords": {"lat": buyer_pin["lat"], "lng": buyer_pin["lng"], "coordinates": [buyer_pin["lng"], buyer_pin["lat"]]},
            "waypoints": waypoints, "polyline": polyline,
            "transit_days": ri.get("transit_days", 0), "risk_score": risk_val,
            "risk_level": "low" if risk_val < 0.15 else ("medium" if risk_val < 0.3 else "high"),
            "status": r.get("status", "planned"), "agents": route_agents,
        })

    return {"buyer_pin": buyer_pin, "port_pins": port_pins, "supplier_pins": supplier_pins, "route_lines": route_lines}


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    _load_env_file()
    llm = LLMClient()
    provider = llm.provider
    model = llm.openai_model if provider == "openai" else llm.openrouter_model
    logger.info("[Startup] provider=%s  model=%s  enabled=%s", provider, model, llm.enabled)
    if not llm.enabled:
        logger.warning("[Startup] LLM NOT ENABLED! Set %s API key in .env",
                        "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY")
    world = await load_world(llm)
    logger.info("[World] Loaded: %d countries, %d suppliers, %d routes.",
                 len(world.get("countries", [])), len(world.get("suppliers", [])), len(world.get("routes", [])))
    registry = AgentRegistry()
    seed_registry(world, registry)

    application.state.world = world
    application.state.registry = registry
    application.state.llm = llm
    application.state.simulations = {}
    application.state.procurement_agent = ProcurementAgent(registry, world, llm)
    application.state.supplier_agent = SupplierAgent(registry, world)
    application.state.logistics_agent = LogisticsAgent(registry, world, llm)
    application.state.negotiation_agent = NegotiationAgent(world, llm)
    application.state.execution_planner = ExecutionPlanner(world)

    yield
    application.state.simulations.clear()
    logger.info("[Shutdown] Cleanup complete.")


app = FastAPI(title="Supply Chain AI Agents", lifespan=lifespan)

# CORS: with allow_credentials=True, browser forbids "*"; use explicit origins.
_DEFAULT_ORIGINS = [
    "https://its-camilo.github.io",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
_raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
if _raw_origins and _raw_origins != "*":
    _allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
else:
    _allowed_origins = _DEFAULT_ORIGINS
if not _allowed_origins:
    _allowed_origins = _DEFAULT_ORIGINS
logger.info("[CORS] allow_origins=%s", _allowed_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def seed_registry(world: Dict[str, Any], registry: AgentRegistry) -> None:
    for supplier in world.get("suppliers", []):
        registry.add(AgentFact(
            identity=supplier["name"], role="Supplier",
            capabilities=supplier.get("capabilities", supplier.get("materials", [])),
            endpoint=supplier.get("endpoint", ""), policies=supplier.get("policies", {}),
            jurisdiction={"jurisdiction": supplier.get("jurisdiction"), "country": supplier.get("country")},
            metadata={"materials": supplier.get("materials", []), "tier": supplier.get("tier"), "country": supplier.get("country")},
        ))
    for manufacturer in world.get("manufacturers", []):
        registry.add(AgentFact(
            identity=manufacturer["name"], role="Manufacturer",
            capabilities=manufacturer.get("capabilities", []),
            endpoint=manufacturer.get("endpoint", ""), policies=manufacturer.get("policies", {}),
            jurisdiction={"jurisdiction": manufacturer.get("jurisdiction"), "country": manufacturer.get("country")},
            metadata={"products": manufacturer.get("products", []), "country": manufacturer.get("country")},
        ))
    for prov in world.get("logistics_providers", []):
        registry.add(AgentFact(
            identity=prov["name"], role="Logistics",
            capabilities=prov.get("modes", []),
            endpoint=prov.get("endpoint", ""), policies=prov.get("policies", {}),
            jurisdiction={"jurisdiction": prov.get("jurisdiction"), "country": prov.get("country")},
            metadata={"modes": prov.get("modes", [])},
        ))
    for retailer in world.get("retailers", []):
        registry.add(AgentFact(
            identity=retailer["name"], role="Retailer",
            capabilities=retailer.get("capabilities", []),
            endpoint=retailer.get("endpoint", ""), policies=retailer.get("policies", {}),
            jurisdiction={"jurisdiction": retailer.get("jurisdiction"), "country": retailer.get("country")},
            metadata={"markets": retailer.get("markets", [])},
        ))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.api_route("/", methods=["HEAD", "GET"])
async def root() -> Response:
    """Root path for uptime monitors (HEAD/GET). Returns 200 so checks pass."""
    return Response(
        content="OK",
        status_code=200,
        media_type="text/plain",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    llm: LLMClient = app.state.llm
    world = app.state.world
    return {
        "status": "ok", "time": time.time(),
        "llm_enabled": llm.enabled,
        "llm_provider": llm.provider if llm.enabled else None,
        "world_loaded": len(world.get("countries", [])) > 0,
        "countries": len(world.get("countries", [])),
        "suppliers": len(world.get("suppliers", [])),
        "routes": len(world.get("routes", [])),
        "active_simulations": len(app.state.simulations),
    }


@app.api_route("/head", methods=["HEAD", "GET"])
async def head() -> Response:
    # Minimal body for GET (avoids white screen); HEAD gets same status/headers, no body
    return Response(
        content="OK",
        status_code=200,
        media_type="text/plain",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/schemas")
async def schemas() -> Dict[str, Any]:
    return MCP_SCHEMAS


class WorldGenerateRequest(BaseModel):
    prompt: Optional[str] = None


@app.post("/world/generate")
async def world_generate(req: WorldGenerateRequest) -> Dict[str, Any]:
    llm: LLMClient = app.state.llm
    if not llm.enabled:
        raise HTTPException(status_code=503, detail="World generation requires LLM")
    world = await generate_world_with_ai(llm, req.prompt)
    return {"world": world}


@app.get("/registry")
async def registry_all() -> Dict[str, Any]:
    registry: AgentRegistry = app.state.registry
    return {"agents": [agent.model_dump() for agent in registry.all()]}


@app.post("/registry/discover")
async def registry_discover(query: DiscoveryQuery) -> Dict[str, Any]:
    registry: AgentRegistry = app.state.registry
    results = registry.discover(query)
    return {"results": [{"agent": item["agent"].model_dump(), "score": item["score"], "rationale": item["rationale"]} for item in results]}


@app.get("/events")
async def events() -> StreamingResponse:
    queue = await event_bus.subscribe()

    async def event_generator() -> Any:
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: {event['type']}\n"
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            event_bus.unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _emit_phase(trace_id: str, phase: str, message: str) -> None:
    await event_bus.publish({"type": "phase", "trace_id": trace_id, "phase": phase, "message": message})


# ---------------------------------------------------------------------------
# POST /process-intent  —  World from JSON, Routes + Negotiation + Summary via AI
# ---------------------------------------------------------------------------
@app.post("/process-intent", response_model=IntentResponse)
async def process_intent(req: IntentRequest) -> IntentResponse:
    trace_id = str(uuid.uuid4())
    disruptions = [d.model_dump() for d in req.disruptions]
    _cleanup_old_simulations(app.state.simulations)

    world: Dict[str, Any] = app.state.world
    llm: LLMClient = app.state.llm
    if not llm.enabled:
        raise HTTPException(status_code=503, detail="Routes, negotiation and summary require an enabled LLM")
    registry: AgentRegistry = app.state.registry
    if req.simulate_disruptions:
        disruptions.extend(world.get("risk_events", []))

    procurement_agent: ProcurementAgent = app.state.procurement_agent
    supplier_agent: SupplierAgent = app.state.supplier_agent
    logistics_agent: LogisticsAgent = app.state.logistics_agent
    negotiation_agent: NegotiationAgent = app.state.negotiation_agent
    execution_planner: ExecutionPlanner = app.state.execution_planner

    # --- Phase 1: AI parses demand ---
    await _emit_phase(trace_id, "generating_world", "Analyzing demand signal...")
    demand = await procurement_agent.demand_signal(req.intent)
    demand_msg = build_mcp_message("demand_signal", procurement_agent.identity, supplier_agent.identity, {"intent": demand.intent, "product": demand.product.get("name"), "materials": demand.materials}, trace={"trace_id": trace_id})
    progress_messages: List[Dict[str, Any]] = [
        {"agent": procurement_agent.identity, "text": f"Demand parsed: {demand.product.get('name', 'product')} — {len(demand.materials)} materials."}
    ]
    await event_bus.publish({"type": "graph_update", "trace_id": trace_id, "payload": {"messages": [demand_msg.model_dump()], "progress_messages": list(progress_messages)}})

    # --- Phase 2: Local supplier validation ---
    await _emit_phase(trace_id, "discovering_suppliers", "Validating suppliers and trust scores...")
    validated_suppliers, trust_logic = supplier_agent.validate_suppliers(demand, req.jurisdiction)
    supplier_msg = build_mcp_message("supplier_validation", supplier_agent.identity, logistics_agent.identity, {"validated_suppliers": validated_suppliers, "trust_logic": trust_logic}, trace={"trace_id": trace_id})

    manufacturer_id = demand.product.get("preferred_manufacturer_id")
    manufacturer = None
    for candidate in world.get("manufacturers", []):
        if candidate.get("id") == manufacturer_id:
            manufacturer = candidate
            break
    if not manufacturer and world.get("manufacturers"):
        manufacturer = world["manufacturers"][0]
    if not manufacturer:
        manufacturer = {"id": "mfg-default", "country": "US", "name": "Default Manufacturer"}

    # --- Phase 3: AI selects routes ---
    await _emit_phase(trace_id, "planning_routes", "AI agents analyzing routes and dispatching to ports...")
    routes = await logistics_agent.route_plan_ai(validated_suppliers, manufacturer.get("country"), disruptions, intent=req.intent)
    route_msg = build_mcp_message("logistics_routing", logistics_agent.identity, negotiation_agent.identity, {"routes": routes, "disruptions": disruptions}, trace={"trace_id": trace_id})

    # --- Phase 4: AI negotiates ---
    await _emit_phase(trace_id, "negotiating", "AI agents negotiating autonomously with suppliers...")
    negotiation = await negotiation_agent.negotiate(demand, validated_suppliers, req.constraints)
    negotiation_msg = build_mcp_message("negotiation", negotiation_agent.identity, execution_planner.identity, negotiation, trace={"trace_id": trace_id})

    # --- Phase 5: Local execution plan ---
    execution_plan = execution_planner.create_execution_plan(demand, routes, negotiation)
    execution_msg = build_mcp_message("execution_plan", execution_planner.identity, "network", execution_plan, trace={"trace_id": trace_id})

    progress_messages.extend([
        {"agent": supplier_agent.identity, "text": f"{len(validated_suppliers)} suppliers validated."},
        {"agent": logistics_agent.identity, "text": f"AI selected routes to {manufacturer.get('country', 'destination')} — {len([r for r in routes if r.get('route')])} active legs."},
        {"agent": negotiation_agent.identity, "text": f"Autonomous negotiation complete. Total: ${negotiation.get('total_cost_estimate', 0):,.2f}."},
        {"agent": execution_planner.identity, "text": f"Execution plan: {execution_plan.get('timeline_days')} days, risk {execution_plan.get('risk_score')}."},
    ])

    discovery_paths: List[Dict[str, Any]] = []
    for item in demand.materials:
        query = DiscoveryQuery(intent=req.intent, role="Supplier", materials=[item["material"]], max_results=3)
        results = registry.discover(query)
        discovery_paths.append({"material": item["material"], "candidates": [{"agent_id": r["agent"].id, "identity": r["agent"].identity, "score": r["score"], "rationale": r["rationale"]} for r in results]})

    buyer_location = (req.buyer_location or "").strip() or "United States"
    buyer_info = lookup_buyer_coordinates(world, buyer_location)
    buyer_coordinates = {"lat": buyer_info["lat"], "lng": buyer_info["lng"]}
    world_context = {
        "buyer_coordinates": buyer_coordinates,
        "countries": world.get("countries", []),
        "laws": world.get("laws", []),
        "trade_agreements": world.get("trade_agreements", []),
    }
    suppliers_raw = world.get("suppliers", [])

    agentic_process = _build_agentic_process_from_routes(buyer_location, validated_suppliers, routes)
    graph = build_dependency_graph(demand, validated_suppliers, routes, agentic_process)

    all_msgs = [demand_msg.model_dump(), supplier_msg.model_dump(), route_msg.model_dump(), negotiation_msg.model_dump(), execution_msg.model_dump()]
    report = {
        "trace_id": trace_id, "intent": req.intent, "timestamp": time.time(),
        "buyer_location": buyer_location, "manufacturer": manufacturer,
        "disruptions": disruptions, "discovery_paths": discovery_paths,
        "trust_logic": trust_logic, "agentic_process": agentic_process,
        "world_context": world_context, "messages": all_msgs,
        "negotiation": negotiation, "execution_plan": execution_plan, "routes": routes,
    }

    sim_state = {
        "trace_id": trace_id, "intent": req.intent, "buyer_location": buyer_location,
        "disruptions": disruptions, "demand": demand, "manufacturer": manufacturer,
        "validated_suppliers": validated_suppliers, "trust_logic": trust_logic,
        "routes": routes, "negotiation": negotiation, "execution_plan": execution_plan,
        "agentic_process": agentic_process, "discovery_paths": discovery_paths,
        "world_context": world_context, "suppliers_raw": suppliers_raw,
        "updated_at": time.time(),
    }
    app.state.simulations[trace_id] = sim_state

    report["map_data"] = _build_map_data(report, sim_state)

    await event_bus.publish({"type": "graph_update", "trace_id": trace_id, "payload": {"messages": all_msgs, "graph": graph, "progress_messages": progress_messages}})

    # --- Phase 6: AI generates summary ---
    await _emit_phase(trace_id, "complete", "All agents finished. AI generating final summary...")
    summary = await generate_summary_ai(llm, report)
    return IntentResponse(trace_id=trace_id, report=report, summary=summary)


# ---------------------------------------------------------------------------
# POST /process-intent/{trace_id}/instructions  —  apply follow-up instructions
# ---------------------------------------------------------------------------
@app.post("/process-intent/{trace_id}/instructions")
async def apply_instructions(trace_id: str, req: InstructionRequest) -> Dict[str, Any]:
    llm: LLMClient = app.state.llm
    if not llm.enabled:
        raise HTTPException(status_code=503, detail="Instructions require LLM")
    sim = app.state.simulations.get(trace_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Trace not found or expired")

    update = await apply_instructions_ai(llm=llm, instructions=req.text, simulation=sim, target_port=req.target_port, material=req.material)

    sim["routes"] = update.get("routes", sim["routes"])
    sim["agentic_process"] = update.get("agentic_process", sim["agentic_process"])
    sim["execution_plan"] = update.get("execution_plan", sim["execution_plan"])
    sim["updated_at"] = time.time()

    progress_messages = update.get("progress_notes", []) or [{"agent": "execution-planner", "text": "Instructions applied."}]
    graph = build_dependency_graph(sim["demand"], sim["validated_suppliers"], sim["routes"], sim["agentic_process"])
    await event_bus.publish({"type": "instruction_update", "trace_id": trace_id, "payload": {"messages": [], "graph": graph, "progress_messages": progress_messages, "agentic_process": sim["agentic_process"], "routes": sim["routes"], "execution_plan": sim["execution_plan"]}})

    report = {
        "trace_id": trace_id, "intent": sim["intent"], "timestamp": time.time(),
        "buyer_location": sim.get("buyer_location"), "manufacturer": sim["manufacturer"],
        "disruptions": sim["disruptions"], "discovery_paths": sim["discovery_paths"],
        "trust_logic": sim["trust_logic"], "agentic_process": sim["agentic_process"],
        "world_context": sim.get("world_context", {}),
        "negotiation": sim["negotiation"], "execution_plan": sim["execution_plan"], "routes": sim["routes"],
    }
    report["map_data"] = _build_map_data(report, sim)
    summary = await generate_summary_ai(llm, report)
    return {"trace_id": trace_id, "report": report, "summary": summary}

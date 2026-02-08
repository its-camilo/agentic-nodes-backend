"""
World loader for supply chain simulation.
Loads a comprehensive pre-built world from world_data.json.
Falls back to AI generation only if the file is missing.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Path to the static world file (next to this script)
_WORLD_FILE = Path(__file__).resolve().parent / "world_data.json"

WORLD_GENERATION_HINT = (
    "Return JSON with keys: countries, materials, products, suppliers, manufacturers, "
    "logistics_providers, retailers, routes, laws (and optionally price_indices, risk_events)."
)

DEFAULT_WORLD_PROMPT = (
    "Generate a realistic supply chain world (JSON). "
    "4-5 countries with real ports; 4-6 materials; "
    "1 product with BOM; 4-6 suppliers; "
    "1 manufacturer; 1 logistics provider; 1 retailer; "
    "6-10 routes; 2-3 laws. Consistent IDs. JSON only."
)


def get_minimal_world() -> Dict[str, Any]:
    """Empty world used when SIMULATION_AI_ONLY=true; no world data is loaded at startup."""
    return {
        "countries": [], "materials": [], "products": [], "suppliers": [],
        "manufacturers": [], "logistics_providers": [], "retailers": [],
        "routes": [], "laws": [], "price_indices": {}, "risk_events": [],
        "multi_tier_relationships": [], "lead_time_variability": {},
        "buyer_locations": {}, "trade_agreements": [],
    }


def _defaults_for_world() -> Dict[str, Any]:
    return {"multi_tier_relationships": [], "lead_time_variability": {}, "price_indices": {}, "risk_events": [], "buyer_locations": {}, "trade_agreements": []}


def _ensure_price_indices_and_variability(world: Dict[str, Any]) -> None:
    materials = [m for m in world.get("materials", []) if "id" in m]
    if not materials:
        return
    if not world.get("price_indices"):
        world["price_indices"] = {m["id"]: [{"date": "2026-01-01", "index": m.get("base_price_index", 100)}] for m in materials}
    if not world.get("lead_time_variability"):
        world["lead_time_variability"] = {
            m["id"]: {"baseline_days": m.get("lead_time_days", 30), "variance_days": min(10, max(2, int(m.get("lead_time_days", 30) * 0.2)))}
            for m in materials
        }


def _load_world_from_file() -> Optional[Dict[str, Any]]:
    """Load world from world_data.json if it exists."""
    if not _WORLD_FILE.is_file():
        return None
    try:
        data = json.loads(_WORLD_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("countries"):
            logger.info("[World] Loaded from %s: %d countries, %d suppliers, %d routes, %d products",
                        _WORLD_FILE.name, len(data.get("countries", [])),
                        len(data.get("suppliers", [])), len(data.get("routes", [])),
                        len(data.get("products", [])))
            return data
        logger.warning("[World] %s exists but seems empty/invalid", _WORLD_FILE.name)
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("[World] Failed to load %s: %s", _WORLD_FILE.name, exc)
        return None


async def generate_world_with_ai(llm: Any, prompt: Optional[str] = None) -> Dict[str, Any]:
    """Generate a complete supply chain world purely via AI inference (fallback)."""
    from agents import LLMClient
    if not isinstance(llm, LLMClient) or not llm.enabled:
        raise ValueError("AI world generation requires an enabled LLM")
    user_prompt = (prompt or os.getenv("WORLD_PROMPT") or "").strip() or DEFAULT_WORLD_PROMPT
    system = "Supply chain expert. Generate a simulation world as JSON. Consistent IDs. Realistic data. JSON only."
    data = await llm.reason_json(system, f"{user_prompt}\n{WORLD_GENERATION_HINT}", max_tokens=4500)
    if not data or not isinstance(data, dict):
        raise ValueError("LLM did not return a valid world JSON")
    defaults = _defaults_for_world()
    for key, value in defaults.items():
        data.setdefault(key, value)
    _ensure_price_indices_and_variability(data)
    return data


async def load_world(llm: Optional[Any] = None) -> Dict[str, Any]:
    """Load the simulation world. Priority: world_data.json > AI generation > minimal."""
    # 1. Try static file first (fast, no LLM needed)
    file_world = _load_world_from_file()
    if file_world:
        defaults = _defaults_for_world()
        for key, value in defaults.items():
            file_world.setdefault(key, value)
        _ensure_price_indices_and_variability(file_world)
        return file_world

    # 2. Fall back to AI generation
    from agents import LLMClient
    if llm is not None and isinstance(llm, LLMClient) and llm.enabled:
        logger.info("[World] No world_data.json found, generating via AI...")
        return await generate_world_with_ai(llm)

    # 3. No file, no LLM → minimal world
    logger.warning("[World] No world_data.json and no LLM available, using minimal world")
    return get_minimal_world()


def lookup_buyer_coordinates(world: Dict[str, Any], buyer_location: str) -> Dict[str, Any]:
    """Look up buyer coordinates and country from the world's buyer_locations map."""
    locations = world.get("buyer_locations", {})
    key = buyer_location.lower().strip()
    # Exact match
    if key in locations:
        loc = locations[key]
        return {"lat": loc["lat"], "lng": loc["lng"], "country": loc.get("country", "US")}
    # Partial match
    for k, v in locations.items():
        if k in key or key in k:
            return {"lat": v["lat"], "lng": v["lng"], "country": v.get("country", "US")}
    # Default to Washington DC
    return {"lat": 38.9072, "lng": -77.0369, "country": "US"}

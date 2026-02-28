"""
API v1 Endpoints - FastAPI route handlers
Orchestrates all services for flight risk analysis and itinerary generation
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, Optional
import logging
import json
import re
from datetime import datetime, timedelta, timezone

from app.models.request import DelayScenario, FlightRiskRequest, ItineraryRequest
from app.models.flight import FlightLeg, ConnectionRisk
from app.services.lufthansa import get_lufthansa_client, LufthansaClient, LufthansaAPIError
from app.services.analyzer import FlightConnectionAnalyzer
from app.services.vertex_ai import get_vertex_ai_service, VertexAIService
from app.core.config import get_settings, Settings
from app.prompts.manager import get_prompt_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["Travel Assistant"])


def _extract_response_text(response: Any) -> str:
    """Extract text robustly from Gemini response across client modes."""
    response_text = getattr(response, "text", None)
    if response_text:
        return response_text

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""

    parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
    texts = [part.text for part in parts if hasattr(part, "text") and part.text]
    return "".join(texts)


def _repair_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """Clean common Gemini formatting artifacts and parse strict JSON."""
    if not raw_text:
        raise ValueError("Empty model response")

    text = raw_text.replace("\ufeff", "").strip()

    # Remove citations like [1], [2], [1, 2]
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)

    # Prefer fenced JSON block if present (closed fence)
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Handle incomplete fence outputs such as "```json\n{...<truncated>"
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

    # Extract from first JSON object if possible
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)
    else:
        first_brace_idx = text.find("{")
        if first_brace_idx != -1:
            text = text[first_brace_idx:]

    text = text.strip()

    parse_variants = [
        text,
        re.sub(r",\s*([\]}])", r"\1", text),  # remove trailing commas before ] or }
    ]
    parsed = None
    parse_error: Optional[Exception] = None
    for candidate in parse_variants:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as e:
            parse_error = e

    if parsed is None:
        raise parse_error or ValueError("Unable to parse JSON")

    if not isinstance(parsed, dict):
        raise ValueError("Response is not a JSON object")
    return parsed


def _parse_utc(iso_utc: str) -> datetime:
    return datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_city_stops(city: str) -> list[Dict[str, Any]]:
    city_lower = city.lower()
    if city_lower == "madrid":
        return [
            {
                "activity_name": "Travel to City Center (Puerta del Sol)",
                "venue_name": "Madrid-Barajas Airport to Puerta del Sol",
                "category": "transport",
                "estimated_cost": "€5-10",
                "place_id": None,
                "latitude": 40.4169,
                "longitude": -3.7035,
                "google_maps_url": "https://www.google.com/maps/search/?api=1&query=Puerta+del+Sol+Madrid",
                "description": "Airport transfer to the historic city center.",
                "transport_to_next": "Walk",
            },
            {
                "activity_name": "Local Tapas Break",
                "venue_name": "Mercado de San Miguel",
                "category": "food",
                "estimated_cost": "€20-30",
                "place_id": None,
                "latitude": 40.4154,
                "longitude": -3.7083,
                "google_maps_url": "https://www.google.com/maps/search/?api=1&query=Mercado+de+San+Miguel+Madrid",
                "description": "Popular food hall with quick local options and central location.",
                "transport_to_next": "Walk",
            },
            {
                "activity_name": "Park & Scenic Walk",
                "venue_name": "Parque del Retiro",
                "category": "activity",
                "estimated_cost": "Free",
                "place_id": None,
                "latitude": 40.4153,
                "longitude": -3.6844,
                "google_maps_url": "https://www.google.com/maps/search/?api=1&query=Parque+del+Retiro+Madrid",
                "description": "Open-air walk suited for a short layover reset.",
                "transport_to_next": "Taxi",
            },
            {
                "activity_name": "Culture Stop",
                "venue_name": "Museo Nacional del Prado",
                "category": "culture",
                "estimated_cost": "€15",
                "place_id": None,
                "latitude": 40.4138,
                "longitude": -3.6921,
                "google_maps_url": "https://www.google.com/maps/search/?api=1&query=Museo+del+Prado+Madrid",
                "description": "Classic Madrid museum stop close to central districts.",
                "transport_to_next": "Taxi",
            },
            {
                "activity_name": "Return to Airport",
                "venue_name": "Puerta del Sol to Madrid-Barajas Airport",
                "category": "transport",
                "estimated_cost": "€30-45",
                "place_id": None,
                "latitude": 40.4722,
                "longitude": -3.5608,
                "google_maps_url": "https://www.google.com/maps/search/?api=1&query=Adolfo+Suarez+Madrid-Barajas+Airport",
                "description": "Return transfer with airport security buffer.",
                "transport_to_next": "N/A",
            },
        ]

    # Fallback for non-Madrid cities
    return [
        {
            "activity_name": "Travel to City Center",
            "venue_name": f"{city} City Center",
            "category": "transport",
            "estimated_cost": "€10-20",
            "place_id": None,
            "latitude": 0.0,
            "longitude": 0.0,
            "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={city.replace(' ', '+')}+city+center",
            "description": "Transfer from airport to city center.",
            "transport_to_next": "Walk",
        },
        {
            "activity_name": "Quick Local Meal",
            "venue_name": f"{city} Local Market",
            "category": "food",
            "estimated_cost": "€15-30",
            "place_id": None,
            "latitude": 0.0,
            "longitude": 0.0,
            "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={city.replace(' ', '+')}+local+market",
            "description": "Short food stop before sightseeing.",
            "transport_to_next": "Walk",
        },
        {
            "activity_name": "City Landmark Visit",
            "venue_name": f"{city} Main Landmark",
            "category": "culture",
            "estimated_cost": "€10-20",
            "place_id": None,
            "latitude": 0.0,
            "longitude": 0.0,
            "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={city.replace(' ', '+')}+landmark",
            "description": "Compact cultural stop in the city.",
            "transport_to_next": "Taxi",
        },
        {
            "activity_name": "Return to Airport",
            "venue_name": f"{city} City Center to Airport",
            "category": "transport",
            "estimated_cost": "€20-40",
            "place_id": None,
            "latitude": 0.0,
            "longitude": 0.0,
            "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={city.replace(' ', '+')}+airport",
            "description": "Return transfer with security buffer.",
            "transport_to_next": "N/A",
        },
    ]


def _build_time_series_fallback(city: str, arrival_utc: str, departure_utc: str) -> Dict[str, Any]:
    """Create deterministic 4-5 step time-series itinerary when model output is unusable."""
    arrival_dt = _parse_utc(arrival_utc)
    departure_dt = _parse_utc(departure_utc)
    safe_window_minutes = int((departure_dt - arrival_dt).total_seconds() // 60) - 120

    city_start = arrival_dt + timedelta(minutes=30)
    city_end = departure_dt - timedelta(minutes=90)
    if city_end <= city_start:
        return {
            "city": city,
            "stay_at_airport": True,
            "safe_window_minutes": max(0, safe_window_minutes),
            "itinerary": [],
            "_fallback_generated": True,
        }

    stops = _default_city_stops(city)
    if len(stops) == 5:
        weights = [1, 1, 1, 1, 1]
    else:
        weights = [1, 1, 1, 1]

    total_minutes = int((city_end - city_start).total_seconds() // 60)
    weight_sum = sum(weights)
    boundaries = [city_start]
    cumulative = 0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if idx == len(weights) - 1:
            boundaries.append(city_end)
        else:
            step_minutes = round(total_minutes * cumulative / weight_sum)
            boundaries.append(city_start + timedelta(minutes=step_minutes))

    itinerary: list[Dict[str, Any]] = []
    for idx, stop in enumerate(stops):
        item = dict(stop)
        item["start_utc"] = _iso_z(boundaries[idx])
        item["end_utc"] = _iso_z(boundaries[idx + 1])
        itinerary.append(item)

    return {
        "city": city,
        "stay_at_airport": False,
        "safe_window_minutes": max(0, safe_window_minutes),
        "itinerary": itinerary,
        "_fallback_generated": True,
    }


def _normalize_place_ids(itinerary_data: Dict[str, Any], allow_null_place_id: bool) -> Dict[str, Any]:
    """Normalize place_id in API key mode where grounded IDs may not exist."""
    itinerary = itinerary_data.get("itinerary")
    if not isinstance(itinerary, list):
        return itinerary_data

    for item in itinerary:
        if not isinstance(item, dict):
            continue
        place_id = item.get("place_id")
        if place_id in ("", "N/A", "unknown"):
            item["place_id"] = None
        if "place_id" not in item and allow_null_place_id:
            item["place_id"] = None
    return itinerary_data


def _validate_itinerary_shape(
    itinerary_data: Dict[str, Any],
    min_activities: int,
    max_activities: int,
    require_place_id: bool
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    itinerary = itinerary_data.get("itinerary")
    if not isinstance(itinerary, list):
        return False, ["'itinerary' must be a list"]

    if len(itinerary) < min_activities or len(itinerary) > max_activities:
        issues.append(f"itinerary length must be {min_activities}-{max_activities}, got {len(itinerary)}")

    required_fields = [
        "start_utc", "end_utc", "activity_name", "venue_name", "category",
        "estimated_cost", "latitude", "longitude", "google_maps_url",
        "description", "transport_to_next"
    ]
    if require_place_id:
        required_fields.append("place_id")

    for idx, item in enumerate(itinerary):
        if not isinstance(item, dict):
            issues.append(f"activity[{idx}] is not an object")
            continue
        for field in required_fields:
            if field not in item or item.get(field) in ("", None):
                issues.append(f"activity[{idx}] missing/empty '{field}'")

    if itinerary:
        first_cat = itinerary[0].get("category")
        last_cat = itinerary[-1].get("category")
        if first_cat != "transport":
            issues.append("first activity category must be 'transport'")
        if last_cat != "transport":
            issues.append("last activity category must be 'transport'")

    return len(issues) == 0, issues


# Dependency injection
def get_analyzer() -> FlightConnectionAnalyzer:
    """Get analyzer instance with settings"""
    settings = get_settings()
    return FlightConnectionAnalyzer(min_connection_time=settings.min_connection_time)


# ============================================================================
# TEST ENDPOINT - Quick Gemini Connection Test
# ============================================================================

@router.get("/test-gemini", summary="Test Gemini AI connection")
async def test_gemini(
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service)
) -> Dict[str, Any]:
    """
    Quick test endpoint to verify Gemini AI is working
    
    Returns a simple AI-generated response to confirm the connection.
    """
    try:
        if not vertex_ai:
            return {
                "status": "error",
                "message": "Gemini AI not configured. Set GOOGLE_GEMINI_API_KEY in .env file"
            }
        
        logger.info("Testing Gemini AI connection...")
        
        # Simple test prompt
        from google.genai.types import GenerateContentConfig
        
        response = vertex_ai.client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'Hello from Madrid!' and tell me one interesting fact about Madrid in 2 sentences.",
            config=GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=100
            )
        )
        
        return {
            "status": "success",
            "message": "Gemini AI is working! ✅",
            "response": response.text,
            "model": "gemini-2.5-flash"
        }
        
    except Exception as e:
        logger.error(f"Gemini test failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Gemini AI test failed: {str(e)}"
        }


@router.post("/test-maps-grounding", summary="Test Google Maps Grounding with Layover Plan")
async def test_maps_grounding(
    request: Dict[str, Any],
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service)
) -> Dict[str, Any]:
    """
    Test Google Maps grounding with realistic layover scenario
    Based on vertex_ai_v2.py architecture - simplified and production-ready
    
    Request body:
    {
        "city": "Madrid",
        "arrival_utc": "2026-03-01T14:00:00Z",
        "departure_utc": "2026-03-01T20:00:00Z",
        "weather": "sunny, 22°C",
        "preferences": ["family-friendly", "outdoor seating", "stroller access"]
    }
    
    Response includes:
    - AI-generated itinerary with Maps grounding (Vertex AI mode)
    - OR location-aware recommendations (API Key mode)
    - Grounding metadata with venue details
    """
    try:
        if not vertex_ai:
            return {
                "status": "error",
                "message": "Gemini AI not configured. Set GOOGLE_GEMINI_API_KEY in .env file"
            }
        
        # Extract request data
        city = request.get("city", "Madrid")
        arrival_utc = request.get("arrival_utc", "2026-03-01T14:00:00Z")
        departure_utc = request.get("departure_utc", "2026-03-01T20:00:00Z")
        weather = request.get("weather", "sunny, 22°C")
        preferences = request.get("preferences", ["sightseeing", "local food"])
        
        logger.info(f"🗺️ Maps grounding test: {city} | {arrival_utc} → {departure_utc} (client: {vertex_ai.client_type})")
        
        # Get coordinates for the city
        airport_code = "MAD" if city.lower() == "madrid" else "FRA"
        lat, lng = vertex_ai.get_airport_coordinates(airport_code)
        
        from google.genai.types import GenerateContentConfig, Tool, GoogleMaps
        
        maps_mode_instruction = (
            "Google Maps grounding is active. Use real verified venues and real place_id values."
            if vertex_ai.client_type == "vertex_ai"
            else "Google Maps grounding is not available in this mode. Use realistic venues and set place_id to null when unavailable."
        )
        place_id_instruction = (
            "place_id: Google Maps CID (long numeric string) - MUST be from actual Maps search"
            if vertex_ai.client_type == "vertex_ai"
            else "place_id: Use null when a verified Maps CID is not available in API key mode"
        )

        # Build prompt with mode-aware Maps/place_id requirements
        prompt = f"""
Role: You are the Layover Architect - an expert travel planner with access to Google Maps API.

CONTEXT:
- City: {city}
- Weather: {weather}
- Arrival Airport (UTC): {arrival_utc}
- Departure from Airport (UTC): {departure_utc}
- Airport Coordinates: {lat}, {lng}
- User Preferences: {', '.join(preferences)}
- Mode: {vertex_ai.client_type}
- Grounding Note: {maps_mode_instruction}

TASK: Create a precise, time-sequenced layover itinerary with 4-6 activities using REAL Google Maps data.

CRITICAL RULES:
1. Time Buffer Calculations:
   - Add 30 minutes after {arrival_utc} for deplaning/passport/luggage
   - Reserve 90 minutes before {departure_utc} for return to airport
   - Calculate exact "safe_window_minutes" = (departure - arrival - 120 minutes)
   - First activity starts at arrival + 30 min buffer
   - Last activity ends at departure - 90 min buffer

2. Venue Verification and Maps Linking:
   - Use real venues in {city} and accurate coordinates
   - Include a valid google_maps_url for every activity
   - In vertex_ai mode, use grounded place_id values from Maps tool
   - In api_key mode, place_id may be null if unavailable

3. Each Activity MUST Include ALL These Fields:
   - start_utc: ISO 8601 format with Z (e.g., "2026-03-01T12:05:00Z")
   - end_utc: ISO 8601 format with Z (calculated from start + duration)
   - activity_name: Descriptive name (e.g., "Travel to City Center (Puerta del Sol)")
   - venue_name: REAL venue name from Google Maps (e.g., "Parque del Retiro")
   - category: MUST BE ONE OF: "transport", "food", "culture", "activity", "shopping"
   - estimated_cost: String with currency (e.g., "€15", "Free", "€20-35")
   - {place_id_instruction}
   - latitude: Exact decimal latitude from Google Maps (e.g., 40.489515)
   - longitude: Exact decimal longitude from Google Maps (e.g., -3.564276)
   - google_maps_url: Full Maps URL: https://maps.google.com/?cid=<place_id>
   - description: 1-2 sentences explaining why this fits preferences
   - transport_to_next: How to reach next activity ("Walk", "Metro", "Taxi", "Train", "N/A")

4. First Activity MUST Be:
   - category: "transport"
   - activity_name: "Travel to City Center (specific landmark)"
   - From airport to first venue in city

5. Last Activity MUST Be:
   - category: "transport"
   - activity_name: "Return to Airport"
   - Back to airport with 90-minute buffer

6. Output Requirements:
   - Valid JSON only, no markdown backticks or code blocks
   - No citations like [1], [2] or footnotes
   - Strictly DO NOT include any text before or after the JSON object (start with '{{' and end with '}}')
   - Use ONLY real, verified venues that exist in {city}
   - All coordinates must be accurate for {city}
   - English language only

OUTPUT JSON SCHEMA (STRICT - Copy this structure exactly):
{{
  "city": "{city}",
  "stay_at_airport": false,
  "safe_window_minutes": <calculate exact minutes>,
  "itinerary": [
    {{
      "start_utc": "2026-03-01T12:05:00Z",
      "end_utc": "2026-03-01T12:45:00Z",
      "activity_name": "Travel to City Center (Puerta del Sol)",
      "venue_name": "Madrid-Barajas Airport to Sol",
      "category": "transport",
      "estimated_cost": "€10",
      "place_id": "7959872553613643503",
      "latitude": 40.4168,
      "longitude": -3.7038,
      "google_maps_url": "https://maps.google.com/?cid=7959872553613643503",
      "description": "Take Metro Line 8 to Nuevos Ministerios, transfer to Line 10 to reach Puerta del Sol, the heart of Madrid.",
      "transport_to_next": "Walk"
    }},
    {{
      "start_utc": "2026-03-01T12:50:00Z",
      "end_utc": "2026-03-01T14:00:00Z",
      "activity_name": "Lunch at Traditional Tapas Bar",
      "venue_name": "Mercado de San Miguel",
      "category": "food",
      "estimated_cost": "€20-30",
      "place_id": "1234567890",
      "latitude": 40.4154,
      "longitude": -3.7086,
      "google_maps_url": "https://maps.google.com/?cid=1234567890",
      "description": "Historic market hall with gourmet tapas stands. Family-friendly with high chairs available and wide aisles for strollers.",
      "transport_to_next": "Walk"
    }}
  ]
}}

REMINDER: Search Google Maps for each venue and use REAL place_id, latitude, longitude values. Generate 4-6 activities total.
"""
        
        # Config based on client type (vertex_ai_v2.py style - simpler)
        if vertex_ai.client_type == "vertex_ai":
            logger.info("🗺️ Using Vertex AI with Maps grounding")
            config = GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4000,
                tools=[Tool(google_maps=GoogleMaps())]
            )
        else:
            logger.info("ℹ️ Using API Key mode (Maps grounding not available)")
            config = GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4000
            )
        
        # Generate content
        response = vertex_ai.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )
        
        # Robust response text extraction (Vertex AI can return candidate parts)
        full_text = _extract_response_text(response)
        
        # Try to parse as JSON (robust error handling)
        itinerary_data = None
        json_parse_error = None
        
        try:
            itinerary_data = _repair_and_parse_json(full_text)
            logger.info("✅ Successfully parsed JSON itinerary after cleanup")

            if "itinerary" not in itinerary_data:
                logger.warning("⚠️ Missing 'itinerary' field in response")
            
        except (json.JSONDecodeError, ValueError) as e:
            json_parse_error = str(e)
            logger.error(f"❌ JSON parse failed (first pass): {e}")
            logger.debug(f"Raw response text: {full_text[:500]}...")

            # Second pass: ask model to repair its own output into strict JSON
            repair_prompt = f"""
Convert the text below into ONE valid JSON object.
Rules:
- Output JSON only (no markdown, no explanations).
- Keep all original keys and values where possible.
- Fix invalid JSON syntax only (quotes, commas, brackets, escaping).
- Start with '{{' and end with '}}'.

TEXT TO REPAIR:
{full_text}
"""
            try:
                repair_response = vertex_ai.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=repair_prompt,
                    config=GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=4000
                    )
                )
                repaired_text = _extract_response_text(repair_response)
                itinerary_data = _repair_and_parse_json(repaired_text)
                itinerary_data["_json_repaired"] = True
                logger.info("✅ JSON parsed successfully after repair pass")
            except Exception as repair_error:
                logger.error(f"❌ JSON repair pass failed: {repair_error}")
                itinerary_data = {
                    "error": "JSON_PARSE_ERROR",
                    "message": f"Failed to parse Gemini response as JSON: {json_parse_error}",
                    "raw_response": full_text[:1000],  # First 1000 chars for debugging
                    "suggestion": "Model output is not valid JSON in API Key mode. Enable Vertex AI (GCP_PROJECT_ID/GCP_LOCATION) for more stable grounding and output."
                }

        # If model output is too short or invalid, generate deterministic time-series fallback
        should_fallback = (
            not isinstance(itinerary_data, dict)
            or "error" in itinerary_data
            or not isinstance(itinerary_data.get("itinerary"), list)
            or len(itinerary_data.get("itinerary", [])) < 4
        )
        if should_fallback:
            logger.warning("⚠️ Itinerary too short/invalid, generating deterministic fallback itinerary")
            itinerary_data = _build_time_series_fallback(
                city=city,
                arrival_utc=arrival_utc,
                departure_utc=departure_utc
            )
        
        # Extract grounding metadata (Vertex AI mode only)
        grounding_metadata = None
        
        if vertex_ai.client_type == "vertex_ai":
            maps_chunks = []
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    gm = candidate.grounding_metadata
                    
                    if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
                        for chunk in gm.grounding_chunks:
                            if hasattr(chunk, "maps") and chunk.maps:
                                maps_chunks.append({
                                    "title": chunk.maps.title if hasattr(chunk.maps, "title") else None,
                                    "uri": chunk.maps.uri if hasattr(chunk.maps, "uri") else None,
                                    "place_id": getattr(chunk.maps, "placeId", None)
                                })
            
            grounding_metadata = {
                "maps_used": len(maps_chunks) > 0,
                "maps_venues_found": len(maps_chunks),
                "maps_data": maps_chunks,
                "note": "✅ Real venue data from Google Maps" if maps_chunks else "⚠️ No Maps data returned"
            }
        
        return {
            "status": "success",
            "model": "gemini-2.5-flash",
            "client_type": vertex_ai.client_type,
            "city": city,
            "coordinates": {"lat": lat, "lng": lng},
            "time_window": {
                "arrival": arrival_utc,
                "departure": departure_utc
            },
            "data": itinerary_data,
            "grounding_metadata": grounding_metadata,
            "note": "✅ Maps grounding active" if vertex_ai.client_type == "vertex_ai" else "ℹ️ API Key mode - enable GCP_PROJECT_ID in .env for Maps grounding"
        }
        
    except Exception as e:
        logger.error(f"Maps grounding test failed: {str(e)}")
        import traceback
        return {
            "status": "error",
            "message": f"Test failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

@router.post("/test-maps-grounding-retry", summary="Retry-based Gemini itinerary generation (API key + Vertex diagnostics)")
async def test_maps_grounding_retry(
    request: Dict[str, Any],
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Experimental endpoint:
    - Retries Gemini generation up to N attempts (default: 3)
    - Avoids deterministic fallback unless explicitly requested
    - In API key mode, accepts null place_id while preserving time-series itinerary
    - Can force a fresh VertexAIService init for diagnostics
    """
    try:
        city = request.get("city", "Madrid")
        arrival_utc = request.get("arrival_utc", "2026-03-01T11:00:00Z")
        departure_utc = request.get("departure_utc", "2026-03-01T19:00:00Z")
        weather = request.get("weather", "sunny, 22C")
        preferences = request.get("preferences", ["family-friendly", "local food"])
        max_attempts = max(1, min(int(request.get("max_attempts", 3)), 5))
        min_activities = max(4, min(int(request.get("min_activities", 4)), 6))
        max_activities = max(min_activities, min(int(request.get("max_activities", 6)), 8))
        allow_null_place_id = bool(request.get("allow_null_place_id", True))
        use_fallback_if_needed = bool(request.get("use_fallback_if_needed", False))
        force_vertex_init = bool(request.get("force_vertex_init", False))

        if not vertex_ai and not force_vertex_init:
            return {
                "status": "error",
                "message": "Gemini AI not configured. Set GOOGLE_GEMINI_API_KEY in .env"
            }

        active_service = vertex_ai
        vertex_diagnostics: Dict[str, Any] = {}

        if force_vertex_init:
            active_service = VertexAIService(
                api_key=settings.google_gemini_api_key,
                project_id=settings.gcp_project_id,
                location=settings.gcp_location
            )
            invalid_project_ids = {
                "your-project-id",
                "your-gcp-project-id",
                "your-complete-gcp-project-id",
                "project-b7be3ea5-e603-419c-87d",
                ""
            }
            vertex_diagnostics = {
                "force_vertex_init": True,
                "configured_project_id": bool(settings.gcp_project_id),
                "configured_location": settings.gcp_location,
                "project_id_blacklisted_in_service": (settings.gcp_project_id or "") in invalid_project_ids,
                "client_type_after_init": active_service.client_type
            }

        airport_code = "MAD" if city.lower() == "madrid" else "FRA"
        lat, lng = active_service.get_airport_coordinates(airport_code)

        from google.genai.types import GenerateContentConfig, Tool, GoogleMaps

        base_prompt = f"""
You are a travel planner. Generate a layover itinerary in strict JSON only.
Context:
- City: {city}
- Arrival UTC: {arrival_utc}
- Departure UTC: {departure_utc}
- Weather: {weather}
- Preferences: {", ".join(preferences)}
- Airport Coordinates: {lat}, {lng}

Rules:
1. Build {min_activities}-{max_activities} activities in time order.
2. Add 30 min after arrival before first activity.
3. End last activity at departure-90 min.
4. First and last activity category must be "transport".
5. Required fields per activity:
   start_utc, end_utc, activity_name, venue_name, category, estimated_cost,
   place_id, latitude, longitude, google_maps_url, description, transport_to_next
6. If place_id is unavailable, use null (not fake IDs).
7. No markdown, no commentary. Output one JSON object only.

Output schema:
{{
  "city": "{city}",
  "stay_at_airport": false,
  "safe_window_minutes": <int>,
  "itinerary": [{{ ...activity... }}]
}}
"""

        if active_service.client_type == "vertex_ai":
            config = GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4000,
                tools=[Tool(google_maps=GoogleMaps())]
            )
        else:
            config = GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4000
            )

        attempt_logs: list[Dict[str, Any]] = []
        best_data: Optional[Dict[str, Any]] = None
        last_issues: list[str] = []
        attempt_prompt = base_prompt

        for attempt in range(1, max_attempts + 1):
            response = active_service.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=attempt_prompt,
                config=config
            )
            full_text = _extract_response_text(response)

            parse_ok = False
            current_data: Optional[Dict[str, Any]] = None
            parse_error = None
            try:
                current_data = _repair_and_parse_json(full_text)
                parse_ok = True
            except Exception as e:
                parse_error = str(e)

            if parse_ok and current_data:
                current_data = _normalize_place_ids(
                    current_data,
                    allow_null_place_id=(active_service.client_type != "vertex_ai" or allow_null_place_id)
                )
                valid, issues = _validate_itinerary_shape(
                    current_data,
                    min_activities=min_activities,
                    max_activities=max_activities,
                    require_place_id=(active_service.client_type == "vertex_ai" and not allow_null_place_id)
                )
                best_data = current_data
                last_issues = issues
                attempt_logs.append({
                    "attempt": attempt,
                    "parse_ok": True,
                    "valid": valid,
                    "issue_count": len(issues)
                })
                if valid:
                    break
                issues_text = "\n".join(f"- {x}" for x in issues[:8])
                attempt_prompt = (
                    base_prompt
                    + "\nPrevious output issues. Regenerate fully and fix these:\n"
                    + issues_text
                )
            else:
                attempt_logs.append({
                    "attempt": attempt,
                    "parse_ok": False,
                    "valid": False,
                    "parse_error": parse_error
                })
                attempt_prompt = (
                    base_prompt
                    + "\nPrevious output was invalid JSON. Regenerate from scratch as strict JSON."
                )

        final_data = best_data
        used_fallback = False
        if not final_data and use_fallback_if_needed:
            final_data = _build_time_series_fallback(city, arrival_utc, departure_utc)
            used_fallback = True
        elif not final_data:
            final_data = {
                "error": "GENERATION_FAILED",
                "message": "Gemini output could not be parsed into valid JSON after retries.",
                "attempts": attempt_logs
            }

        return {
            "status": "success",
            "endpoint": "test-maps-grounding-retry",
            "model": "gemini-2.5-flash",
            "client_type": active_service.client_type,
            "city": city,
            "coordinates": {"lat": lat, "lng": lng},
            "time_window": {"arrival": arrival_utc, "departure": departure_utc},
            "attempts": attempt_logs,
            "last_issues": last_issues,
            "used_fallback": used_fallback,
            "data": final_data,
            "vertex_diagnostics": vertex_diagnostics,
            "note": (
                "✅ Vertex AI mode: grounded place_id expected"
                if active_service.client_type == "vertex_ai"
                else "ℹ️ API key mode: place_id may be null, Maps grounding metadata unavailable"
            )
        }
    except Exception as e:
        logger.error(f"Retry endpoint failed: {str(e)}")
        import traceback
        return {
            "status": "error",
            "message": f"Retry endpoint failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


# ============================================================================
# CORE INTEGRATION ENDPOINTS (Flight Status + Generate Plan)
# ============================================================================

@router.get("/flight-status", summary="Get real-time flight status (Core Integration)")
async def get_flight_status_core(
    flight_number: str,
    date: str,
    lufthansa: LufthansaClient = Depends(get_lufthansa_client)
) -> Dict[str, Any]:
    """
    Core Integration Endpoint: Query Lufthansa API for flight status
    
    Returns parsed flight data using Pydantic models.
    This endpoint is designed for integration with /generate-plan.
    
    Args:
        flight_number: Flight number (e.g., IB774, LH400)
        date: Date in YYYY-MM-DD format
    
    Returns:
        ParsedFlightResponse with structured flight data
    """
    try:
        logger.info(f"[Core] Querying flight status: {flight_number} on {date}")
        
        # Call Lufthansa service
        response = lufthansa.get_flight_status(flight_number, date)
        
        if not response.flights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flight {flight_number} not found for date {date}"
            )
        
        # Return ParsedFlightResponse as JSON
        return {
            "flights": [flight.model_dump() for flight in response.flights],
            "total_count": response.total_count,
            "query_type": response.query_type,
            "timestamp": response.timestamp
        }
        
    except LufthansaAPIError as e:
        logger.error(f"Lufthansa API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Lufthansa API unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error fetching flight status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/generate-plan", summary="Generate AI-powered recovery plan with Google Search Grounding")
async def generate_plan_core(
    request: Dict[str, Any],
    lufthansa: LufthansaClient = Depends(get_lufthansa_client),
    analyzer: FlightConnectionAnalyzer = Depends(get_analyzer)
) -> Dict[str, Any]:
    """
    AI-Powered Recovery Plan Generator with Structured Time-Series Output
    
    This endpoint provides VERIFIED itineraries in structured JSON format:
    - ⏰ Time-Series Format: UTC timestamps for Timeline UI rendering
    - 🗺️ Precise Coordinates: Lat/lng for each activity via Google Maps Tool
    - ♿ Accessibility: Stroller/elderly-friendly venue verification
    - 🚗 Transport Modes: Taxi/Train/Walk between activities
    - 🤖 Model: gemini-2.5-flash with structured output
    
    Data Flow:
    1. Fetch flight status from Lufthansa API
    2. Calculate available time (arrival + 30min → connection - 90min)
    3. Generate structured itinerary with precise coordinates
    4. Return time-series JSON for UI rendering
    
    Args:
        request: JSON with:
            - flight_number: Flight number (e.g., "IB774")
            - date: Date in YYYY-MM-DD format
            - connection_departure_time: Next flight departure (ISO format)
            - budget: Budget in EUR (default: 100)
            - preferences: Activity preferences (default: "sightseeing")
            - passenger_type: Type of passenger (default: "standard")
              Options: "standard", "family_with_baby", "elderly", "business"
            - weather_info: Current weather (optional)
            - traffic_info: Current traffic conditions (optional)
    
    Returns:
        - status: "success" or "error"
        - ai_itinerary: Structured JSON with time-series data:
          {
            "city": "Madrid",
            "arrival_airport_coords": {"lat": 40.47, "lng": -3.56},
            "itinerary": [
              {
                "start_utc": "2026-03-01T14:00:00Z",
                "end_utc": "2026-03-01T15:30:00Z",
                "activity_name": "Activity name",
                "location_name": "Venue name",
                "description": "Tailored description for passenger type",
                "latitude": 40.4168,
                "longitude": -3.7038,
                "maps_url": "https://maps.google.com/...",
                "transport_mode_to_next": "Taxi/Train/Walk",
                "accessibility_notes": "Stroller-friendly, elevator access"
              }
            ]
          }
        - structured_output: Boolean indicating if JSON format is used
        - risk_analysis: Connection risk assessment
        - flight_details: Flight information
    """
    try:
        # Extract and validate request data
        flight_number = request.get("flight_number")
        date = request.get("date")
        connection_departure_time = request.get("connection_departure_time")
        budget = request.get("budget", 100)
        preferences = request.get("preferences", "sightseeing")
        passenger_type = request.get("passenger_type", "standard")
        weather_info = request.get("weather_info", "moderate temperature, partly cloudy")
        traffic_info = request.get("traffic_info", "normal traffic conditions")
        
        if not all([flight_number, date, connection_departure_time]):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing required fields: flight_number, date, connection_departure_time"
            )
        
        logger.info(f"[AI Grounding] Generating plan for {flight_number} on {date} (Passenger: {passenger_type})")
        
        # Step 1: Get flight status (internal call)
        try:
            flight_response = lufthansa.get_flight_status(flight_number, date)
        except LufthansaAPIError as e:
            logger.error(f"Lufthansa API error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Lufthansa API unavailable: {str(e)}"
            )
        
        if not flight_response.flights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flight {flight_number} not found for date {date}"
            )
        
        flight = flight_response.flights[0]
        
        # Step 2: Analyze connection risk
        risk = analyzer.calculate_connection_risk(
            arriving_flight=flight,
            next_departure_time=connection_departure_time
        )
        
        # Calculate delay duration
        delay_minutes = analyzer.calculate_delay_duration(flight)
        
        # Step 3: Check if we should trigger AI-powered recovery mode
        risk_summary = analyzer.get_risk_summary(risk)
        
        ai_result = None
        if analyzer.should_trigger_recovery_mode(risk):
            # Step 4: Generate STRUCTURED AI-powered itinerary with Google Maps & Search
            vertex_ai = get_vertex_ai_service()
            
            if vertex_ai:
                logger.info(f"🤖 Triggering AI recovery mode (Structured JSON) for {passenger_type} at {flight.arrival_airport}")
                
                # Calculate start time in UTC (arrival time + 30 min buffer for landing/customs)
                from datetime import datetime, timedelta
                
                # Parse arrival time and add buffer
                if flight.arrival_estimated:
                    arrival_dt = datetime.fromisoformat(flight.arrival_estimated.replace('Z', '+00:00'))
                elif flight.arrival_actual:
                    arrival_dt = datetime.fromisoformat(flight.arrival_actual.replace('Z', '+00:00'))
                else:
                    arrival_dt = datetime.fromisoformat(flight.arrival_scheduled.replace('Z', '+00:00'))
                
                # Add 30 minutes for deplaning and customs
                start_time_utc = (arrival_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # Calculate available hours (up to connection time minus 90 min return buffer)
                connection_dt = datetime.fromisoformat(connection_departure_time.replace('Z', '+00:00'))
                available_minutes = (connection_dt - arrival_dt).total_seconds() / 60 - 90  # 90 min return buffer
                duration_hours = min(max(int(available_minutes / 60), 2), 8)  # Between 2-8 hours
                
                logger.info(f"Start UTC: {start_time_utc}, Duration: {duration_hours}h")
                
                ai_result = vertex_ai.get_structured_itinerary(
                    start_time_utc=start_time_utc,
                    duration_hours=duration_hours,
                    arrival_airport=flight.arrival_airport,
                    arrival_terminal=flight.arrival_terminal or "Unknown",
                    budget=budget,
                    preferences=preferences,
                    passenger_type=passenger_type,
                    weather_info=weather_info,
                    traffic_info=traffic_info
                )
            else:
                logger.warning("⚠️ Vertex AI not configured - recovery mode unavailable")
        
        # Step 5: Return comprehensive response with AI itinerary and grounding data
        return {
            "status": "success",
            "risk_analysis": {
                "is_at_risk": risk.is_at_risk,
                "risk_level": risk.risk_level,
                "risk_emoji": risk_summary["emoji"],
                "buffer_minutes": risk.buffer_minutes,
                "risk_factors": risk.risk_factors,
                "arrival_time": risk.arrival_time,
                "next_departure_time": risk.next_departure_time,
                "recommended_action": risk.recommended_action,
                "recovery_needed": risk_summary["recovery_needed"]
            },
            "flight_details": {
                "flight_number": f"{flight.airline_code}{flight.flight_number}",
                "route": f"{flight.departure_airport} → {flight.arrival_airport}",
                "arrival_terminal": flight.arrival_terminal,
                "delay_minutes": delay_minutes,
                "status": flight.time_status_definition
            },
            "passenger_profile": {
                "passenger_type": passenger_type,
                "budget": budget,
                "preferences": preferences
            },
            "ai_itinerary": ai_result.get("itinerary_data") if ai_result and ai_result.get("success") else None,
            "structured_output": ai_result.get("structured_output", False) if ai_result else False,
            "grounding_enabled": ai_result.get("grounding_enabled", False) if ai_result else False,
            "grounding_metadata": ai_result.get("grounding_metadata") if ai_result and ai_result.get("success") else None,
            "ai_model": ai_result.get("model") if ai_result else None,
            "error": ai_result.get("error") if ai_result and not ai_result.get("success") else None,
            "note": "Structured time-series JSON with UTC timestamps and precise coordinates for Timeline UI and Google Maps integration. Check grounding_metadata to see if Google Maps/Search were used."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate-plan: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


# ============================================================================
# LEGACY ENDPOINTS (Keep for backwards compatibility)
# ============================================================================

@router.post("/generate-plan-legacy", summary="Generate Madrid itinerary plan (Legacy)")
async def generate_plan_legacy(
    scenario: DelayScenario,
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service)
) -> Dict[str, Any]:
    """
    Legacy endpoint - Simple itinerary generation
    
    Creates a Madrid recovery plan based on user input without real-time flight data.
    """
    try:
        logger.info(f"Generating plan for {scenario.delay_location} -> {scenario.last_destination}")
        
        # Use Vertex AI to generate itinerary
        result = vertex_ai.generate_itinerary(
            delay_minutes=0,  # Unknown in legacy mode
            arrival_terminal="Unknown",
            buffer_minutes=0,
            flight_status="DELAYED",
            budget=100,  # Default
            preferences=scenario.user_input,
            duration_hours=6
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate itinerary"
            )
        
        return {
            "status": "success",
            "itinerary": result["itinerary"],
            "metadata": {
                "engine": result["model"],
                "location": scenario.delay_location,
                "destination": scenario.last_destination
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/analyze-flight-risk", summary="Comprehensive flight risk analysis")
async def analyze_flight_risk(
    request: FlightRiskRequest,
    lufthansa: LufthansaClient = Depends(get_lufthansa_client),
    analyzer: FlightConnectionAnalyzer = Depends(get_analyzer)
) -> Dict[str, Any]:
    """
    Complete flight connection risk analysis
    
    Steps:
    1. Query Lufthansa API for origin flight status
    2. Calculate connection risk
    3. If HIGH/CRITICAL risk, generate recovery plan with Gemini (if GCP configured)
    4. Return comprehensive analysis with recommendations
    """
    try:
        logger.info(f"Analyzing risk for {request.origin_flight_number} on {request.origin_departure_date}")
        
        # Step 1: Get origin flight status
        try:
            flight_response = lufthansa.get_flight_status(
                request.origin_flight_number,
                request.origin_departure_date
            )
        except LufthansaAPIError as e:
            logger.error(f"Lufthansa API error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Lufthansa API unavailable: {str(e)}"
            )
        
        if not flight_response.flights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flight {request.origin_flight_number} not found for date {request.origin_departure_date}"
            )
        
        # Get origin flight
        origin_flight = flight_response.flights[0]
        
        # Step 2: Calculate connection risk
        risk = analyzer.calculate_connection_risk(
            arriving_flight=origin_flight,
            next_departure_time=request.connection_departure_time,
            next_flight_number=request.connection_flight_number
        )
        
        # Get risk summary
        risk_summary = analyzer.get_risk_summary(risk)
        delay_minutes = analyzer.calculate_delay_duration(origin_flight)
        
        # Auto-detect layover city if not provided
        layover_city = request.layover_city or origin_flight.arrival_airport
        
        # Step 3: Generate recovery plan if needed (requires GCP)
        recovery_plan = None
        vertex_ai = get_vertex_ai_service()
        
        if analyzer.should_trigger_recovery_mode(risk):
            if vertex_ai:
                logger.info(f"Triggering recovery mode - generating {layover_city} itinerary")
                
                itinerary_result = vertex_ai.generate_itinerary(
                    delay_minutes=delay_minutes,
                    arrival_terminal=origin_flight.arrival_terminal or "Unknown",
                    buffer_minutes=risk.buffer_minutes,
                    flight_status=origin_flight.time_status_definition or "DELAYED",
                    budget=request.budget,
                    preferences=request.preferences,
                    duration_hours=6
                )
                
                if itinerary_result["success"]:
                    recovery_plan = itinerary_result["itinerary"]
            else:
                logger.warning("Recovery mode triggered but GCP not configured - skipping itinerary generation")
        
        # Step 4: Build comprehensive response
        return {
            "status": "success",
            "risk_analysis": {
                "is_at_risk": risk.is_at_risk,
                "risk_level": risk.risk_level,
                "risk_emoji": risk_summary["emoji"],
                "buffer_minutes": risk.buffer_minutes,
                "risk_factors": risk.risk_factors,
                "recommended_action": risk.recommended_action,
                "recovery_needed": risk_summary["recovery_needed"]
            },
            "flight_details": {
                "flight_number": f"{origin_flight.airline_code}{origin_flight.flight_number}",
                "route": f"{origin_flight.departure_airport} → {origin_flight.arrival_airport}",
                "scheduled_departure": origin_flight.departure_scheduled,
                "scheduled_arrival": origin_flight.arrival_scheduled,
                "estimated_arrival": origin_flight.arrival_estimated,
                "actual_arrival": origin_flight.arrival_actual,
                "delay_minutes": delay_minutes,
                "status": origin_flight.time_status_definition,
                "terminal": origin_flight.arrival_terminal,
                "gate": origin_flight.arrival_gate
            },
            "connection": {
                "next_flight": request.connection_flight_number,
                "next_departure": request.connection_departure_time,
                "buffer_minutes": risk.buffer_minutes
            },
            "recovery_plan": recovery_plan,
            "passenger_profile": {
                "budget": request.budget,
                "preferences": request.preferences
            },
            "metadata": {
                "analysis_timestamp": flight_response.timestamp,
                "query_type": flight_response.query_type,
                "ai_model": "gemini-2.5-flash" if recovery_plan else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in risk analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.post("/flight-status-detailed", summary="Get real-time flight status (Detailed)")
async def get_flight_status_detailed(
    flight_number: str,
    date: str,
    lufthansa: LufthansaClient = Depends(get_lufthansa_client)
) -> Dict[str, Any]:
    """
    Query Lufthansa API for flight status with detailed formatting
    
    Args:
        flight_number: Flight number (e.g., IB778)
        date: Date in YYYY-MM-DD format
    """
    try:
        logger.info(f"Querying flight status: {flight_number} on {date}")
        
        response = lufthansa.get_flight_status(flight_number, date)
        
        if not response.flights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flight {flight_number} not found for date {date}"
            )
        
        # Convert to dict
        flights_data = [
            {
                "flight_number": f"{f.airline_code}{f.flight_number}",
                "route": f"{f.departure_airport} → {f.arrival_airport}",
                "departure": {
                    "scheduled": f.departure_scheduled,
                    "estimated": f.departure_estimated,
                    "actual": f.departure_actual,
                    "terminal": f.departure_terminal,
                    "gate": f.departure_gate
                },
                "arrival": {
                    "scheduled": f.arrival_scheduled,
                    "estimated": f.arrival_estimated,
                    "actual": f.arrival_actual,
                    "terminal": f.arrival_terminal,
                    "gate": f.arrival_gate
                },
                "status": {
                    "code": f.time_status_code,
                    "definition": f.time_status_definition,
                    "is_delayed": f.is_delayed(),
                    "is_cancelled": f.is_cancelled()
                },
                "aircraft": {
                    "code": f.aircraft_code,
                    "registration": f.aircraft_registration
                }
            }
            for f in response.flights
        ]
        
        return {
            "status": "success",
            "flights": flights_data,
            "total_count": response.total_count,
            "query_type": response.query_type,
            "timestamp": response.timestamp
        }
        
    except LufthansaAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching flight status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/route-status", summary="Get all flights on a route")
async def get_route_status(
    origin: str,
    destination: str,
    date: str,
    lufthansa: LufthansaClient = Depends(get_lufthansa_client)
) -> Dict[str, Any]:
    """
    Query all flights between two airports on a specific date
    
    Args:
        origin: 3-letter IATA origin airport code (e.g., HAM)
        destination: 3-letter IATA destination airport code (e.g., MAD)
        date: Date in YYYY-MM-DD format
    """
    try:
        logger.info(f"Querying route: {origin} -> {destination} on {date}")
        
        response = lufthansa.get_flight_status_by_route(origin, destination, date)
        
        flights_summary = [
            {
                "flight_number": f"{f.airline_code}{f.flight_number}",
                "departure_time": f.departure_scheduled,
                "arrival_time": f.arrival_scheduled,
                "status": f.time_status_definition,
                "terminal": f.arrival_terminal
            }
            for f in response.flights
        ]
        
        return {
            "status": "success",
            "route": f"{origin} → {destination}",
            "date": date,
            "flights": flights_summary,
            "total_count": response.total_count
        }
        
    except LufthansaAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@router.get("/health", summary="Health check endpoint")
async def health_check(
    settings: Settings = Depends(get_settings),
    lufthansa: LufthansaClient = Depends(get_lufthansa_client)
) -> Dict[str, Any]:
    """
    Check API health and service availability
    """
    from datetime import datetime
    
    # Check if Vertex AI is configured
    vertex_ai_status = "configured" if settings.gcp_project_id else "not_configured"
    
    health_status = {
        "status": "healthy",
        "environment": settings.environment,
        "coverage": "Global - Any Route Worldwide",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": {
            "lufthansa_api": "unknown",
            "vertex_ai": vertex_ai_status,
            "config": "loaded"
        }
    }
    
    # Check Lufthansa API
    try:
        if lufthansa.health_check():
            health_status["services"]["lufthansa_api"] = "healthy"
        else:
            health_status["services"]["lufthansa_api"] = "unhealthy"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["lufthansa_api"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

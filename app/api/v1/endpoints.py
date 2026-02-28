"""
API v1 Endpoints - limited surface
Contains only:
- /test-maps-grounding
- /flight-status
"""

from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.services.lufthansa import LufthansaAPIError, LufthansaClient, get_lufthansa_client
from app.services.vertex_ai import VertexAIService, get_vertex_ai_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Travel Assistant"])

# Lightweight city-to-airport defaults for common global hubs.
CITY_TO_AIRPORT = {
    "madrid": "MAD",
    "barcelona": "BCN",
    "frankfurt": "FRA",
    "munich": "MUC",
    "berlin": "BER",
    "london": "LHR",
    "paris": "CDG",
    "amsterdam": "AMS",
    "rome": "FCO",
    "milan": "MXP",
    "new york": "JFK",
    "los angeles": "LAX",
    "chicago": "ORD",
}

AIRPORT_TO_CITY = {value: key.title() for key, value in CITY_TO_AIRPORT.items()}


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
    """Clean common Gemini artifacts and parse strict JSON object."""
    if not raw_text:
        raise ValueError("Empty model response")

    text = raw_text.replace("\ufeff", "").strip()
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1)
    else:
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

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
        re.sub(r",\s*([\]}])", r"\1", text),
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


def _read_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_iata(raw_code: Any) -> Optional[str]:
    if not isinstance(raw_code, str):
        return None
    letters = "".join(ch for ch in raw_code.upper() if ch.isalpha())
    if len(letters) < 3:
        return None
    return letters[:3]


def _resolve_location_context(request: Dict[str, Any], vertex_ai: VertexAIService) -> tuple[str, str, float, float]:
    city = str(request.get("city", "Unknown City")).strip() or "Unknown City"

    requested_airport = (
        request.get("airport_code")
        or request.get("arrival_airport")
        or request.get("iata_code")
    )
    airport_code = _extract_iata(requested_airport)
    if not airport_code:
        airport_code = CITY_TO_AIRPORT.get(city.lower(), "DEFAULT")

    request_lat = _read_float(request.get("latitude") or request.get("lat"))
    request_lng = _read_float(request.get("longitude") or request.get("lng"))

    if request_lat is not None and request_lng is not None:
        lat, lng = request_lat, request_lng
    else:
        lat, lng = vertex_ai.get_airport_coordinates(airport_code)

    return city, airport_code, lat, lng


def _normalize_to_utc_iso(dt_value: Optional[str]) -> Optional[str]:
    """Normalize ISO datetime to UTC Z format; treat naive values as UTC."""
    if not dt_value:
        return None
    try:
        parsed = datetime.fromisoformat(dt_value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return _iso_z(parsed.astimezone(timezone.utc))
    except ValueError:
        return None


def _enrich_request_from_flight(
    request: Dict[str, Any],
    lufthansa: LufthansaClient,
) -> Dict[str, Any]:
    """
    If flight_number/date is provided, fetch Lufthansa status and hydrate request fields:
    - airport_code (arrival airport)
    - city (best effort from airport)
    - arrival_utc (effective arrival)
    """
    flight_number = request.get("flight_number")
    date = request.get("date")
    if not flight_number or not date:
        return {}

    flight_response = lufthansa.get_flight_status(flight_number, date)
    if not flight_response.flights:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flight {flight_number} not found for date {date}",
        )

    flight = flight_response.flights[0]
    effective_arrival = (
        flight.arrival_actual or flight.arrival_estimated or flight.arrival_scheduled
    )
    normalized_arrival = _normalize_to_utc_iso(effective_arrival)
    if normalized_arrival:
        request["arrival_utc"] = normalized_arrival

    arrival_airport = flight.arrival_airport
    if arrival_airport:
        request["airport_code"] = arrival_airport
        request["arrival_airport"] = arrival_airport
        if not request.get("city"):
            request["city"] = AIRPORT_TO_CITY.get(arrival_airport, "Unknown City")

    if request.get("connection_departure_utc") and not request.get("departure_utc"):
        request["departure_utc"] = request["connection_departure_utc"]

    return {
        "flight_number": flight_number,
        "date": date,
        "arrival_airport": flight.arrival_airport,
        "effective_arrival_raw": effective_arrival,
        "effective_arrival_utc": normalized_arrival,
    }


def _default_city_stops(city: str, airport_code: str, lat: float, lng: float) -> list[Dict[str, Any]]:
    city_query = city.replace(" ", "+")
    return [
        {
            "activity_name": "Travel to City Center",
            "venue_name": f"{airport_code} Airport to {city} City Center",
            "category": "transport",
            "estimated_cost": "EUR10-20",
            "place_id": None,
            "latitude": lat,
            "longitude": lng,
            "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={city_query}+city+center",
            "description": "Transfer from airport to city center.",
            "transport_to_next": "Walk",
        },
        {
            "activity_name": "Quick Local Meal",
            "venue_name": f"{city} Local Market",
            "category": "food",
            "estimated_cost": "EUR15-30",
            "place_id": None,
            "latitude": lat,
            "longitude": lng,
            "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={city_query}+local+market",
            "description": "Short food stop before sightseeing.",
            "transport_to_next": "Walk",
        },
        {
            "activity_name": "City Landmark Visit",
            "venue_name": f"{city} Main Landmark",
            "category": "culture",
            "estimated_cost": "EUR10-20",
            "place_id": None,
            "latitude": lat,
            "longitude": lng,
            "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={city_query}+landmark",
            "description": "Compact cultural stop in the city.",
            "transport_to_next": "Taxi",
        },
        {
            "activity_name": "Scenic Activity Break",
            "venue_name": f"{city} Park or Waterfront",
            "category": "activity",
            "estimated_cost": "Free",
            "place_id": None,
            "latitude": lat,
            "longitude": lng,
            "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={city_query}+park",
            "description": "Short open-air break before returning to airport.",
            "transport_to_next": "Taxi",
        },
        {
            "activity_name": "Return to Airport",
            "venue_name": f"{city} City Center to {airport_code} Airport",
            "category": "transport",
            "estimated_cost": "EUR20-40",
            "place_id": None,
            "latitude": lat,
            "longitude": lng,
            "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={airport_code}+airport",
            "description": "Return transfer with security buffer.",
            "transport_to_next": "N/A",
        },
    ]


def _build_time_series_fallback(
    city: str,
    airport_code: str,
    lat: float,
    lng: float,
    arrival_utc: str,
    departure_utc: str,
) -> Dict[str, Any]:
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

    stops = _default_city_stops(city, airport_code, lat, lng)
    weights = [1] * len(stops)

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


def _minutes_to_human(minutes: int) -> Optional[str]:
    if minutes <= 0:
        return None
    hours = minutes // 60
    rem = minutes % 60
    if hours and rem:
        return f"{hours}h {rem}m"
    if hours:
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    return f"{rem} min"


@router.post("/test-maps-grounding", summary="Test Google Maps Grounding with Layover Plan")
async def test_maps_grounding(
    request: Dict[str, Any],
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service),
    lufthansa: LufthansaClient = Depends(get_lufthansa_client),
) -> Dict[str, Any]:
    """Test layover itinerary generation with robust JSON parsing."""
    try:
        if not vertex_ai:
            return {
                "status": "error",
                "message": "Gemini AI not configured. Set GOOGLE_GEMINI_API_KEY in .env file",
            }

        # Optional workflow: hydrate itinerary context from Lufthansa flight status.
        flight_context = _enrich_request_from_flight(request, lufthansa)

        city = str(request.get("city", "Unknown City")).strip() or "Unknown City"
        arrival_utc = request.get("arrival_utc", "2026-03-01T14:00:00Z")
        departure_utc = (
            request.get("departure_utc")
            or request.get("connection_departure_utc")
            or "2026-03-01T20:00:00Z"
        )
        if request.get("flight_number") and request.get("date"):
            if not (request.get("departure_utc") or request.get("connection_departure_utc")):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="When using flight_number/date, provide departure_utc or connection_departure_utc",
                )
        weather = request.get("weather", "sunny, 22C")
        preferences = request.get("preferences", ["sightseeing", "local food"])

        logger.info(
            "Maps grounding test: %s | %s -> %s (client: %s)",
            city,
            arrival_utc,
            departure_utc,
            vertex_ai.client_type,
        )

        city, airport_code, lat, lng = _resolve_location_context(request, vertex_ai)

        from google.genai.types import GenerateContentConfig, Tool
        try:
            from google.genai.types import GoogleMaps  # type: ignore
            has_google_maps = True
        except ImportError:
            GoogleMaps = None  # type: ignore
            has_google_maps = False

        maps_mode_instruction = (
            "Google Maps grounding is active. Use real verified venues and real place_id values."
            if vertex_ai.client_type == "vertex_ai"
            else "Google Maps grounding is not available in this mode. Use realistic venues and set place_id to null when unavailable."
        )
        place_id_instruction = (
            "place_id: Google Maps CID (long numeric string) from actual Maps search"
            if vertex_ai.client_type == "vertex_ai"
            else "place_id: Use null when a verified Maps CID is not available in API key mode"
        )

        prompt = f"""
Role: You are the Layover Architect.

CONTEXT:
- City: {city}
- Arrival Airport: {airport_code}
- Weather: {weather}
- Arrival Airport (UTC): {arrival_utc}
- Departure from Airport (UTC): {departure_utc}
- Airport Coordinates: {lat}, {lng}
- User Preferences: {', '.join(preferences)}
- Mode: {vertex_ai.client_type}
- Grounding Note: {maps_mode_instruction}

TASK: Create a time-sequenced layover itinerary with 4-6 activities for this city and airport.

Rules:
1. Add 30 minutes after arrival before first activity.
2. End last activity at departure minus 90 minutes.
3. First and last activity category must be transport.
4. Each activity must include:
   start_utc, end_utc, activity_name, venue_name, category, estimated_cost,
   place_id, latitude, longitude, google_maps_url, description, transport_to_next.
5. {place_id_instruction}
6. Output valid JSON only, no markdown.
"""

        if vertex_ai.client_type == "vertex_ai" and has_google_maps and GoogleMaps is not None:
            config = GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4000,
                tools=[Tool(google_maps=GoogleMaps())],
            )
        else:
            config = GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4000,
            )

        response = vertex_ai.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )

        full_text = _extract_response_text(response)
        raw_model_text = full_text

        try:
            itinerary_data = _repair_and_parse_json(full_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("JSON parse failed (first pass): %s", e)
            repair_prompt = f"""
Convert the text below into ONE valid JSON object.
- Output JSON only.
- Keep original keys and values where possible.
- Fix invalid syntax only.

TEXT TO REPAIR:
{full_text}
"""
            try:
                repair_response = vertex_ai.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=repair_prompt,
                    config=GenerateContentConfig(temperature=0.0, max_output_tokens=4000),
                )
                repaired_text = _extract_response_text(repair_response)
                itinerary_data = _repair_and_parse_json(repaired_text)
                itinerary_data["_json_repaired"] = True
            except Exception:
                itinerary_data = {
                    "error": "JSON_PARSE_ERROR",
                    "message": f"Failed to parse Gemini response as JSON: {str(e)}",
                    "raw_response": full_text[:1000],
                }

        should_fallback = (
            not isinstance(itinerary_data, dict)
            or "error" in itinerary_data
            or not isinstance(itinerary_data.get("itinerary"), list)
            or len(itinerary_data.get("itinerary", [])) < 4
        )
        if should_fallback:
            itinerary_data = _build_time_series_fallback(
                city=city,
                airport_code=airport_code,
                lat=lat,
                lng=lng,
                arrival_utc=arrival_utc,
                departure_utc=departure_utc,
            )

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
                                maps_chunks.append(
                                    {
                                        "title": chunk.maps.title if hasattr(chunk.maps, "title") else None,
                                        "uri": chunk.maps.uri if hasattr(chunk.maps, "uri") else None,
                                        "place_id": getattr(chunk.maps, "placeId", None),
                                    }
                                )
            grounding_metadata = {
                "maps_used": len(maps_chunks) > 0,
                "maps_venues_found": len(maps_chunks),
                "maps_data": maps_chunks,
            }

        return {
            "status": "success",
            "model": "gemini-2.5-flash",
            "client_type": vertex_ai.client_type,
            "city": city,
            "airport_code": airport_code,
            "coordinates": {"lat": lat, "lng": lng},
            "time_window": {"arrival": arrival_utc, "departure": departure_utc},
            "flight_context": flight_context or None,
            "raw_model_text": raw_model_text,
            "data": itinerary_data,
            "grounding_metadata": grounding_metadata,
            "note": "Maps grounding active" if vertex_ai.client_type == "vertex_ai" else "API Key mode",
        }

    except Exception as e:
        logger.error("Maps grounding test failed: %s", str(e))
        import traceback

        return {
            "status": "error",
            "message": f"Test failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@router.post("/test-maps-grounding-v2", summary="Test Maps Grounding via V2 API-key flow (v1alpha)")
async def test_maps_grounding_v2(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    V2-style direct API key endpoint (Jupyter/Colab-aligned):
    - Uses GOOGLE_GEMINI_API_KEY directly
    - Uses genai.Client(api_key=..., http_options={'api_version': 'v1alpha'})
    - Accepts frontend_answers payload structure
    """
    try:
        api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "message": "GOOGLE_GEMINI_API_KEY not configured"
            }

        city = request.get("city", "Madrid")
        arrival_utc = request.get("arrival_utc", "2026-03-01T11:00:00Z")
        departure_utc = request.get("departure_utc", "2026-03-01T19:00:00Z")
        weather = request.get("weather", "Sunny")
        frontend_answers = request.get("frontend_answers", [])

        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"},
        )

        prompt = f"""
Role: You are the Layover Architect.
Goal: Create a high-precision, JSON-formatted travel itinerary with costs and categories.

CONTEXT:
- City: {city} | Weather: {weather}
- Arrival (UTC): {arrival_utc} | Departure (UTC): {departure_utc}
- User Preferences: {json.dumps(frontend_answers)}

STRICT LOGICAL RULES:
1. Safe Window: Calculate buffers (30m deplaning, 35m passport, 90m return).
2. Categories: Each activity MUST have a 'category' field: "food", "activity", "transport", or "culture".
3. Costs: Each activity MUST have an 'estimated_cost' string (e.g., "€15", "Free", "€25-40").
4. Language: English only.
5. Clean Output: Ensure descriptions do NOT contain citations or reference numbers in brackets.
6. Return 4-6 activities, first and last must be transport.

OUTPUT JSON STRUCTURE:
{{
  "stay_at_airport": false,
  "safe_window_minutes": int,
  "itinerary": [
    {{
      "start_utc": "ISO8601",
      "end_utc": "ISO8601",
      "activity_name": "string",
      "venue_name": "string",
      "category": "food | activity | transport | culture",
      "estimated_cost": "string",
      "place_id": "string or null",
      "latitude": float,
      "longitude": float,
      "google_maps_url": "string",
      "description": "string",
      "transport_to_next": "string"
    }}
  ]
}}

Output JSON only. No markdown.
"""

        config_kwargs: Dict[str, Any] = {"temperature": 0.1}
        try:
            # Some SDK versions may not include GoogleMaps type.
            config_kwargs["tools"] = [types.Tool(google_maps=types.GoogleMaps())]
            maps_tool_enabled = True
        except Exception:
            maps_tool_enabled = False

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        full_text = getattr(response, "text", None) or ""
        if not full_text and getattr(response, "candidates", None):
            parts = getattr(getattr(response.candidates[0], "content", None), "parts", None) or []
            full_text = "".join([p.text for p in parts if hasattr(p, "text") and p.text])

        try:
            data = _repair_and_parse_json(full_text)
        except Exception as parse_err:
            data = {
                "error": "JSON_PARSE_ERROR",
                "message": str(parse_err),
                "raw_response": full_text[:2000],
            }

        return {
            "status": "success",
            "endpoint": "test-maps-grounding-v2",
            "model": "gemini-2.5-flash",
            "api_mode": "api_key_v1alpha",
            "maps_tool_enabled": maps_tool_enabled,
            "city": city,
            "time_window": {"arrival": arrival_utc, "departure": departure_utc},
            "frontend_answers": frontend_answers,
            "raw_model_text": full_text,
            "data": data,
        }
    except Exception as e:
        logger.error("V2 maps grounding test failed: %s", str(e))
        import traceback
        return {
            "status": "error",
            "message": f"V2 test failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@router.get("/flight-status", summary="Get real-time flight status")
async def get_flight_status_core(
    flight_number: str,
    date: str,
    lufthansa: LufthansaClient = Depends(get_lufthansa_client),
) -> Dict[str, Any]:
    """Query Lufthansa API for flight status."""
    try:
        logger.info("[Core] Querying flight status: %s on %s", flight_number, date)
        response = lufthansa.get_flight_status(flight_number, date)

        if not response.flights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flight {flight_number} not found for date {date}",
            )

        return {
            "flights": [flight.model_dump() for flight in response.flights],
            "total_count": response.total_count,
            "query_type": response.query_type,
            "timestamp": response.timestamp,
        }

    except LufthansaAPIError as e:
        logger.error("Lufthansa API error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Lufthansa API unavailable: {str(e)}",
        )
    except Exception as e:
        logger.error("Error fetching flight status: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/check-flight", summary="Check flight and return compact delay/layover card data")
async def check_flight(
    request: Dict[str, Any],
    lufthansa: LufthansaClient = Depends(get_lufthansa_client),
) -> Dict[str, Any]:
    """
    Returns compact flight card fields:
    {
      "status": "delayed" | "layover",
      "city": "Berlin",
      "delayTime": "2 hours" | null,
      "safe_window_minutes": 180,
      "weather_info": {"degree": 8, "status": "cloudy"}
    }
    """
    try:
        flight_number = request.get("flight_number")
        date = request.get("date")
        if not flight_number or not date:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing required fields: flight_number, date",
            )

        response = lufthansa.get_flight_status(flight_number, date)
        if not response.flights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flight {flight_number} not found for date {date}",
            )

        flight = response.flights[0]
        arrival_airport = flight.arrival_airport or "DEFAULT"
        city = request.get("city") or AIRPORT_TO_CITY.get(arrival_airport, arrival_airport)

        weather_payload = request.get("weather_info") or {}
        weather_info = {
            "degree": weather_payload.get("degree", request.get("weather_degree")),
            "status": weather_payload.get("status", request.get("weather_status", "unknown")),
        }

        scheduled_arrival = _normalize_to_utc_iso(flight.arrival_scheduled)
        effective_arrival = _normalize_to_utc_iso(
            flight.arrival_actual or flight.arrival_estimated or flight.arrival_scheduled
        )

        delay_minutes = 0
        if scheduled_arrival and effective_arrival:
            delay_minutes = max(0, int((_parse_utc(effective_arrival) - _parse_utc(scheduled_arrival)).total_seconds() // 60))

        connection_departure_utc = request.get("connection_departure_utc") or request.get("departure_utc")
        if connection_departure_utc and effective_arrival:
            safe_window_minutes = int(
                (_parse_utc(connection_departure_utc) - _parse_utc(effective_arrival)).total_seconds() // 60
            ) - 120
            safe_window_minutes = max(0, safe_window_minutes)
        else:
            # Fallback estimate when no connection departure is provided
            safe_window_minutes = max(0, 360 - delay_minutes)

        is_delayed = delay_minutes > 0 or flight.is_delayed()
        return {
            "status": "delayed" if is_delayed else "layover",
            "city": city,
            "delayTime": _minutes_to_human(delay_minutes) if is_delayed else None,
            "safe_window_minutes": safe_window_minutes,
            "weather_info": weather_info,
        }
    except LufthansaAPIError as e:
        logger.error("Lufthansa API error in check-flight: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Lufthansa API unavailable: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in check-flight: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

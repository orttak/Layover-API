"""Vertex AI Service - Google AI Gemini integration
Orchestrates AI-powered itinerary generation and status updates
WITH Google Maps integration and dynamic location-aware grounding
"""

import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleMaps,
    HttpOptions,
    Tool,
    ToolConfig,
    RetrievalConfig,
    LatLng
)

from app.prompts.manager import PromptManager

logger = logging.getLogger(__name__)


# Airport coordinates mapping (major airports worldwide)
AIRPORT_COORDINATES = {
    # Spain
    "MAD": (40.4168, -3.7038),  # Madrid
    "BCN": (41.2974, 2.0833),   # Barcelona
    # Germany
    "FRA": (50.0379, 8.5622),   # Frankfurt
    "MUC": (48.3537, 11.7750),  # Munich
    "BER": (52.3667, 13.5033),  # Berlin
    # UK
    "LHR": (51.4700, -0.4543),  # London Heathrow
    "LGW": (51.1537, -0.1821),  # London Gatwick
    # France
    "CDG": (49.0097, 2.5479),   # Paris CDG
    "ORY": (48.7233, 2.3794),   # Paris Orly
    # Netherlands
    "AMS": (52.3105, 4.7683),   # Amsterdam
    # Italy
    "FCO": (41.8003, 12.2389),  # Rome
    "MXP": (45.6301, 8.7231),   # Milan
    # USA
    "JFK": (40.6413, -73.7781), # New York JFK
    "LAX": (33.9416, -118.4085),# Los Angeles
    "ORD": (41.9742, -87.9073), # Chicago
    # Default fallback (Madrid)
    "DEFAULT": (40.4168, -3.7038)
}


class VertexAIService:
    """
    Service for interacting with Google AI (Gemini models)
    
    Features:
    - Google Maps integration with dynamic location-aware grounding
    - Automatic airport coordinate detection from Lufthansa API
    - Prompt-as-Config integration
    - Dynamic model parameter loading
    - Error handling and retries
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize Google AI service (supports both API Key and Vertex AI)
        
        Args:
            api_key: Google Gemini API key (uses GOOGLE_GEMINI_API_KEY env var if not provided)
            project_id: GCP Project ID for Vertex AI (optional)
            location: GCP Location for Vertex AI (default: europe-west3)
            prompt_manager: Optional PromptManager instance
        """
        # Check if we should use Vertex AI or API Key
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location or os.getenv("GCP_LOCATION", "europe-west3")
        self.api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        
        # Blacklist incomplete/invalid project IDs
        invalid_project_ids = [
            "your-project-id", 
            "your-gcp-project-id", 
            "your-complete-gcp-project-id",
            "project-b7be3ea5-e603-419c-87d",  # Incomplete project ID
            ""
        ]
        
        if self.project_id and self.project_id not in invalid_project_ids:
            # Use Vertex AI Client
            try:
                self.client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location
                )
                logger.info(f"Initialized Vertex AI client: {self.project_id} @ {self.location}")
                self.client_type = "vertex_ai"
            except Exception as e:
                logger.warning(f"Vertex AI init failed: {e}, falling back to API Key")
                if not self.api_key:
                    raise ValueError("Neither Vertex AI nor API Key configured properly")
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options=HttpOptions(api_version="v1")
                )
                logger.info("Initialized Google AI client with API Key")
                self.client_type = "api_key"
        else:
            # Use API Key Client
            if not self.api_key:
                raise ValueError("GOOGLE_GEMINI_API_KEY not provided")
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=HttpOptions(api_version="v1")
            )
            logger.info("Initialized Google AI client with API Key and Maps integration")
            self.client_type = "api_key"
        
        # Prompt manager for config-based prompts
        self.prompt_manager = prompt_manager or PromptManager()
        
        # Google Maps tool configuration
        self._maps_tool = Tool(
            google_maps=GoogleMaps(
                enable_widget=False  # Return venue data, not widget tokens
            )
        )
        logger.info("Google Maps tool initialized")
    
    @staticmethod
    def get_airport_coordinates(airport_code: str) -> Tuple[float, float]:
        """
        Get coordinates for an airport code
        
        Args:
            airport_code: 3-letter IATA airport code (e.g., MAD, FRA, JFK)
        
        Returns:
            Tuple of (latitude, longitude)
        """
        coords = AIRPORT_COORDINATES.get(airport_code.upper())
        if coords:
            logger.info(f"Found coordinates for {airport_code}: {coords}")
            return coords
        else:
            logger.warning(f"Airport {airport_code} not in database, using default (Madrid)")
            return AIRPORT_COORDINATES["DEFAULT"]
    
    def _get_generation_config(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_output_tokens: int = 2048,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ) -> GenerateContentConfig:
        """
        Create generation config with Maps tool and location coordinates
        
        Args:
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_output_tokens: Maximum tokens to generate
            latitude: Location latitude (optional, uses default if not provided)
            longitude: Location longitude (optional, uses default if not provided)
        
        Returns:
            GenerateContentConfig with Maps integration
        """
        # Use provided coordinates or default to Madrid
        if latitude is None or longitude is None:
            latitude, longitude = AIRPORT_COORDINATES["DEFAULT"]
            logger.info(f"Using default coordinates: ({latitude}, {longitude})")
        
        return GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            tools=[self._maps_tool],
            tool_config=ToolConfig(
                retrieval_config=RetrievalConfig(
                    lat_lng=LatLng(
                        latitude=latitude,
                        longitude=longitude
                    ),
                    language_code="en_US"
                )
            )
        )
    
    def get_structured_itinerary(
        self,
        start_time_utc: str,
        duration_hours: int,
        arrival_airport: str,
        arrival_terminal: str,
        budget: int,
        preferences: str,
        passenger_type: str = "standard",
        weather_info: str = "moderate temperature",
        traffic_info: str = "normal traffic"
    ) -> Dict[str, Any]:
        """
        Generate STRUCTURED time-series itinerary with precise coordinates
        
        Returns JSON with:
        - UTC timestamps for each activity
        - Precise lat/lng from Google Maps Tool
        - Transport modes between activities
        - Stroller/elderly accessibility info
        
        Args:
            start_time_utc: ISO 8601 UTC start time (e.g., "2026-03-01T14:00:00Z")
            duration_hours: Total itinerary duration in hours
            arrival_airport: IATA code (e.g., MAD, FRA, JFK)
            arrival_terminal: Terminal number/name
            budget: Budget in EUR
            preferences: Activity preferences
            passenger_type: standard, family_with_baby, elderly, business
            weather_info: Current weather
            traffic_info: Traffic conditions
        
        Returns:
            Structured JSON with time-series itinerary
        """
        try:
            # Get coordinates for arrival airport
            latitude, longitude = self.get_airport_coordinates(arrival_airport)
            city_name = self._get_city_name(arrival_airport)
            
            logger.info(f"Generating STRUCTURED itinerary for {passenger_type} at {arrival_airport}")
            
            # 🎯 MASTER PROMPT - Directive ve JSON Schema zorunluluğu
            if self.client_type == "vertex_ai":
                maps_instruction = "1. 🗺️ Google Maps aracını (tool) KESINLIKLE kullanarak mekanların gerçekliğini doğrula."
            else:
                maps_instruction = "1. 🗺️ Bildiğin gerçek mekanları öner (Google Maps verisi kullanılamıyor, API key modunda)."
            
            master_prompt = f"""Sistem Rolü:
Sen dünyanın en iyi seyahat asistanı ve lojistik uzmanısın. Kullanıcının uçuş riski veya gecikmesi nedeniyle bulunduğu şehirde geçireceği süreyi (recovery plan) yönetiyorsun.

Görev:
Aşağıdaki parametrelere göre {duration_hours} saatlik, adım adım bir zaman çizelgesi (time-series) oluştur:
- Şehir: {city_name}
- Havalimanı: {arrival_airport} (Terminal: {arrival_terminal})
- Yolcu Tipi: {passenger_type}
- Başlangıç Zamanı (UTC): {start_time_utc}
- Bütçe: {budget} EUR
- Tercihler: {preferences}
- Hava Durumu: {weather_info}
- Trafik Durumu: {traffic_info}

Kritik Kurallar:
{maps_instruction}
2. ✅ Her durak için MUTLAKA gerçek bir 'maps_url' üret (format: https://www.google.com/maps/search/?api=1&query=VENUE+{city_name.replace(' ', '+')}).
3. 📍 Koordinatları (latitude, longitude) float tipinde KESIN olarak ver.
4. ⏰ Tüm zamanları ISO 8601 UTC formatında (YYYY-MM-DDTHH:MM:SSZ) ayarla.
5. 🎯 JSON dışında hiçbir metin, açıklama veya markdown karakteri (```json gibi) DÖNDÜRME. Sadece saf JSON objesini döndür.
6. 👶 Yolcu tipi "{passenger_type}" ise, açıklamalarda bebek dostu/erişilebilirlik detaylarını MUTLAKA belirt.
7. 🚦 Aktiviteler arasında gerçekçi ulaşım süreleri ekle (Taxi/Metro/Walk).
8. 💰 Tüm öneriler {budget} EUR bütçesine uygun olmalı.

Çıktı Şeması (Bu yapıya %100 uymak ZORUNDASIN):
{{
  "city": "{city_name}",
  "arrival_airport_coords": {{"lat": {latitude}, "lng": {longitude}}},
  "itinerary": [
    {{
      "start_utc": "YYYY-MM-DDTHH:MM:SSZ",
      "end_utc": "YYYY-MM-DDTHH:MM:SSZ",
      "activity_name": "Museum Visit",
      "location_name": "Prado Museum",
      "description": "World-class art collection. Tailored for {passenger_type}.",
      "latitude": 40.4138,
      "longitude": -3.6921,
      "maps_url": "https://www.google.com/maps/search/?api=1&query=Prado+Museum+Madrid",
      "transport_mode_to_next": "Taxi (15 min)",
      "accessibility_notes": "Elevator, stroller access available"
    }}
  ]
}}

ÖNEMLI: Yanıtını JSON olarak başlat ve bitir. Hiçbir ek açıklama ekleme."""
            
            # Generation config with STRICT structured output and LOW temperature
            # Note: Tools/grounding only work with Vertex AI, not API key mode
            if self.client_type == "vertex_ai":
                generation_config = GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.8,
                    top_k=20,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    tools=[self._maps_tool],
                    tool_config=ToolConfig(
                        retrieval_config=RetrievalConfig(
                            lat_lng=LatLng(
                                latitude=latitude,
                                longitude=longitude
                            ),
                            language_code="en_US"
                        )
                    )
                )
                logger.info(f"🚀 Calling gemini-2.5-flash with MASTER PROMPT + Maps Tool (Vertex AI mode, temp=0.2)...")
            else:
                # API key mode - no tools/grounding support
                generation_config = GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.8,
                    top_k=20,
                    max_output_tokens=4096,
                    response_mime_type="application/json"
                )
                logger.info(f"🚀 Calling gemini-2.5-flash with MASTER PROMPT (API key mode, temp=0.2)...")
            
            # Call Gemini 2.5 Flash with structured output and MASTER PROMPT
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=master_prompt,
                config=generation_config
            )
            
            # Extract grounding metadata to verify Maps/Search usage (only if Vertex AI mode)
            grounding_metadata = self._extract_grounding_metadata(response) if self.client_type == "vertex_ai" else {}
            maps_used = len(grounding_metadata.get("maps_data", [])) > 0
            search_used = len(grounding_metadata.get("sources", [])) > 0
            
            if self.client_type == "vertex_ai":
                logger.info(f"🗺️ Google Maps used: {maps_used} ({len(grounding_metadata.get('maps_data', []))} venues)")
                logger.info(f"🔍 Google Search used: {search_used} ({len(grounding_metadata.get('sources', []))} sources)")
            else:
                logger.info(f"ℹ️ API key mode: Grounding not available (use Vertex AI for Maps/Search grounding)")
            
            # Parse JSON response
            import json
            try:
                itinerary_data = json.loads(response.text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Fallback: try to extract JSON from response
                text = response.text.strip()
                if text.startswith('```json'):
                    text = text.split('```json')[1].split('```')[0]
                itinerary_data = json.loads(text)
            
            logger.info(f"Successfully generated structured itinerary with {len(itinerary_data.get('itinerary', []))} activities")
            
            return {
                "success": True,
                "itinerary_data": itinerary_data,
                "model": "gemini-2.5-flash",
                "client_type": self.client_type,  # vertex_ai or api_key
                "temperature": 0.2,  # Low temp for structured output
                "passenger_type": passenger_type,
                "grounding_enabled": self.client_type == "vertex_ai",  # Only available in Vertex AI mode
                "grounding_metadata": {
                    "google_maps_used": maps_used if self.client_type == "vertex_ai" else False,
                    "google_search_used": search_used if self.client_type == "vertex_ai" else False,
                    "maps_venues_found": len(grounding_metadata.get("maps_data", [])) if self.client_type == "vertex_ai" else 0,
                    "search_sources_found": len(grounding_metadata.get("sources", [])) if self.client_type == "vertex_ai" else 0,
                    "maps_data": grounding_metadata.get("maps_data", []) if self.client_type == "vertex_ai" else [],
                    "search_sources": grounding_metadata.get("sources", []) if self.client_type == "vertex_ai" else [],
                    "entry_points": grounding_metadata.get("entry_points", []) if self.client_type == "vertex_ai" else [],
                    "note": "✅ Google Maps Tool enabled" if self.client_type == "vertex_ai" else "ℹ️ Using API key mode - Maps grounding requires Vertex AI (set GCP_PROJECT_ID)"
                },
                "structured_output": True,
                "location": f"{arrival_airport} ({latitude}, {longitude})"
            }
            
        except Exception as e:
            logger.error(f"Structured generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "itinerary_data": None,
                "structured_output": False
            }
    
    def get_grounded_itinerary(
        self,
        delay_minutes: int,
        arrival_terminal: str,
        buffer_minutes: int,
        flight_status: str,
        budget: int,
        preferences: str,
        duration_hours: int = 6,
        passenger_type: str = "standard",
        weather_info: str = "moderate temperature",
        traffic_info: str = "normal traffic",
        arrival_airport: str = "MAD"
    ) -> Dict[str, Any]:
        """
        Generate grounded recovery itinerary using Gemini + Google Search
        
        This method uses Google Search Grounding to verify real-time data like:
        - Current venue opening hours
        - Stroller accessibility for families
        - Real-time traffic and weather conditions
        
        Args:
            delay_minutes: Flight delay duration
            arrival_terminal: Madrid arrival terminal
            buffer_minutes: Connection buffer time
            flight_status: Current flight status
            budget: Passenger budget in EUR
            preferences: Passenger preferences
            duration_hours: Available time for itinerary
            passenger_type: Type of passenger (standard, family_with_baby, elderly, etc.)
            weather_info: Current weather conditions
            traffic_info: Current traffic conditions
        
        Returns:
            Dictionary with itinerary, grounding sources, and maps links
        """
        try:
            logger.info(f"Generating GROUNDED itinerary for {passenger_type} passenger")
            
            # Load prompt template
            prompt_vars = {
                'delay_minutes': delay_minutes,
                'arrival_terminal': arrival_terminal,
                'buffer_minutes': buffer_minutes,
                'flight_status': flight_status,
                'budget': budget,
                'preferences': preferences,
                'duration_hours': duration_hours,
                'passenger_type': passenger_type,
                'weather_info': weather_info,
                'traffic_info': traffic_info
            }
            
            prompt_data = self.prompt_manager.format_prompt('itinerary_gen', prompt_vars)
            
            # Get model name and generation config
            model_name = prompt_data['model_name']
            
            # Generation config with dynamic airport coordinates
            generation_config = self._get_generation_config(
                temperature=prompt_data['parameters'].get('temperature', 0.7),
                top_p=prompt_data['parameters'].get('top_p', 0.9),
                top_k=prompt_data['parameters'].get('top_k', 40),
                max_output_tokens=prompt_data['parameters'].get('max_output_tokens', 2048),
                latitude=latitude,
                longitude=longitude
            )
            
            # Generate with Google Maps integration
            logger.info(f"Calling Gemini with Google Maps + location grounding ({arrival_airport})...")
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt_data['prompt'],
                config=generation_config
            )
            
            # Extract grounding metadata
            grounding_metadata = self._extract_grounding_metadata(response)
            
            # Parse response
            itinerary_text = response.text
            
            # Get city name from airport code
            city_name = self._get_city_name(arrival_airport)
            
            # Generate Google Maps links
            maps_links = self._generate_maps_links(itinerary_text, city_name)
            
            return {
                "success": True,
                "itinerary": itinerary_text,
                "model": model_name,
                "grounding_sources": grounding_metadata.get("sources", []),
                "maps_data": grounding_metadata.get("maps_data", []),
                "search_entry_points": grounding_metadata.get("entry_points", []),
                "maps_links": maps_links,
                "passenger_type": passenger_type,
                "grounding_enabled": True,
                "location": f"{arrival_airport} ({latitude}, {longitude})"
            }
            
        except Exception as e:
            logger.error(f"Grounding failed: {str(e)}. Falling back to standard generation...")
            # Fallback to standard generation without grounding
            prompt_vars['arrival_airport'] = arrival_airport
            return self._fallback_generation(prompt_vars)
    
    def _extract_grounding_metadata(self, response) -> Dict[str, Any]:
        """Extract grounding metadata from Google Maps-enhanced response"""
        try:
            metadata = {
                "sources": [],
                "maps_data": [],
                "entry_points": []
            }
            
            # Extract Maps grounding data
            if hasattr(response, 'grounding_metadata'):
                grounding = response.grounding_metadata
                
                # Maps-specific grounding chunks
                if hasattr(grounding, 'grounding_chunks'):
                    for chunk in grounding.grounding_chunks:
                        chunk_info = {}
                        
                        # Web sources
                        if hasattr(chunk, 'web') and chunk.web:
                            chunk_info["type"] = "web"
                            chunk_info["url"] = chunk.web.uri if hasattr(chunk.web, 'uri') else None
                            chunk_info["title"] = chunk.web.title if hasattr(chunk.web, 'title') else None
                        
                        # Maps places
                        if hasattr(chunk, 'retrieved_context') and chunk.retrieved_context:
                            chunk_info["type"] = "maps_place"
                            chunk_info["text"] = chunk.retrieved_context.text if hasattr(chunk.retrieved_context, 'text') else None
                            chunk_info["uri"] = chunk.retrieved_context.uri if hasattr(chunk.retrieved_context, 'uri') else None
                        
                        if chunk_info:
                            if chunk_info.get("type") == "maps_place":
                                metadata["maps_data"].append(chunk_info)
                            else:
                                metadata["sources"].append(chunk_info)
                
                # Search entry points
                if hasattr(grounding, 'search_entry_point'):
                    metadata["entry_points"].append({
                        "rendered_content": grounding.search_entry_point.rendered_content
                    })
                
                # Grounding supports (citations)
                if hasattr(grounding, 'grounding_supports'):
                    for support in grounding.grounding_supports:
                        if hasattr(support, 'segment'):
                            support_info = {
                                "text": support.segment.text if hasattr(support.segment, 'text') else None,
                                "start_index": support.segment.start_index if hasattr(support.segment, 'start_index') else None,
                                "end_index": support.segment.end_index if hasattr(support.segment, 'end_index') else None
                            }
                            metadata["sources"].append(support_info)
            
            logger.info(f"Extracted {len(metadata['maps_data'])} Maps venues, {len(metadata['sources'])} web sources")
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract grounding metadata: {str(e)}")
            return {"sources": [], "maps_data": [], "entry_points": []}
    
    @staticmethod
    def _get_city_name(airport_code: str) -> str:
        """Get city name from airport code"""
        city_mapping = {
            "MAD": "Madrid", "BCN": "Barcelona",
            "FRA": "Frankfurt", "MUC": "Munich", "BER": "Berlin",
            "LHR": "London", "LGW": "London",
            "CDG": "Paris", "ORY": "Paris",
            "AMS": "Amsterdam",
            "FCO": "Rome", "MXP": "Milan",
            "JFK": "New York", "LAX": "Los Angeles", "ORD": "Chicago"
        }
        return city_mapping.get(airport_code.upper(), airport_code)
    
    def _generate_maps_links(self, itinerary_text: str, city: str) -> List[Dict[str, str]]:
        """Generate Google Maps links for mentioned venues"""
        # Simple implementation: extract potential venue names
        # In production, use NLP or regex to identify venues
        maps_links = []
        
        # Example venues that might be mentioned
        common_venues = [
            "Prado Museum", "Retiro Park", "Plaza Mayor", 
            "Royal Palace", "Puerta del Sol", "Gran Via"
        ]
        
        for venue in common_venues:
            if venue.lower() in itinerary_text.lower():
                # Generate Google Maps search URL
                maps_url = f"https://www.google.com/maps/search/{venue.replace(' ', '+')}+{city}"
                maps_links.append({
                    "venue": venue,
                    "maps_url": maps_url
                })
        
        logger.info(f"Generated {len(maps_links)} Google Maps links")
        return maps_links
    
    def _fallback_generation(self, prompt_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to standard generation without Maps grounding"""
        try:
            logger.warning("Using fallback generation without Google Maps grounding")
            
            prompt_data = self.prompt_manager.format_prompt('itinerary_gen', prompt_vars)
            
            # Simple config without Maps tool
            simple_config = GenerateContentConfig(
                temperature=prompt_data['parameters'].get('temperature', 0.7),
                top_p=prompt_data['parameters'].get('top_p', 0.9),
                max_output_tokens=prompt_data['parameters'].get('max_output_tokens', 2048)
                # NO tools
            )
            
            response = self.client.models.generate_content(
                model=prompt_data['model_name'],
                contents=prompt_data['prompt'],
                config=simple_config
            )
            
            return {
                "success": True,
                "itinerary": response.text,
                "model": prompt_data['model_name'],
                "grounding_sources": [],
                "maps_data": [],
                "search_entry_points": [],
                "maps_links": [],
                "grounding_enabled": False,
                "warning": "Google Maps grounding unavailable - using standard generation"
            }
            
        except Exception as e:
            logger.error(f"Fallback generation also failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "itinerary": None,
                "grounding_enabled": False
            }
    
    def generate_itinerary(
        self,
        delay_minutes: int,
        arrival_terminal: str,
        buffer_minutes: int,
        flight_status: str,
        budget: int,
        preferences: str,
        duration_hours: int = 6
    ) -> Dict[str, Any]:
        """
        Generate Madrid recovery itinerary using Gemini
        
        Args:
            delay_minutes: Flight delay duration
            arrival_terminal: Madrid arrival terminal
            buffer_minutes: Connection buffer time
            flight_status: Current flight status
            budget: Passenger budget in EUR
            preferences: Passenger preferences (comma-separated)
            duration_hours: Available time for itinerary
        
        Returns:
            Dictionary with itinerary details
        
        Example:
            service = VertexAIService(project_id="my-project")
            itinerary = service.generate_itinerary(
                delay_minutes=180,
                arrival_terminal="4",
                buffer_minutes=45,
                flight_status="DELAYED",
                budget=150,
                preferences="museums, tapas, parks",
                duration_hours=6
            )
        """
        # Get prompt configuration
        prompt_data = self.prompt_manager.format_prompt(
            'itinerary_gen',
            {
                'delay_minutes': delay_minutes,
                'arrival_terminal': arrival_terminal,
                'buffer_minutes': buffer_minutes,
                'flight_status': flight_status,
                'budget': budget,
                'preferences': preferences,
                'duration_hours': duration_hours
            }
        )
        
        logger.info(f"Generating itinerary: budget={budget}, duration={duration_hours}h")
        
        try:
            gen_config = GenerateContentConfig(
                temperature=prompt_data['parameters'].get('temperature', 0.7),
                max_output_tokens=prompt_data['parameters'].get('max_output_tokens', 1024)
            )
            
            response = self.client.models.generate_content(
                model=prompt_data['model_name'],
                contents=prompt_data['prompt'],
                config=gen_config
            )
            
            # Extract text response
            result_text = response.text
            
            logger.info("Successfully generated itinerary")
            
            return {
                "success": True,
                "itinerary": result_text,
                "model": prompt_data['model_name'],
                "parameters": prompt_data['parameters'],
                "metadata": prompt_data.get('metadata', {})
            }
        
        except Exception as e:
            logger.error(f"Error generating itinerary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "itinerary": None
            }
    
    def generate_status_update(
        self,
        airline_code: str,
        flight_number: str,
        departure_airport: str,
        arrival_airport: str,
        scheduled_departure: str,
        scheduled_arrival: str,
        departure_status: str,
        arrival_status: str,
        estimated_arrival: Optional[str],
        time_status: str,
        flight_status: str,
        next_flight: str,
        next_flight_time: str,
        buffer_minutes: int,
        arrival_terminal: str,
        arrival_gate: str,
        risk_level: str,
        risk_factors: list
    ) -> Dict[str, Any]:
        """
        Generate passenger-friendly status update message
        
        Args:
            Various flight status parameters
        
        Returns:
            Dictionary with status message
        """
        # Get prompt configuration
        prompt_data = self.prompt_manager.format_prompt(
            'status_update',
            {
                'airline_code': airline_code,
                'flight_number': flight_number,
                'departure_airport': departure_airport,
                'arrival_airport': arrival_airport,
                'scheduled_departure': scheduled_departure,
                'scheduled_arrival': scheduled_arrival,
                'departure_status': departure_status,
                'arrival_status': arrival_status,
                'estimated_arrival': estimated_arrival or 'N/A',
                'time_status': time_status,
                'flight_status': flight_status,
                'next_flight': next_flight,
                'next_flight_time': next_flight_time,
                'buffer_minutes': buffer_minutes,
                'arrival_terminal': arrival_terminal,
                'arrival_gate': arrival_gate,
                'risk_level': risk_level,
                'risk_factors': ', '.join(risk_factors) if risk_factors else 'None'
            }
        )
        
        logger.info(f"Generating status update for {airline_code}{flight_number}")
        
        try:
            gen_config = GenerateContentConfig(
                temperature=prompt_data['parameters'].get('temperature', 0.7),
                max_output_tokens=prompt_data['parameters'].get('max_output_tokens', 1024)
            )
            
            response = self.client.models.generate_content(
                model=prompt_data['model_name'],
                contents=prompt_data['prompt'],
                config=gen_config
            )
            
            result_text = response.text
            
            logger.info("Successfully generated status update")
            
            return {
                "success": True,
                "message": result_text,
                "model": prompt_data['model_name']
            }
        
        except Exception as e:
            logger.error(f"Error generating status update: {str(e)}")
            # Return fallback message
            return {
                "success": False,
                "message": f"Flight {airline_code}{flight_number} status: {time_status}. "
                          f"Connection buffer: {buffer_minutes} minutes. Risk: {risk_level}.",
                "error": str(e)
            }
    
    def generate_custom_content(
        self,
        prompt: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate custom content with specified parameters
        
        Args:
            prompt: Text prompt
            model_name: Model to use (default: gemini-2.5-flash)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
        
        Returns:
            Generated text
        """
        try:
            gen_config = GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=gen_config
            )
            
            return response.text
        
        except Exception as e:
            logger.error(f"Error in custom generation: {str(e)}")
            raise


# Singleton instance
_service_instance: Optional[VertexAIService] = None


def get_vertex_ai_service(
    api_key: Optional[str] = None
) -> Optional[VertexAIService]:
    """
    Get singleton Google AI service instance
    
    Args:
        api_key: Google Gemini API key (uses GOOGLE_GEMINI_API_KEY env var if not provided)
    
    Returns:
        VertexAIService instance or None if API key not configured
    """
    global _service_instance
    
    if _service_instance is None or api_key:
        if not api_key:
            api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_GEMINI_API_KEY not configured - AI features disabled")
                return None
        
        _service_instance = VertexAIService(api_key=api_key)
    
    return _service_instance


if __name__ == "__main__":
    # Test Google AI service
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🧪 Testing Google AI Service (with Maps)\n")
    print("=" * 60)
    
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    
    if not api_key:
        print("⚠️  GOOGLE_GEMINI_API_KEY not configured. Skipping test.")
        print("   Set GOOGLE_GEMINI_API_KEY in .env to test Google AI integration.")
    else:
        try:
            service = get_vertex_ai_service()
            
            print(f"✅ Service initialized with API key")
            print(f"   Default coordinates: {AIRPORT_COORDINATES['DEFAULT']}")
            print(f"   Airports in database: {len(AIRPORT_COORDINATES)}")
            
            # Test simple generation
            print("\n🔍 Testing Madrid venue recommendation...")
            result = service.generate_custom_content(
                "Recommend the best tapas bar near Retiro Park in Madrid.",
                model_name="gemini-2.5-flash",
                max_tokens=200
            )
            
            print(f"Response: {result[:200]}...")
            print("\n✅ Google AI service with Maps working successfully!")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("   Make sure GOOGLE_GEMINI_API_KEY is valid.")

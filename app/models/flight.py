"""
Flight data models - Pydantic schemas for Lufthansa API responses
Handles strict parsing and validation of flight status data
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class TimeStatusCode(str, Enum):
    """Flight time status codes from Lufthansa API"""
    ON_TIME = "OT"
    DELAYED = "DL"
    EARLY = "FE"
    CANCELLED = "CD"
    NO_STATUS = "NA"


class FlightStatusCode(str, Enum):
    """Flight operational status codes"""
    NORMAL = "NA"
    DELAYED = "DL"
    HEAVY_DELAY = "HD"
    CANCELLED = "CD"
    NO_INFO = "NI"


class TimeInfo(BaseModel):
    """Nested time information from Lufthansa API"""
    scheduled: Optional[str] = None
    estimated: Optional[str] = None
    actual: Optional[str] = None
    status_code: Optional[str] = None
    status_definition: Optional[str] = None


class LocationInfo(BaseModel):
    """Airport location with terminal/gate details"""
    airport_code: str
    terminal: Optional[str] = None
    gate: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "airport_code": "MAD",
                "terminal": "4",
                "gate": "A23"
            }
        }


class FlightLeg(BaseModel):
    """
    Flattened flight leg representation
    Extracts essential fields from nested Lufthansa JSON structures
    
    Key Fields:
    - Handles both single flight objects and lists
    - Extracts scheduled, estimated, and actual times
    - Captures terminal and gate information
    - Tracks flight status codes for risk assessment
    """
    
    # Flight identification
    airline_code: str = Field(..., description="2-letter IATA airline code")
    flight_number: str = Field(..., description="Flight number without airline prefix")
    
    # Departure information
    departure_airport: str = Field(..., description="3-letter IATA departure airport code")
    departure_scheduled: Optional[str] = Field(None, description="Scheduled departure time (ISO format)")
    departure_estimated: Optional[str] = Field(None, description="Estimated departure time (ISO format)")
    departure_actual: Optional[str] = Field(None, description="Actual departure time (ISO format)")
    departure_terminal: Optional[str] = Field(None, description="Departure terminal")
    departure_gate: Optional[str] = Field(None, description="Departure gate")
    departure_status_code: Optional[str] = Field(None, description="Departure time status code")
    
    # Arrival information
    arrival_airport: str = Field(..., description="3-letter IATA arrival airport code")
    arrival_scheduled: Optional[str] = Field(None, description="Scheduled arrival time (ISO format)")
    arrival_estimated: Optional[str] = Field(None, description="Estimated arrival time (ISO format)")
    arrival_actual: Optional[str] = Field(None, description="Actual arrival time (ISO format)")
    arrival_terminal: Optional[str] = Field(None, description="Arrival terminal")
    arrival_gate: Optional[str] = Field(None, description="Arrival gate")
    arrival_status_code: Optional[str] = Field(None, description="Arrival time status code")
    
    # Status codes for risk analysis
    time_status_code: Optional[str] = Field(None, description="Overall time status: OT, DL, FE, CD, NA")
    time_status_definition: Optional[str] = Field(None, description="Human-readable status definition")
    flight_status_code: Optional[str] = Field(None, description="Flight status: NA, DL, HD, CD, NI")
    flight_status_definition: Optional[str] = Field(None, description="Human-readable flight status")
    
    # Aircraft information
    aircraft_code: Optional[str] = Field(None, description="Aircraft type code")
    aircraft_registration: Optional[str] = Field(None, description="Aircraft registration number")
    
    @field_validator('departure_scheduled', 'departure_estimated', 'departure_actual',
                     'arrival_scheduled', 'arrival_estimated', 'arrival_actual', mode='before')
    @classmethod
    def validate_datetime_format(cls, v):
        """Ensure datetime strings are in valid ISO format"""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                # Try parsing to validate format
                datetime.fromisoformat(v.replace('Z', '+00:00'))
                return v
            except ValueError:
                return None
        return None
    
    def get_effective_arrival_time(self) -> Optional[str]:
        """Get the most accurate arrival time available (actual > estimated > scheduled)"""
        return self.arrival_actual or self.arrival_estimated or self.arrival_scheduled
    
    def get_effective_departure_time(self) -> Optional[str]:
        """Get the most accurate departure time available (actual > estimated > scheduled)"""
        return self.departure_actual or self.departure_estimated or self.departure_scheduled
    
    def is_delayed(self) -> bool:
        """Check if flight is delayed based on status codes"""
        delay_codes = ["DL", "HD"]
        return (
            self.time_status_code in delay_codes or
            self.flight_status_code in delay_codes
        )
    
    def is_cancelled(self) -> bool:
        """Check if flight is cancelled"""
        return self.flight_status_code == "CD"
    
    class Config:
        json_schema_extra = {
            "example": {
                "airline_code": "IB",
                "flight_number": "778",
                "departure_airport": "HAM",
                "departure_scheduled": "2026-03-01T07:00:00",
                "arrival_airport": "MAD",
                "arrival_scheduled": "2026-03-01T10:00:00",
                "arrival_terminal": "4",
                "arrival_gate": "A23",
                "time_status_code": "OT",
                "time_status_definition": "Flight On Time"
            }
        }


class ConnectionRisk(BaseModel):
    """
    Risk analysis for flight connections
    Determines if passenger is at risk of missing connecting flight
    """
    
    is_at_risk: bool = Field(
        ...,
        description="True if connection time < 60 minutes or flight delayed/cancelled"
    )
    buffer_minutes: int = Field(
        ...,
        description="Minutes between arrival and next departure"
    )
    risk_level: str = Field(
        ...,
        description="Risk severity: LOW, MEDIUM, HIGH, CRITICAL"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="List of factors contributing to risk"
    )
    arrival_time: Optional[str] = Field(None, description="Effective arrival time used in calculation")
    next_departure_time: Optional[str] = Field(None, description="Next flight departure time")
    
    # Additional context
    recommended_action: Optional[str] = Field(None, description="Recommended action for passenger")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_at_risk": True,
                "buffer_minutes": 45,
                "risk_level": "HIGH",
                "risk_factors": ["SHORT_CONNECTION_45min", "DELAYED"],
                "arrival_time": "2026-03-01T10:45:00",
                "next_departure_time": "2026-03-01T11:30:00",
                "recommended_action": "Contact airline for rebooking options"
            }
        }


class ParsedFlightResponse(BaseModel):
    """
    Complete parsed flight status response
    Wrapper for list of flight legs with metadata
    """
    
    flights: List[FlightLeg] = Field(
        default_factory=list,
        description="List of parsed flight legs"
    )
    total_count: int = Field(..., description="Total number of flights in response")
    query_type: str = Field(
        ...,
        description="Type of query: status, route, arrivals, departures, customer_info"
    )
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp of query"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "flights": [],
                "total_count": 2,
                "query_type": "route",
                "timestamp": "2026-03-01T12:00:00Z"
            }
        }


class LufthansaRawFlight(BaseModel):
    """
    Raw Lufthansa API response structure
    Used for initial validation before parsing
    """
    Departure: Dict[str, Any] = Field(default_factory=dict)
    Arrival: Dict[str, Any] = Field(default_factory=dict)
    MarketingCarrier: Dict[str, Any] = Field(default_factory=dict)
    OperatingCarrier: Optional[Dict[str, Any]] = None
    Equipment: Optional[Dict[str, Any]] = None
    FlightStatus: Optional[Dict[str, Any]] = None
    ServiceType: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields from API


def parse_lufthansa_response(
    raw_json: Dict[str, Any],
    query_type: str = "status"
) -> ParsedFlightResponse:
    """
    Parse Lufthansa Flight Status API response into structured FlightLeg objects
    
    Handles both single flight objects and lists of flights.
    Extracts essential fields from deeply nested JSON structures.
    
    Args:
        raw_json: Raw JSON response from Lufthansa API
        query_type: Type of query (status, route, arrivals, departures, customer_info)
    
    Returns:
        ParsedFlightResponse with list of FlightLeg objects
    
    Raises:
        ValueError: If response structure is invalid
    """
    try:
        resource = raw_json.get('FlightStatusResource', {})
        flights_data = resource.get('Flights', {}).get('Flight', [])
        
        # Handle both single flight and list of flights
        if not isinstance(flights_data, list):
            flights_data = [flights_data] if flights_data else []
        
        parsed_flights = []
        
        for flight in flights_data:
            # Extract nested data
            departure = flight.get('Departure', {})
            arrival = flight.get('Arrival', {})
            marketing_carrier = flight.get('MarketingCarrier', {})
            operating_carrier = flight.get('OperatingCarrier', {})
            equipment = flight.get('Equipment', {})
            flight_status = flight.get('FlightStatus', {})
            
            # Use operating carrier if available, otherwise marketing carrier
            carrier = operating_carrier if operating_carrier else marketing_carrier
            
            # Build FlightLeg
            flight_leg = FlightLeg(
                # Identification
                airline_code=carrier.get('AirlineID', 'N/A'),
                flight_number=carrier.get('FlightNumber', 'N/A'),
                
                # Departure
                departure_airport=departure.get('AirportCode', 'N/A'),
                departure_scheduled=departure.get('ScheduledTimeLocal', {}).get('DateTime'),
                departure_estimated=departure.get('EstimatedTimeLocal', {}).get('DateTime'),
                departure_actual=departure.get('ActualTimeLocal', {}).get('DateTime'),
                departure_terminal=departure.get('Terminal', {}).get('Name'),
                departure_gate=departure.get('Terminal', {}).get('Gate'),
                departure_status_code=departure.get('TimeStatus', {}).get('Code'),
                
                # Arrival
                arrival_airport=arrival.get('AirportCode', 'N/A'),
                arrival_scheduled=arrival.get('ScheduledTimeLocal', {}).get('DateTime'),
                arrival_estimated=arrival.get('EstimatedTimeLocal', {}).get('DateTime'),
                arrival_actual=arrival.get('ActualTimeLocal', {}).get('DateTime'),
                arrival_terminal=arrival.get('Terminal', {}).get('Name'),
                arrival_gate=arrival.get('Terminal', {}).get('Gate'),
                arrival_status_code=arrival.get('TimeStatus', {}).get('Code'),
                
                # Status
                time_status_code=arrival.get('TimeStatus', {}).get('Code'),
                time_status_definition=arrival.get('TimeStatus', {}).get('Definition'),
                flight_status_code=flight_status.get('Code'),
                flight_status_definition=flight_status.get('Definition'),
                
                # Aircraft
                aircraft_code=equipment.get('AircraftCode'),
                aircraft_registration=equipment.get('AircraftRegistration')
            )
            
            parsed_flights.append(flight_leg)
        
        # Get total count from metadata
        meta = resource.get('Meta', {})
        total_count = meta.get('TotalCount', len(parsed_flights))
        
        return ParsedFlightResponse(
            flights=parsed_flights,
            total_count=total_count,
            query_type=query_type
        )
        
    except Exception as e:
        # Return empty result on error with error indication
        return ParsedFlightResponse(
            flights=[],
            total_count=0,
            query_type=f"{query_type}_ERROR: {str(e)}"
        )


def format_flight_summary(flight: FlightLeg) -> str:
    """
    Generate a concise, human-readable flight summary
    
    Args:
        flight: FlightLeg object
        
    Returns:
        Formatted string with key flight information
    """
    status_emoji = "✅" if flight.time_status_code == "OT" else "⚠️"
    
    return (
        f"{status_emoji} {flight.airline_code}{flight.flight_number}: "
        f"{flight.departure_airport} → {flight.arrival_airport} | "
        f"Dep: {flight.departure_scheduled or 'N/A'} | "
        f"Arr: {flight.arrival_scheduled or 'N/A'} | "
        f"Status: {flight.time_status_definition or 'N/A'} | "
        f"Terminal: {flight.arrival_terminal or 'N/A'}, "
        f"Gate: {flight.arrival_gate or 'TBD'}"
    )

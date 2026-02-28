"""
Request models - Pydantic schemas for user input validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class DelayScenario(BaseModel):
    """
    Basic delay scenario input
    Legacy model for simple itinerary generation
    """
    delay_location: str = Field(
        default="Madrid",
        description="City where passenger is delayed"
    )
    last_destination: str = Field(
        default="Bogota",
        description="Final destination of passenger"
    )
    user_input: str = Field(
        ...,
        description="Free-form user input with preferences and context"
    )
    default_info_about_passenger: bool = Field(
        default=False,
        description="Whether to use default passenger profile"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "delay_location": "Madrid",
                "last_destination": "Bogota",
                "user_input": "I like museums and tapas. Budget is 150 EUR.",
                "default_info_about_passenger": False
            }
        }


class FlightRiskRequest(BaseModel):
    """
    Comprehensive flight risk analysis request
    Works for any flight connection worldwide
    """
    
    # Origin flight information
    origin_flight_number: str = Field(
        ...,
        description="Origin flight number (e.g., LH1114, BA456)",
        min_length=4,
        max_length=10
    )
    origin_departure_date: str = Field(
        ...,
        description="Origin flight departure date in format YYYY-MM-DD"
    )
    
    # Connection flight information
    connection_flight_number: str = Field(
        ...,
        description="Connecting flight number"
    )
    connection_departure_time: str = Field(
        ...,
        description="Connection departure time in ISO format (YYYY-MM-DDTHH:MM:SS)"
    )
    
    # Optional: Specify layover city for itinerary generation
    layover_city: Optional[str] = Field(
        None,
        description="Layover city name (auto-detected from flight data if not provided)"
    )
    
    # Passenger preferences
    budget: int = Field(
        default=100,
        ge=0,
        le=10000,
        description="Budget in local currency for recovery plan"
    )
    preferences: str = Field(
        default="sightseeing, local food",
        description="Comma-separated activity preferences for layover itinerary"
    )
    
    # Optional passenger info
    passenger_name: Optional[str] = Field(None, description="Passenger name for personalization")
    contact_email: Optional[str] = Field(None, description="Email for notifications")
    
    @field_validator('origin_departure_date')
    @classmethod
    def validate_origin_date(cls, v):
        """Validate departure date format"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError(f"Invalid origin_departure_date format, use YYYY-MM-DD")
    
    @field_validator('connection_departure_time')
    @classmethod
    def validate_connection_time(cls, v):
        """Validate departure time format"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError(f"Invalid connection_departure_time format, use ISO format")
    
    class Config:
        json_schema_extra = {
            "example": {
                "origin_flight_number": "LH1114",
                "origin_departure_date": "2026-03-01",
                "connection_flight_number": "IB6312",
                "connection_departure_time": "2026-03-01T15:30:00",
                "layover_city": "Madrid",
                "budget": 150,
                "preferences": "museums, local cuisine, architecture",
                "passenger_name": "John Doe",
                "contact_email": "john@example.com"
            }
        }


class ItineraryRequest(BaseModel):
    """
    Request for generating layover city itinerary
    Used when connection risk is confirmed
    """
    
    location: str = Field(..., description="Layover city name (e.g., Madrid, Paris, Dubai)")
    duration_hours: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Duration of itinerary in hours"
    )
    budget: int = Field(
        default=100,
        ge=0,
        le=10000,
        description="Budget in local currency"
    )
    preferences: str = Field(
        default="sightseeing, food",
        description="Activity preferences"
    )
    
    # Contextual information
    delay_minutes: int = Field(
        default=0,
        description="Flight delay duration for context"
    )
    arrival_terminal: Optional[str] = Field(
        None,
        description="Arrival terminal for transport planning"
    )
    next_flight_time: Optional[str] = Field(
        None,
        description="Next flight time to plan return"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": "Paris",
                "duration_hours": 6,
                "budget": 150,
                "preferences": "museums, cafes, architecture",
                "delay_minutes": 180,
                "arrival_terminal": "2E",
                "next_flight_time": "2026-03-02T10:00:00"
            }
        }


class RouteAnalysisRequest(BaseModel):
    """
    Request to analyze multi-leg journey
    Supports any origin-destination pair
    """
    
    departure_date: str = Field(
        ...,
        description="Initial departure date (YYYY-MM-DD)"
    )
    passenger_profile: Optional[dict] = Field(
        None,
        description="Optional passenger profile data"
    )
    
    @field_validator('departure_date')
    @classmethod
    def validate_departure_date(cls, v):
        """Ensure departure date is valid and not in the past"""
        try:
            dep_date = datetime.strptime(v, '%Y-%m-%d')
            if dep_date.date() < datetime.now().date():
                raise ValueError("Departure date cannot be in the past")
            return v
        except ValueError as e:
            raise ValueError(f"Invalid departure_date: {str(e)}")
    
    class Config:
        json_schema_extra = {
            "example": {
                "departure_date": "2026-03-15",
                "passenger_profile": {
                    "budget": 200,
                    "preferences": "culture, food"
                }
            }
        }

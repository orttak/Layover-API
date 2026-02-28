"""
Models package - Pydantic schemas for data validation
"""

from .flight import (
    FlightLeg,
    ParsedFlightResponse,
    ConnectionRisk,
    TimeStatusCode,
    FlightStatusCode
)
from .request import (
    DelayScenario,
    FlightRiskRequest,
    ItineraryRequest
)

__all__ = [
    "FlightLeg",
    "ParsedFlightResponse",
    "ConnectionRisk",
    "TimeStatusCode",
    "FlightStatusCode",
    "DelayScenario",
    "FlightRiskRequest",
    "ItineraryRequest"
]

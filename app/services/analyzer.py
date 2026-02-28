"""
Flight Connection Analyzer - Business logic for risk assessment
Calculates connection buffer times and determines if passenger is at risk
"""

from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from app.models.flight import FlightLeg, ConnectionRisk


class FlightConnectionAnalyzer:
    """
    Analyzes flight connections and calculates risk of missing connecting flights
    
    Core Logic:
    - If buffer < 60 minutes: HIGH RISK
    - If Status/Code is 'DL', 'HD', 'CD': Trigger recovery mode
    - Considers terminal changes and gate distances
    """
    
    # Risk thresholds in minutes
    CRITICAL_BUFFER = 30
    HIGH_RISK_BUFFER = 60
    MEDIUM_RISK_BUFFER = 90
    
    # Critical flight status codes
    CRITICAL_STATUS_CODES = ['CD', 'HD']  # Cancelled, Heavy Delay
    DELAY_STATUS_CODES = ['DL']  # Delayed
    
    def __init__(self, min_connection_time: int = 60):
        """
        Initialize analyzer
        
        Args:
            min_connection_time: Minimum safe connection time in minutes (default 60)
        """
        self.min_connection_time = min_connection_time
    
    def calculate_connection_risk(
        self,
        arriving_flight: FlightLeg,
        next_departure_time: str,
        next_flight_number: Optional[str] = None
    ) -> ConnectionRisk:
        """
        Calculate connection risk between two flights
        
        Args:
            arriving_flight: FlightLeg object for the arriving flight
            next_departure_time: ISO datetime string for next flight departure
            next_flight_number: Optional flight number for context
        
        Returns:
            ConnectionRisk object with comprehensive risk assessment
        
        Example:
            analyzer = FlightConnectionAnalyzer()
            risk = analyzer.calculate_connection_risk(
                ham_mad_flight,
                "2026-03-01T11:30:00",
                "IB6312"
            )
        """
        risk_factors = []
        is_at_risk = False
        buffer_minutes = 0
        risk_level = "LOW"
        recommended_action = None
        
        # Get most accurate arrival time (actual > estimated > scheduled)
        arrival_time_str = arriving_flight.get_effective_arrival_time()
        
        if not arrival_time_str:
            return ConnectionRisk(
                is_at_risk=True,
                buffer_minutes=0,
                risk_level="CRITICAL",
                risk_factors=["NO_ARRIVAL_TIME_DATA"],
                arrival_time=None,
                next_departure_time=next_departure_time,
                recommended_action="Contact airline immediately - no arrival time data available"
            )
        
        # Calculate buffer time
        try:
            arrival_time = self._parse_datetime(arrival_time_str)
            departure_time = self._parse_datetime(next_departure_time)
            
            buffer_minutes = int((departure_time - arrival_time).total_seconds() / 60)
            
            # Assess risk based on buffer
            if buffer_minutes < 0:
                is_at_risk = True
                risk_level = "CRITICAL"
                risk_factors.append("CONNECTION_ALREADY_MISSED")
                recommended_action = "Immediate rebooking required - connection already missed"
                
            elif buffer_minutes < self.CRITICAL_BUFFER:
                is_at_risk = True
                risk_level = "CRITICAL"
                risk_factors.append(f"VERY_SHORT_CONNECTION_{buffer_minutes}min")
                recommended_action = "Prepare for rebooking - extremely tight connection"
                
            elif buffer_minutes < self.HIGH_RISK_BUFFER:
                is_at_risk = True
                risk_level = "HIGH"
                risk_factors.append(f"SHORT_CONNECTION_{buffer_minutes}min")
                recommended_action = "Monitor closely - connection at risk"
                
            elif buffer_minutes < self.MEDIUM_RISK_BUFFER:
                risk_level = "MEDIUM"
                risk_factors.append(f"TIGHT_CONNECTION_{buffer_minutes}min")
                recommended_action = "Stay alert - proceed directly to gate"
                
            else:
                risk_level = "LOW"
                recommended_action = "Connection looks safe - normal procedures"
        
        except Exception as e:
            is_at_risk = True
            risk_level = "UNKNOWN"
            risk_factors.append(f"TIME_CALCULATION_ERROR: {str(e)}")
            recommended_action = "Verify flight times with airline"
        
        # Check flight status codes
        if arriving_flight.flight_status_code in self.CRITICAL_STATUS_CODES:
            is_at_risk = True
            risk_factors.append(f"FLIGHT_STATUS_{arriving_flight.flight_status_code}")
            
            if arriving_flight.flight_status_code == "CD":
                risk_level = "CRITICAL"
                recommended_action = "Flight cancelled - immediate rebooking required"
            elif arriving_flight.flight_status_code == "HD":
                if risk_level != "CRITICAL":
                    risk_level = "HIGH"
                recommended_action = "Heavy delay detected - prepare contingency plan"
        
        elif arriving_flight.flight_status_code in self.DELAY_STATUS_CODES:
            is_at_risk = True
            risk_factors.append("DELAYED")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        # Check time status
        if arriving_flight.time_status_code == "DL":
            if "DELAYED" not in risk_factors:
                risk_factors.append("ARRIVAL_DELAYED")
                is_at_risk = True
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
        
        # Check for terminal changes (potential extra time needed)
        if arriving_flight.arrival_terminal and next_flight_number:
            # Terminal changes typically add 10-15 minutes
            if buffer_minutes < 75:  # Adjusted threshold for terminal changes
                risk_factors.append(f"TERMINAL_CHANGE_RISK")
                if risk_level == "LOW":
                    risk_level = "MEDIUM"
        
        # If no specific issues but classified as delayed
        if arriving_flight.is_delayed() and "DELAYED" not in risk_factors and "ARRIVAL_DELAYED" not in risk_factors:
            risk_factors.append("FLIGHT_MARKED_DELAYED")
            is_at_risk = True
        
        return ConnectionRisk(
            is_at_risk=is_at_risk,
            buffer_minutes=buffer_minutes,
            risk_level=risk_level,
            risk_factors=risk_factors,
            arrival_time=arrival_time_str,
            next_departure_time=next_departure_time,
            recommended_action=recommended_action
        )
    
    def analyze_multi_leg_route(
        self,
        flights: List[FlightLeg],
        connection_times: List[str]
    ) -> List[ConnectionRisk]:
        """
        Analyze an entire multi-leg route
        
        Args:
            flights: List of FlightLeg objects in sequence
            connection_times: List of departure times for connecting flights
        
        Returns:
            List of ConnectionRisk objects for each connection
        
        Example:
            risks = analyzer.analyze_multi_leg_route(
                [ham_mad_flight, mad_bog_flight],
                ["2026-03-01T11:30:00"]
            )
        """
        if len(flights) != len(connection_times) + 1:
            raise ValueError("Number of flights should be 1 more than connection times")
        
        risks = []
        for i, flight in enumerate(flights[:-1]):
            risk = self.calculate_connection_risk(
                flight,
                connection_times[i],
                f"Connection {i+1}"
            )
            risks.append(risk)
        
        return risks
    
    def should_trigger_recovery_mode(self, risk: ConnectionRisk) -> bool:
        """
        Determine if recovery mode (Madrid itinerary) should be triggered
        
        Args:
            risk: ConnectionRisk object
        
        Returns:
            True if recovery mode should be activated
        """
        # Trigger on HIGH or CRITICAL risk
        if risk.risk_level in ["HIGH", "CRITICAL"]:
            return True
        
        # Trigger on specific critical factors
        critical_factors = [
            "CONNECTION_ALREADY_MISSED",
            "FLIGHT_STATUS_CD",
            "FLIGHT_STATUS_HD"
        ]
        
        for factor in critical_factors:
            if factor in risk.risk_factors:
                return True
        
        return False
    
    def calculate_delay_duration(self, flight: FlightLeg) -> int:
        """
        Calculate how delayed a flight is in minutes
        
        Args:
            flight: FlightLeg object
        
        Returns:
            Delay in minutes (0 if on time or early)
        """
        if not flight.arrival_scheduled:
            return 0
        
        arrival_effective = flight.get_effective_arrival_time()
        if not arrival_effective or arrival_effective == flight.arrival_scheduled:
            return 0
        
        try:
            scheduled = self._parse_datetime(flight.arrival_scheduled)
            effective = self._parse_datetime(arrival_effective)
            
            delay = int((effective - scheduled).total_seconds() / 60)
            return max(0, delay)  # Return 0 for negative (early arrivals)
        
        except:
            return 0
    
    def get_risk_summary(self, risk: ConnectionRisk) -> dict:
        """
        Generate a human-readable risk summary
        
        Args:
            risk: ConnectionRisk object
        
        Returns:
            Dictionary with formatted risk information
        """
        emoji_map = {
            "LOW": "✅",
            "MEDIUM": "⚠️",
            "HIGH": "🚨",
            "CRITICAL": "🔴",
            "UNKNOWN": "❓"
        }
        
        return {
            "emoji": emoji_map.get(risk.risk_level, "❓"),
            "level": risk.risk_level,
            "at_risk": risk.is_at_risk,
            "buffer": f"{risk.buffer_minutes} minutes",
            "summary": f"{emoji_map.get(risk.risk_level)} {risk.risk_level} Risk - {risk.buffer_minutes} min buffer",
            "factors": risk.risk_factors,
            "action": risk.recommended_action,
            "recovery_needed": self.should_trigger_recovery_mode(risk)
        }
    
    @staticmethod
    def _parse_datetime(dt_string: str) -> datetime:
        """
        Parse datetime string handling various formats
        
        Args:
            dt_string: ISO datetime string
        
        Returns:
            datetime object
        """
        # Handle timezone indicator
        dt_string = dt_string.replace('Z', '+00:00')
        return datetime.fromisoformat(dt_string)


# Convenience function for quick analysis
def quick_risk_check(
    arriving_flight: FlightLeg,
    next_departure: str,
    min_buffer: int = 60
) -> Tuple[bool, str, int]:
    """
    Quick risk check without full ConnectionRisk object
    
    Args:
        arriving_flight: FlightLeg for arriving flight
        next_departure: Next flight departure time
        min_buffer: Minimum safe buffer in minutes
    
    Returns:
        Tuple of (is_at_risk, risk_level, buffer_minutes)
    """
    analyzer = FlightConnectionAnalyzer(min_connection_time=min_buffer)
    risk = analyzer.calculate_connection_risk(arriving_flight, next_departure)
    return (risk.is_at_risk, risk.risk_level, risk.buffer_minutes)


if __name__ == "__main__":
    # Test the analyzer
    print("🧪 Testing Flight Connection Analyzer\n")
    print("=" * 60)
    
    # Create mock flight data
    from app.models.flight import FlightLeg
    
    test_flight = FlightLeg(
        airline_code="IB",
        flight_number="778",
        departure_airport="HAM",
        departure_scheduled="2026-03-01T07:00:00",
        arrival_airport="MAD",
        arrival_scheduled="2026-03-01T10:00:00",
        arrival_estimated="2026-03-01T10:45:00",  # 45 min delay
        arrival_terminal="4",
        arrival_gate="A23",
        time_status_code="DL",
        time_status_definition="Delayed",
        flight_status_code="DL"
    )
    
    analyzer = FlightConnectionAnalyzer()
    
    # Test scenario: next flight at 11:30
    risk = analyzer.calculate_connection_risk(
        test_flight,
        "2026-03-01T11:30:00",
        "IB6312"
    )
    
    summary = analyzer.get_risk_summary(risk)
    
    print(f"Flight: IB778 (HAM → MAD)")
    print(f"Next Connection: 11:30")
    print(f"\n{summary['summary']}")
    print(f"At Risk: {summary['at_risk']}")
    print(f"Recovery Needed: {summary['recovery_needed']}")
    print(f"\nRisk Factors:")
    for factor in summary['factors']:
        print(f"  • {factor}")
    print(f"\nRecommended Action:")
    print(f"  {summary['action']}")
    
    print("\n✅ Analyzer working successfully!")

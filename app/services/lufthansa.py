"""
Lufthansa API Client - Handles authentication and flight status queries
Wraps Lufthansa Flight Status API with clean interface
"""

import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from functools import lru_cache
import logging

from app.models.flight import ParsedFlightResponse, parse_lufthansa_response

logger = logging.getLogger(__name__)


class LufthansaAPIError(Exception):
    """Custom exception for Lufthansa API errors"""
    pass


class LufthansaClient:
    """
    Client for Lufthansa Flight Status API
    
    Features:
    - OAuth 2.0 authentication
    - Multiple query types (status, route, arrivals, departures)
    - Automatic token refresh
    - Response parsing with Pydantic models
    """
    
    BASE_URL = "https://api.lufthansa.com/v1"
    TOKEN_EXPIRY_BUFFER = 300  # Request new token 5 min before expiry
    
    def __init__(self, client_id: str, client_secret: str):
        """
        Initialize Lufthansa API client
        
        Args:
            client_id: Lufthansa OAuth client ID
            client_secret: Lufthansa OAuth client secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
    
    def _get_access_token(self, force_refresh: bool = False) -> str:
        """
        Get OAuth access token (cached with auto-refresh)
        
        Args:
            force_refresh: Force token refresh even if current token is valid
        
        Returns:
            Access token string
        
        Raises:
            LufthansaAPIError: If authentication fails
        """
        # Check if we have a valid token
        if (not force_refresh and 
            self._access_token and 
            self._token_expires_at and 
            datetime.now() < self._token_expires_at):
            return self._access_token
        
        # Request new token
        url = f"{self.BASE_URL}/oauth/token"
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials"
        }
        
        try:
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data.get("access_token")
            
            # Calculate expiry time (typically 3600 seconds)
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = datetime.now() + timedelta(
                seconds=expires_in - self.TOKEN_EXPIRY_BUFFER
            )
            
            logger.info(f"Obtained new Lufthansa access token, expires in {expires_in}s")
            return self._access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to obtain Lufthansa access token: {str(e)}")
            raise LufthansaAPIError(f"Authentication failed: {str(e)}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make authenticated request to Lufthansa API
        
        Args:
            endpoint: API endpoint path (without base URL)
            params: Optional query parameters
        
        Returns:
            JSON response as dictionary
        
        Raises:
            LufthansaAPIError: If request fails
        """
        token = self._get_access_token()
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Lufthansa API HTTP error: {e.response.status_code} - {e.response.text}")
            raise LufthansaAPIError(f"API request failed: {e.response.status_code} - {e.response.text}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Lufthansa API request error: {str(e)}")
            raise LufthansaAPIError(f"Request failed: {str(e)}")
    
    def get_flight_status(
        self,
        flight_number: str,
        date: str
    ) -> ParsedFlightResponse:
        """
        Get status of a specific flight
        
        Args:
            flight_number: Flight number (e.g., 'LH400', 'IB778')
            date: Date in YYYY-MM-DD format
        
        Returns:
            ParsedFlightResponse with flight details
        
        Example:
            client = LufthansaClient(client_id, client_secret)
            response = client.get_flight_status('IB778', '2026-03-01')
        """
        endpoint = f"operations/flightstatus/{flight_number}/{date}"
        
        logger.info(f"Querying flight status: {flight_number} on {date}")
        raw_data = self._make_request(endpoint)
        
        return parse_lufthansa_response(raw_data, query_type="status")
    
    def get_flight_status_by_route(
        self,
        origin: str,
        destination: str,
        date: str
    ) -> ParsedFlightResponse:
        """
        Get all flights on a specific route for a date
        
        Args:
            origin: 3-letter IATA origin airport code
            destination: 3-letter IATA destination airport code
            date: Date in YYYY-MM-DD format
        
        Returns:
            ParsedFlightResponse with list of flights on route
        
        Example:
            response = client.get_flight_status_by_route('HAM', 'MAD', '2026-03-01')
        """
        endpoint = f"operations/flightstatus/route/{origin}/{destination}/{date}"
        
        logger.info(f"Querying route: {origin} -> {destination} on {date}")
        raw_data = self._make_request(endpoint)
        
        return parse_lufthansa_response(raw_data, query_type="route")
    
    def get_arrivals(
        self,
        airport_code: str,
        from_datetime: str
    ) -> ParsedFlightResponse:
        """
        Get arriving flights at an airport
        
        Args:
            airport_code: 3-letter IATA airport code
            from_datetime: Start datetime in YYYY-MM-DDTHH:MM format
        
        Returns:
            ParsedFlightResponse with arriving flights
        
        Example:
            response = client.get_arrivals('MAD', '2026-03-01T14:00')
        """
        endpoint = f"operations/flightstatus/arrivals/{airport_code}/{from_datetime}"
        
        logger.info(f"Querying arrivals at {airport_code} from {from_datetime}")
        raw_data = self._make_request(endpoint)
        
        return parse_lufthansa_response(raw_data, query_type="arrivals")
    
    def get_departures(
        self,
        airport_code: str,
        from_datetime: str
    ) -> ParsedFlightResponse:
        """
        Get departing flights from an airport
        
        Args:
            airport_code: 3-letter IATA airport code
            from_datetime: Start datetime in YYYY-MM-DDTHH:MM format
        
        Returns:
            ParsedFlightResponse with departing flights
        
        Example:
            response = client.get_departures('MAD', '2026-03-01T14:00')
        """
        endpoint = f"operations/flightstatus/departures/{airport_code}/{from_datetime}"
        
        logger.info(f"Querying departures from {airport_code} from {from_datetime}")
        raw_data = self._make_request(endpoint)
        
        return parse_lufthansa_response(raw_data, query_type="departures")
    
    def get_customer_flight_info(
        self,
        flight_number: str,
        date: str
    ) -> ParsedFlightResponse:
        """
        Get detailed customer flight information (gates, terminals, etc.)
        
        Args:
            flight_number: Flight number
            date: Date in YYYY-MM-DD format
        
        Returns:
            ParsedFlightResponse with detailed flight info
        
        Example:
            response = client.get_customer_flight_info('LH400', '2026-03-01')
        """
        endpoint = f"operations/customerflightinformation/{flight_number}/{date}"
        
        logger.info(f"Querying customer flight info: {flight_number} on {date}")
        raw_data = self._make_request(endpoint)
        
        return parse_lufthansa_response(raw_data, query_type="customer_info")
    
    def health_check(self) -> bool:
        """
        Check if API is accessible and credentials are valid
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            self._get_access_token(force_refresh=True)
            return True
        except LufthansaAPIError:
            return False


# Singleton pattern for easy reuse
_client_instance: Optional[LufthansaClient] = None


def get_lufthansa_client(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None
) -> LufthansaClient:
    """
    Get singleton Lufthansa client instance
    
    Args:
        client_id: Optional client ID (uses Settings if not provided)
        client_secret: Optional client secret (uses Settings if not provided)
    
    Returns:
        LufthansaClient instance
    """
    global _client_instance
    
    if _client_instance is None or (client_id and client_secret):
        if not client_id or not client_secret:
            # Try to get from Settings (handles .env loading)
            from app.core.config import get_settings
            settings = get_settings()
            
            client_id = client_id or settings.lufthansa_client_id
            client_secret = client_secret or settings.lufthansa_client_secret
            
            if not client_id or not client_secret:
                raise ValueError("Lufthansa credentials not provided")
        
        _client_instance = LufthansaClient(client_id, client_secret)
    
    return _client_instance


if __name__ == "__main__":
    # Test the Lufthansa client
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🧪 Testing Lufthansa API Client\n")
    print("=" * 60)
    
    try:
        client = get_lufthansa_client()
        
        # Health check
        print("\n🏥 Health Check...")
        if client.health_check():
            print("✅ API is healthy")
        else:
            print("❌ API health check failed")
            exit(1)
        
        # Test route query
        print("\n🔍 Testing route query (HAM -> MAD)...")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        response = client.get_flight_status_by_route("HAM", "MAD", tomorrow)
        
        print(f"✅ Found {response.total_count} flight(s)")
        for flight in response.flights[:2]:  # Show first 2
            print(f"  • {flight.airline_code}{flight.flight_number}: "
                  f"{flight.departure_airport} → {flight.arrival_airport} "
                  f"({flight.time_status_definition})")
        
        print("\n✅ Lufthansa client working successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

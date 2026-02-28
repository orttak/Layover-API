import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from models import (
    parse_lufthansa_response,
    calculate_connection_risk,
    format_flight_summary,
    ParsedFlightResponse,
    FlightLeg,
    ConnectionRisk
)

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("LUFTHANSA_CLIENT_ID")
CLIENT_SECRET = os.getenv("LUFTHANSA_CLIENT_SECRET")
BASE_URL = "https://api.lufthansa.com/v1"

def get_access_token():
    """Get OAuth token from Lufthansa API"""
    url = f"{BASE_URL}/oauth/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials"
    }
    
    response = requests.post(url, data=data)
    if response.status_code == 200:
        token = response.json().get("access_token")
        print(f"✅ Access Token obtained: {token[:20]}...")
        return token
    else:
        print(f"❌ Token Error: {response.status_code} - {response.text}")
        return None

def get_flight_status(token, flight_number, date):
    """Query 1: Specific flight status by flight number and date"""
    url = f"{BASE_URL}/operations/flightstatus/{flight_number}/{date}"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"\n📍 Query 1: Flight Status for {flight_number} on {date}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        parsed = parse_lufthansa_response(data, query_type="status")
        
        print(f"✅ Status: {response.status_code}")
        print(f"✈️  Found {parsed.total_count} flight(s)\n")
        
        for flight in parsed.flights:
            print(format_flight_summary(flight))
        
        return parsed
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def get_flight_status_by_route(token, origin, destination, date):
    """Query 2: Flight status by route (HAM -> MAD)"""
    url = f"{BASE_URL}/operations/flightstatus/route/{origin}/{destination}/{date}"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"\n📍 Query 2: Route Status {origin} -> {destination} on {date}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        parsed = parse_lufthansa_response(data, query_type="route")
        
        print(f"✅ Status: {response.status_code}")
        print(f"✈️  Found {parsed.total_count} flight(s)\n")
        
        for flight in parsed.flights:
            print(format_flight_summary(flight))
        
        return parsed
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def get_arrivals(token, airport_code, from_datetime):
    """Query 3: Arrivals at airport (checking Madrid arrivals)"""
    url = f"{BASE_URL}/operations/flightstatus/arrivals/{airport_code}/{from_datetime}"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"\n📍 Query 3: Arrivals at {airport_code} from {from_datetime}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        parsed = parse_lufthansa_response(data, query_type="arrivals")
        
        print(f"✅ Status: {response.status_code}")
        print(f"✈️  Found {parsed.total_count} arrival(s)\n")
        
        for flight in parsed.flights:
            print(format_flight_summary(flight))
        
        return parsed
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def get_departures(token, airport_code, from_datetime):
    """Query 4: Departures from airport (checking Madrid departures to Bogota)"""
    url = f"{BASE_URL}/operations/flightstatus/departures/{airport_code}/{from_datetime}"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"\n📍 Query 4: Departures from {airport_code} from {from_datetime}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        parsed = parse_lufthansa_response(data, query_type="departures")
        
        print(f"✅ Status: {response.status_code}")
        print(f"✈️  Found {parsed.total_count} departure(s)\n")
        
        for flight in parsed.flights:
            print(format_flight_summary(flight))
        
        return parsed
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def get_customer_flight_info(token, flight_number, date):
    """Query 5: Customer flight information (terminal, gate, etc.)"""
    url = f"{BASE_URL}/operations/customerflightinformation/{flight_number}/{date}"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"\n📍 Query 5: Customer Info for {flight_number} on {date}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        parsed = parse_lufthansa_response(data, query_type="customer_info")
        
        print(f"✅ Status: {response.status_code}")
        print(f"✈️  Found {parsed.total_count} flight(s)\n")
        
        for flight in parsed.flights:
            print(format_flight_summary(flight))
        
        return parsed
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


if __name__ == "__main__":
    print("=" * 80)
    print("🛫 LUFTHANSA API TEST - HAM → MAD → BOG Scenario (with Pydantic Models)")
    print("=" * 80)
    
    # Get OAuth token
    token = get_access_token()
    
    if not token:
        print("\n❌ Cannot proceed without access token")
        exit(1)
    
    # Date formats for Lufthansa API
    # Format: YYYY-MM-DD for date
    # Format: YYYY-MM-DDTHH:MM for datetime
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    
    date_str = tomorrow.strftime("%Y-%m-%d")  # Example: 2026-03-01
    datetime_str = tomorrow.strftime("%Y-%m-%dT14:00")  # Example: 2026-03-01T14:00
    
    print(f"\n📅 Test Date: {date_str}")
    print(f"🕐 Test DateTime: {datetime_str}")
    
    # Test all queries
    print("\n" + "=" * 80)
    print("Starting API Tests with Parsed Responses...")
    print("=" * 80)
    
    # 1. Check specific flight (real flight: IB774)
    print("\n🔍 Testing individual flight status...")
    flight_result = get_flight_status(token, "IB774", date_str)
    
    # 2. Check route Hamburg -> Madrid
    print("\n\n🔍 Testing route query (Hamburg → Madrid)...")
    route_result = get_flight_status_by_route(token, "HAM", "MAD", date_str)
    
    # 3. Check arrivals at Madrid airport
    print("\n\n🔍 Testing Madrid arrivals...")
    arrivals_result = get_arrivals(token, "MAD", datetime_str)
    
    # 4. Check departures from Madrid airport (for Bogota connection)
    print("\n\n🔍 Testing Madrid departures...")
    departures_result = get_departures(token, "MAD", datetime_str)
    
    # 5. Get detailed customer info for a flight
    print("\n\n🔍 Testing customer flight information...")
    customer_result = get_customer_flight_info(token, "IB778", date_str)
    
    # Risk Analysis Example
    print("\n\n" + "=" * 80)
    print("🚨 CONNECTION RISK ANALYSIS")
    print("=" * 80)
    
    if route_result and route_result.flights:
        # Use first Hamburg->Madrid flight
        ham_mad_flight = route_result.flights[0]
        
        # Assuming Bogota flight departs at 11:30 (just for testing)
        bogota_departure = f"{date_str}T11:30:00"
        
        risk = calculate_connection_risk(ham_mad_flight, bogota_departure)
        
        print(f"\n✈️  Analyzing: {ham_mad_flight.airline_code}{ham_mad_flight.flight_number}")
        print(f"   {ham_mad_flight.departure_airport} → {ham_mad_flight.arrival_airport}")
        print(f"\n📊 Risk Assessment:")
        print(f"   Risk Level: {risk.risk_level}")
        print(f"   At Risk: {'⚠️  YES' if risk.is_at_risk else '✅ NO'}")
        print(f"   Connection Buffer: {risk.buffer_minutes} minutes")
        print(f"   Madrid Arrival: {risk.arrival_time}")
        print(f"   Bogota Departure: {risk.next_departure_time}")
        
        if risk.risk_factors:
            print(f"\n⚠️  Risk Factors:")
            for factor in risk.risk_factors:
                print(f"      • {factor}")
        
        if risk.is_at_risk:
            print(f"\n🚨 RECOVERY MODE TRIGGERED!")
            print(f"   Recommendation: Generate Madrid itinerary & hotel options")
    
    print("\n" + "=" * 80)
    print("✅ All queries completed with structured Pydantic models!")
    print("=" * 80)


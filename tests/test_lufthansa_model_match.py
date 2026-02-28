"""
Basit test: Lufthansa API cevabı modellerimizle uyuyor mu?
"""
import os
from dotenv import load_dotenv
from app.services.lufthansa import LufthansaClient
from app.models.flight import ParsedFlightResponse, FlightLeg

# Load environment
load_dotenv()

def test_lufthansa_response_matches_model():
    """
    Test: IB774 için Lufthansa'dan gelen cevap Pydantic modellerine uyuyor mu?
    """
    # Lufthansa client oluştur
    client_id = os.getenv("LUFTHANSA_CLIENT_ID")
    client_secret = os.getenv("LUFTHANSA_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("⚠️  Lufthansa credentials yok, test atlanıyor")
        return
    
    client = LufthansaClient(client_id, client_secret)
    
    # IB774 için istek yap
    response = client.get_flight_status("IB774", "2026-03-01")
    
    # Tip kontrolü
    assert isinstance(response, ParsedFlightResponse), "Response ParsedFlightResponse değil!"
    assert response.total_count >= 0, "Total count negative olamaz!"
    assert response.query_type == "status", f"Query type yanlış: {response.query_type}"
    
    # Flight varsa kontrol et
    if response.flights:
        flight = response.flights[0]
        assert isinstance(flight, FlightLeg), "Flight FlightLeg değil!"
        assert flight.airline_code == "IB", f"Airline code yanlış: {flight.airline_code}"
        assert flight.flight_number == "774", f"Flight number yanlış: {flight.flight_number}"
        assert flight.departure_airport == "HAM", f"Departure yanlış: {flight.departure_airport}"
        assert flight.arrival_airport == "MAD", f"Arrival yanlış: {flight.arrival_airport}"
        
        print(f"✅ Model uyumu TAMAM!")
        print(f"   Flight: {flight.airline_code}{flight.flight_number}")
        print(f"   Route: {flight.departure_airport} → {flight.arrival_airport}")
        print(f"   Status: {flight.time_status_definition}")
    else:
        print("⚠️  Uçuş bulunamadı ama model uyumlu")
    
    print("✅ TEST PASSED - Modeller uyumlu!")

# Travel Delay Assistant API 🛫

**Global Flight Connection Risk Analyzer**

Real-time flight risk analysis using Lufthansa API + optional AI-powered recovery plans.

---

## Sistem Nasıl Çalışıyor

1. **Lufthansa API** → Uçuş durumu çekiliyor (gecikme, terminal, gate)
2. **Risk Analizi** → Aktarma süresi hesaplanıp risk seviyesi belirleniyor
3. **AI İtinerary** (opsiyonel) → Yüksek risk durumunda GCP Vertex AI devreye giriyor

---

## Kurulum

```bash
# .env dosyası oluştur
LUFTHANSA_CLIENT_ID=your_client_id
LUFTHANSA_CLIENT_SECRET=your_secret

# GCP (opsiyonel - sadece AI itinerary için)
GCP_PROJECT_ID=your-project-id

# Sunucuyu başlat
uvicorn app.main:app --reload --port 8000
```

---

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### 2. Flight Risk Analysis (Ana Endpoint)

```bash
curl -X POST http://localhost:8000/api/v1/analyze-flight-risk \
  -H "Content-Type: application/json" \
  -d '{
    "origin_flight_number": "LH400",
    "origin_departure_date": "2026-03-01",
    "connection_flight_number": "LH500",
    "connection_departure_time": "2026-03-01T16:00:00",
    "budget": 100,
    "preferences": "food, museums"
  }'
```

**Response:**

- `risk_analysis`: Risk seviyesi (LOW/MEDIUM/HIGH/CRITICAL), buffer süresi
- `flight_details`: Gerçek zamanlı uçuş bilgileri (gecikme, terminal, gate)
- `connection`: Aktarma detayları
- `recovery_plan`: AI önerisi (sadece GCP aktifse)

### 3. Flight Status Only

```bash
curl http://localhost:8000/api/v1/flight-status/LH400/2026-03-01
```

### 4. Simple Itinerary (Legacy)

```bash
curl -X POST http://localhost:8000/api/v1/generate-plan \
  -H "Content-Type: application/json" \
  -d '{
    "delay_location": "Madrid",
    "delay_minutes": 180,
    "last_destination": "Bogota",
    "passenger_budget": 100,
    "passenger_preferences": "food"
  }'
```

---

## Swagger Docs

http://localhost:8000/docs

---

## 📡 API Endpoints

### 1. **POST /api/v1/generate-plan** (Legacy)

Simple itinerary generation with Vertex AI.

**Request**:

```json
{
  "flight_number": "LH1114",
  "date": "2024-01-15",
  "delay_duration": 120,
  "budget": "medium"
}
```

### 2. **POST /api/v1/analyze-flight-risk** (Primary)

Full risk analysis with Lufthansa data + conditional recovery plan.

**Request**:

```json
{
  "origin_flight": {
    "flight_number": "LH1114",
    "date": "2024-01-15"
  },
  "connection_info": {
    "arrival_airport": "MAD",
    "departure_airport": "MAD",
    "connection_flight_number": "IB6312",
    "scheduled_connection_time": "2024-01-15T15:30:00"
  },
  "budget": "medium"
}
```

**Response**:

```json
{
  "status": "success",
  "flight_data": {...},
  "risk_analysis": {
    "risk_level": "HIGH",
    "buffer_minutes": 45,
    "recommendations": [...]
  },
  "recovery_plan": {
    "itinerary": "Visit Retiro Park, Prado Museum...",
    "estimated_cost": "€15-30"
  }
}
```

### 3. **POST /api/v1/flight-status**

Direct flight status query.

**Request**:

```json
{
  "flight_number": "LH1114",
  "date": "2024-01-15"
}
```

### 4. **GET /api/v1/route-status**

All flights between airports.

**Query Parameters**:

- `origin`: Airport code (e.g., HAM)
- `destination`: Airport code (e.g., MAD)
- `date`: YYYY-MM-DD

### 5. **GET /api/v1/health**

Service health check with Lufthansa API validation.

---

## 🧪 Testing

### Test with curl

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Risk analysis (Hamburg → Madrid → Bogota)
curl -X POST http://localhost:8000/api/v1/analyze-flight-risk \
  -H "Content-Type: application/json" \
  -d '{
    "origin_flight": {
      "flight_number": "LH1114",
      "date": "2024-01-15"
    },
    "connection_info": {
      "arrival_airport": "MAD",
      "departure_airport": "MAD",
      "connection_flight_number": "IB6312",
      "scheduled_connection_time": "2024-01-15T15:30:00"
    },
    "budget": "medium"
  }'
```

### Python Testing

```python
import requests

# Analyze flight risk
response = requests.post(
    "http://localhost:8000/api/v1/analyze-flight-risk",
    json={
        "origin_flight": {
            "flight_number": "LH1114",
            "date": "2024-01-15"
        },
        "connection_info": {
            "arrival_airport": "MAD",
            "departure_airport": "MAD",
            "connection_flight_number": "IB6312",
            "scheduled_connection_time": "2024-01-15T15:30:00"
        },
        "budget": "medium"
    }
)
print(response.json())
```

---

## 🐳 Deployment

### Docker Build

```bash
# Build image
docker build -t travel-assistant .

# Run container
docker run -p 8080:8080 --env-file .env travel-assistant

# Test
curl http://localhost:8080/api/v1/health
```

### Google Cloud Run

```bash
# Set variables
export PROJECT_ID=your-project-id
export REGION=us-central1
export SERVICE_NAME=travel-delay-assistant

# Build and deploy
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars "ENVIRONMENT=production" \
  --set-secrets "LUFTHANSA_CLIENT_ID=lufthansa-id:latest,LUFTHANSA_CLIENT_SECRET=lufthansa-secret:latest"

# Get URL
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
```

---

## 🔧 Configuration

### Environment Variables

| Variable                  | Required | Default     | Description                       |
| ------------------------- | -------- | ----------- | --------------------------------- |
| `LUFTHANSA_CLIENT_ID`     | ✅       | -           | Lufthansa API OAuth client ID     |
| `LUFTHANSA_CLIENT_SECRET` | ✅       | -           | Lufthansa API OAuth secret        |
| `GCP_PROJECT_ID`          | ✅       | -           | Google Cloud project ID           |
| `GCP_LOCATION`            | ❌       | us-central1 | Vertex AI region                  |
| `ENVIRONMENT`             | ❌       | development | Environment name                  |
| `DEBUG`                   | ❌       | false       | Enable debug mode                 |
| `LOG_LEVEL`               | ❌       | INFO        | Logging level                     |
| `MIN_CONNECTION_TIME`     | ❌       | 60          | Min connection time (minutes)     |
| `CRITICAL_BUFFER`         | ❌       | 30          | Critical risk threshold (minutes) |

### Prompt Configuration

Edit JSON files in `app/prompts/configs/`:

```json
{
  "system_role": "You are a travel assistant...",
  "user_template": "Generate itinerary for {city}...",
  "model_params": {
    "temperature": 0.7,
    "max_output_tokens": 2048
  }
}
```

---

## 📊 Business Logic

### Risk Level Calculation

```python
def calculate_risk(buffer_minutes: int, flight_status: str) -> str:
    if buffer_minutes < 30 or status in ["DL", "HD", "CD"]:
        return "CRITICAL"
    elif buffer_minutes < 60:
        return "HIGH"
    elif buffer_minutes < 90:
        return "MEDIUM"
    else:
        return "LOW"
```

### Recovery Mode Trigger

Recovery plan is generated when:

- Risk level is **HIGH** or **CRITICAL**
- Connection buffer < 60 minutes
- Flight status indicates delays (DL, HD, CD)

---

## 🛠️ Development

### Project Structure

- **models/**: Pydantic models for data validation and parsing
- **services/**: Business logic and external API clients
- **prompts/**: Prompt-as-Config JSON templates
- **core/**: Configuration and shared utilities
- **api/**: FastAPI route handlers

### Key Design Patterns

1. **Singleton Services**: Single instance for LufthansaClient, VertexAIService
2. **Dependency Injection**: Services injected via FastAPI `Depends()`
3. **Prompt-as-Config**: JSON-based prompt management with hot-reload
4. **Type Safety**: Comprehensive Pydantic models throughout

---

## 🐛 Troubleshooting

### Lufthansa API 404 Errors

```python
# Flights are only available for:
# - Yesterday to +5 days from today
# - Real flight numbers only (check route first)

# Solution: Query route to get valid flights
lufthansa.get_flight_status_by_route("HAM", "MAD", "2024-01-15")
```

### Import Errors

```bash
# Ensure pydantic-settings is installed
pip install pydantic-settings

# Verify Python 3.10+
python --version
```

### Vertex AI Errors

```bash
# Authenticate with GCP
gcloud auth application-default login

# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com
```

---

## 📝 License

Proprietary - Hackathon Project

---

## 👥 Contributors

- **Developer**: Travel Assistant Team
- **Created**: 2024
- **Route**: HAM → MAD → BOG Specialist

---

## 📚 API Documentation

Full interactive documentation available at:

- Swagger UI: `/docs`
- ReDoc: `/redoc`

---

**Built with FastAPI + Pydantic + Lufthansa API + Google Vertex AI (Gemini)** 🚀

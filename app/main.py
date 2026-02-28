"""
Travel Delay Assistant - Main FastAPI Application
Global flight connection risk analysis and recovery planning system
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Load .env file explicitly (before importing config)
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from app.core.config import get_settings
from app.api.v1.endpoints import router as v1_router

# Initialize settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Google Gemini: {'Enabled' if settings.google_gemini_api_key else 'Disabled'}")
    logger.info("=" * 60)
    
    # Initialize services (lazy loading handled by singletons)
    try:
        from app.services.lufthansa import get_lufthansa_client
        from app.services.vertex_ai import get_vertex_ai_service
        
        # Test Lufthansa connection
        lufthansa = get_lufthansa_client()
        if lufthansa.health_check():
            logger.info("✅ Lufthansa API: Connected")
        else:
            logger.warning("⚠️  Lufthansa API: Connection issues")
        
        # Google AI will initialize on first use
        google_ai = get_vertex_ai_service()
        if google_ai:
            logger.info("✅ Google AI: Configured with Maps integration")
        else:
            logger.warning("⚠️  Google AI: Not configured (set GOOGLE_API_KEY)")
        
    except Exception as e:
        logger.error(f"❌ Service initialization error: {str(e)}")
        # Don't crash - services will retry on demand
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info(f"👋 Shutting down {settings.app_name}")
    logger.info("=" * 60)


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    **Global Travel Delay Assistant API**
    
    Intelligent flight connection risk analysis and recovery planning for any route worldwide.
    
    **Core Features:**
    - **Real-time Flight Data**: Live status from Lufthansa API (expandable to other carriers)
    - **Connection Risk Analysis**: Predictive delay impact on flight connections
    - **AI Recovery Planning**: Gemini-powered itineraries for layover cities
    - **Multi-airline Support**: Works with any IATA flight number
    - **Global Coverage**: Supports any airport pair worldwide
    
    **Use Cases:**
    - Missed connection predictions
    - Layover city recommendations
    - Real-time rebooking assistance
    - Multi-leg journey optimization
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages"""
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Invalid request data",
            "errors": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(v1_router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint - API information
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "coverage": "Global - Any Route Worldwide",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "generate_plan": "POST /api/v1/generate-plan",
            "analyze_risk": "POST /api/v1/analyze-flight-risk",
            "flight_status": "POST /api/v1/flight-status",
            "route_status": "GET /api/v1/route-status"
        }
    }


# Additional utility endpoint
@app.get("/info", tags=["Root"])
async def info() -> Dict[str, Any]:
    """
    Detailed API information and configuration
    """
    return {
        "application": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug
        },
        "configuration": {
            "min_connection_time": f"{settings.min_connection_time} minutes",
            "critical_buffer": f"{settings.critical_buffer} minutes",
            "ai_model": settings.default_model_name,
            "api_timeout": f"{settings.api_timeout} seconds"
        },
        "services": {
            "lufthansa_api": "Enabled",
            "vertex_ai": f"Enabled ({settings.default_model_name})",
            "gcp_project": settings.gcp_project_id,
            "gcp_location": settings.gcp_location
        },
        "endpoints_count": len([route for route in app.routes if hasattr(route, 'methods')])
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development(),
        log_level=settings.log_level.lower()
    )

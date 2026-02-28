"""
Core Configuration - Environment variables and app settings
Uses Pydantic BaseSettings for type-safe configuration management
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    
    Compatible with GCP Secret Manager and Cloud Run environment variables
    """
    
    # Application settings
    app_name: str = Field(default="Travel Delay Assistant", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=False, description="Debug mode")
    
    # GCP Settings (Optional - for legacy Vertex AI)
    gcp_project_id: Optional[str] = Field(default=None, description="Google Cloud Project ID (legacy)")
    gcp_location: str = Field(default="us-central1", description="GCP region for Vertex AI (legacy)")
    
    # Google Gemini Settings (for AI and Maps integration)
    google_gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key for AI and Maps")
    
    # Lufthansa API Configuration
    lufthansa_client_id: str = Field(..., description="Lufthansa API OAuth client ID")
    lufthansa_client_secret: str = Field(..., description="Lufthansa API OAuth client secret")
    
    # API Configuration
    api_rate_limit: int = Field(default=100, description="API rate limit per minute")
    api_timeout: int = Field(default=30, description="API request timeout in seconds")
    
    # Flight Connection Settings
    min_connection_time: int = Field(
        default=60,
        ge=30,
        le=180,
        description="Minimum safe connection time in minutes"
    )
    critical_buffer: int = Field(
        default=30,
        ge=15,
        le=60,
        description="Critical connection buffer threshold in minutes"
    )
    
    # AI Model Settings
    default_model_name: str = Field(
        default="gemini-2.5-flash",
        description="Default Vertex AI model"
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default model temperature"
    )
    max_output_tokens: int = Field(
        default=2048,
        ge=256,
        le=8192,
        description="Maximum output tokens for AI generation"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # CORS Settings (for frontend integration)
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    
    # Service URLs (for external integrations)
    weather_api_url: Optional[str] = Field(None, description="Weather API URL")
    maps_api_key: Optional[str] = Field(None, description="Google Maps API key")
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value"""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level"""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()
    
    @field_validator("gcp_project_id")
    @classmethod
    def validate_project_id(cls, v):
        """Warn if project ID is placeholder (don't fail in dev)"""
        if v in ["your-project-id", "your-gcp-project-id", ""]:
            import warnings
            warnings.warn("gcp_project_id is set to placeholder value. Vertex AI features may not work.")
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    def get_log_config(self) -> dict:
        """Get logging configuration"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": self.log_format
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.log_level,
                    "formatter": "json" if self.is_production() else "default",
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": self.log_level,
                "handlers": ["console"]
            }
        }


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings (singleton)
    
    Returns:
        Settings instance
    
    Raises:
        ValueError: If required environment variables are missing
    """
    global _settings
    
    if _settings is None:
        try:
            _settings = Settings()
        except Exception as e:
            # Provide helpful error message
            raise ValueError(
                f"Failed to load settings. Please check your .env file. Error: {str(e)}"
            )
    
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings (useful for testing)
    
    Returns:
        New Settings instance
    """
    global _settings
    _settings = None
    return get_settings()


if __name__ == "__main__":
    # Test configuration loading
    print("🧪 Testing Configuration\n")
    print("=" * 60)
    
    try:
        settings = get_settings()
        
        print("\n📋 Application Settings:")
        print(f"  App Name: {settings.app_name}")
        print(f"  Version: {settings.app_version}")
        print(f"  Environment: {settings.environment}")
        print(f"  Debug: {settings.debug}")
        
        print("\n☁️  GCP Settings:")
        print(f"  Project ID: {settings.gcp_project_id}")
        print(f"  Location: {settings.gcp_location}")
        
        print("\n✈️  Lufthansa API:")
        print(f"  Client ID: {settings.lufthansa_client_id[:10]}...")
        print(f"  Client Secret: {'*' * 10}")
        
        print("\n⚙️  Connection Settings:")
        print(f"  Min Connection Time: {settings.min_connection_time} min")
        print(f"  Critical Buffer: {settings.critical_buffer} min")
        
        print("\n🤖 AI Settings:")
        print(f"  Model: {settings.default_model_name}")
        print(f"  Temperature: {settings.default_temperature}")
        print(f"  Max Tokens: {settings.max_output_tokens}")
        
        print("\n📝 Logging:")
        print(f"  Level: {settings.log_level}")
        
        print("\n✅ Configuration loaded successfully!")
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {str(e)}")
        print("\nMake sure your .env file contains:")
        print("  - GCP_PROJECT_ID")
        print("  - LUFTHANSA_CLIENT_ID")
        print("  - LUFTHANSA_CLIENT_SECRET")

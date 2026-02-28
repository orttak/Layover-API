"""
Prompt Manager - Loads and manages JSON-based prompt configurations
Implements Prompt-as-Config pattern for maintainable AI interactions
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache


class PromptManager:
    """
    Manages loading and formatting of prompt configurations from JSON files
    
    Features:
    - Hot-reload of prompt configs
    - Template variable injection
    - Model parameter management
    - Validation of required fields
    """
    
    def __init__(self, configs_dir: Optional[Path] = None):
        """
        Initialize PromptManager
        
        Args:
            configs_dir: Path to directory containing JSON prompt configs
                        Defaults to app/prompts/configs/
        """
        if configs_dir is None:
            # Default to configs directory relative to this file
            self.configs_dir = Path(__file__).parent / "configs"
        else:
            self.configs_dir = Path(configs_dir)
        
        if not self.configs_dir.exists():
            raise FileNotFoundError(f"Prompts config directory not found: {self.configs_dir}")
        
        self._cache = {}
    
    @lru_cache(maxsize=10)
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a prompt configuration from JSON file
        
        Args:
            config_name: Name of config file without .json extension
                        (e.g., 'itinerary_gen')
        
        Returns:
            Dictionary containing prompt configuration
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        config_path = self.configs_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Prompt config not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ['model_name', 'prompt_template']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in {config_name}.json")
            
            return config
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {str(e)}")
    
    def format_prompt(self, config_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load config and inject variables into prompt template
        
        Args:
            config_name: Name of the prompt configuration
            variables: Dictionary of variables to inject into template
        
        Returns:
            Dictionary with formatted prompt and model parameters
        
        Example:
            manager = PromptManager()
            prompt_data = manager.format_prompt(
                'itinerary_gen',
                {
                    'delay_minutes': 120,
                    'budget': 150,
                    'preferences': 'museums, tapas'
                }
            )
        """
        config = self.load_config(config_name)
        
        # Get template
        template = config['prompt_template']
        
        # Fill missing variables with placeholders
        safe_variables = {}
        for key, value in variables.items():
            # Convert all values to strings
            safe_variables[key] = str(value) if value is not None else "N/A"
        
        # Format template with available variables
        try:
            formatted_prompt = template.format(**safe_variables)
        except KeyError as e:
            # If a required variable is missing, raise error
            raise ValueError(f"Missing required variable {e} for prompt '{config_name}'")
        
        return {
            'prompt': formatted_prompt,
            'system_instruction': config.get('system_instruction', ''),
            'model_name': config['model_name'],
            'parameters': config.get('parameters', {}),
            'response_format': config.get('response_format', 'text'),
            'safety_settings': config.get('safety_settings', []),
            'metadata': config.get('metadata', {})
        }
    
    def get_model_parameters(self, config_name: str) -> Dict[str, Any]:
        """
        Get model parameters from config
        
        Args:
            config_name: Name of the prompt configuration
        
        Returns:
            Dictionary of model parameters (temperature, top_p, etc.)
        """
        config = self.load_config(config_name)
        return config.get('parameters', {})
    
    def get_system_instruction(self, config_name: str) -> str:
        """
        Get system instruction from config
        
        Args:
            config_name: Name of the prompt configuration
        
        Returns:
            System instruction string
        """
        config = self.load_config(config_name)
        return config.get('system_instruction', '')
    
    def list_available_prompts(self) -> list[str]:
        """
        List all available prompt configurations
        
        Returns:
            List of config names (without .json extension)
        """
        return [
            f.stem for f in self.configs_dir.glob("*.json")
        ]
    
    def reload_config(self, config_name: str):
        """
        Clear cache and reload a specific configuration
        Useful for development/testing
        
        Args:
            config_name: Name of config to reload
        """
        self.load_config.cache_clear()
        return self.load_config(config_name)
    
    def validate_variables(self, config_name: str, variables: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate that all required template variables are provided
        
        Args:
            config_name: Name of the prompt configuration
            variables: Dictionary of variables to validate
        
        Returns:
            Tuple of (is_valid, list_of_missing_variables)
        """
        config = self.load_config(config_name)
        template = config['prompt_template']
        
        # Extract variable names from template
        import re
        required_vars = set(re.findall(r'\{(\w+)\}', template))
        provided_vars = set(variables.keys())
        
        missing_vars = required_vars - provided_vars
        
        return (len(missing_vars) == 0, list(missing_vars))


# Singleton instance for easy import
_default_manager = None

def get_prompt_manager(configs_dir: Optional[Path] = None) -> PromptManager:
    """
    Get singleton PromptManager instance
    
    Args:
        configs_dir: Optional custom configs directory
    
    Returns:
        PromptManager instance
    """
    global _default_manager
    if _default_manager is None or configs_dir is not None:
        _default_manager = PromptManager(configs_dir)
    return _default_manager


# Convenience functions
def format_prompt(config_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to format a prompt using default manager"""
    manager = get_prompt_manager()
    return manager.format_prompt(config_name, variables)


def get_model_params(config_name: str) -> Dict[str, Any]:
    """Convenience function to get model parameters"""
    manager = get_prompt_manager()
    return manager.get_model_parameters(config_name)


if __name__ == "__main__":
    # Test the prompt manager
    print("🧪 Testing Prompt Manager\n")
    print("=" * 60)
    
    try:
        manager = PromptManager()
        
        print("\n📋 Available Prompts:")
        for prompt_name in manager.list_available_prompts():
            print(f"  • {prompt_name}")
        
        print("\n🔍 Testing itinerary_gen prompt:\n")
        
        result = manager.format_prompt(
            'itinerary_gen',
            {
                'delay_minutes': 180,
                'arrival_terminal': '4',
                'buffer_minutes': 45,
                'flight_status': 'DELAYED',
                'budget': 150,
                'preferences': 'museums, tapas bars, architecture',
                'duration_hours': 6
            }
        )
        
        print(f"Model: {result['model_name']}")
        print(f"Temperature: {result['parameters'].get('temperature')}")
        print(f"\nSystem Instruction:\n{result['system_instruction'][:100]}...")
        print(f"\nFormatted Prompt:\n{result['prompt'][:300]}...")
        
        print("\n✅ Prompt Manager working successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

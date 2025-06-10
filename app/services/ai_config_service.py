from typing import Optional, List
import uuid
from datetime import datetime, timezone
from supabase import create_client, Client
from app.config import get_settings
from app.schemas.ai_config import (
    AIProvider, 
    AIModel, 
    AIProviderConfig, 
    UserAIConfig, 
    GeminiConfigRequest,
    UserAIConfigDisplay
)
from app.utils.logger import logger
from app.utils.encryption import encrypt_password, decrypt_password

settings = get_settings()


class AIConfigService:
    """Service for managing AI provider configurations in Supabase"""
    
    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured")
        
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
    
    async def get_providers(self) -> List[AIProvider]:
        """
        Get all available AI providers
        
        Returns:
            List[AIProvider]: List of available providers
        """
        try:
            response = self.supabase.table('ai_providers').select('*').eq('is_active', True).execute()
            
            providers = [
                AIProvider(
                    id=row['id'],
                    name=row['name'],
                    display_name=row['display_name'],
                    is_active=row['is_active']
                )
                for row in response.data
            ]
            
            logger.info(f"Retrieved {len(providers)} AI providers")
            return providers
            
        except Exception as e:
            logger.error(f"Error retrieving AI providers: {str(e)}")
            raise
    
    async def get_models_by_provider(self, provider_name: str) -> List[AIModel]:
        """
        Get all available models for a specific provider
        
        Args:
            provider_name: Name of the provider (e.g., 'gemini')
            
        Returns:
            List[AIModel]: List of available models for the provider
        """
        try:
            response = self.supabase.table('ai_models').select(
                '''
                id,
                provider_id,
                name,
                display_name,
                version,
                is_active,
                ai_providers!inner(name)
                '''
            ).eq('ai_providers.name', provider_name).eq('is_active', True).execute()
            
            models = [
                AIModel(
                    id=row['id'],
                    provider_id=row['provider_id'],
                    name=row['name'],
                    display_name=row['display_name'],
                    version=row['version'],
                    is_active=row['is_active']
                )
                for row in response.data
            ]
            
            logger.info(f"Retrieved {len(models)} AI models for provider {provider_name}")
            return models
            
        except Exception as e:
            logger.error(f"Error retrieving AI models for provider {provider_name}: {str(e)}")
            raise
    
    async def get_user_ai_config(self, user_id: str) -> Optional[UserAIConfigDisplay]:
        """
        Get user's AI configuration with masked API key
        
        Args:
            user_id: User's ID
            
        Returns:
            UserAIConfigDisplay if found, None otherwise
        """
        try:
            # Query user_ai_configs with all related data
            response = self.supabase.table('user_ai_configs').select(
                '''
                id,
                user_id,
                created_at,
                updated_at,
                ai_provider_configs!inner(
                    id,
                    api_key,
                    ai_providers!inner(name, display_name),
                    ai_models!inner(name, display_name)
                )
                '''
            ).eq('user_id', user_id).execute()
            
            if not response.data:
                logger.info(f"No AI configuration found for user: {user_id}")
                return None
            
            config_data = response.data[0]
            provider_config = config_data['ai_provider_configs']
            provider = provider_config['ai_providers']
            model = provider_config['ai_models']
            
            # Decrypt and mask API key
            decrypted_key = decrypt_password(provider_config['api_key'])
            masked_key = self._mask_api_key(decrypted_key)
            
            user_config = UserAIConfigDisplay(
                id=config_data['id'],
                user_id=config_data['user_id'],
                provider_name=provider['name'],
                provider_display_name=provider['display_name'],
                api_key_masked=masked_key,
                default_model_name=model['name'],
                default_model_display_name=model['display_name'],
                created_at=datetime.fromisoformat(config_data['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(config_data['updated_at'].replace('Z', '+00:00'))
            )
            
            logger.info(f"Retrieved AI configuration for user: {user_id}")
            return user_config
            
        except Exception as e:
            logger.error(f"Error retrieving user AI config for {user_id}: {str(e)}")
            raise
    
    async def create_gemini_config(self, user_id: str, config_request: GeminiConfigRequest) -> UserAIConfigDisplay:
        """
        Create a new Gemini configuration for a user
        
        Args:
            user_id: User's ID
            config_request: Gemini configuration data
            
        Returns:
            UserAIConfigDisplay: Created configuration with masked API key
        """
        try:
            # Check if user already has a configuration
            existing_config = await self.get_user_ai_config(user_id)
            if existing_config:
                raise ValueError(f"AI configuration already exists for user: {user_id}")
            
            # Get Gemini provider
            gemini_provider = self.supabase.table('ai_providers').select('id').eq('name', 'gemini').execute()
            if not gemini_provider.data:
                raise ValueError("Gemini provider not found")
            provider_id = gemini_provider.data[0]['id']
            
            # Get the requested model and validate it exists
            model_response = self.supabase.table('ai_models').select('id, name, display_name').eq(
                'provider_id', provider_id
            ).eq('name', config_request.default_model_name).eq('is_active', True).execute()
            
            if not model_response.data:
                # Get available models to show in error message
                available_models = self.supabase.table('ai_models').select('name, display_name').eq(
                    'provider_id', provider_id
                ).eq('is_active', True).execute()
                available_names = [model['name'] for model in available_models.data] if available_models.data else []
                raise ValueError(f"Model '{config_request.default_model_name}' not found for Gemini. Available models: {', '.join(available_names)}")
            model_id = model_response.data[0]['id']
            
            # Create AI provider config with encrypted API key
            provider_config_id = str(uuid.uuid4())
            provider_config_response = self.supabase.table('ai_provider_configs').insert({
                'id': provider_config_id,
                'provider_id': provider_id,
                'api_key': encrypt_password(config_request.api_key),
                'default_model_id': model_id,
                'config_data': {}
            }).execute()
            
            if not provider_config_response.data:
                raise Exception("Failed to create AI provider configuration")
            
            # Create user AI config
            user_config_id = str(uuid.uuid4())
            user_config_response = self.supabase.table('user_ai_configs').insert({
                'id': user_config_id,
                'user_id': user_id,
                'active_provider_config_id': provider_config_id
            }).execute()
            
            if not user_config_response.data:
                raise Exception("Failed to create user AI configuration")
            
            logger.info(f"Created Gemini configuration for user: {user_id}")
            
            # Return the created configuration
            return await self.get_user_ai_config(user_id)
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error creating Gemini config for {user_id}: {str(e)}")
            raise Exception(f"Failed to create Gemini configuration: {str(e)}")
    
    async def update_gemini_config(self, user_id: str, config_request: GeminiConfigRequest) -> UserAIConfigDisplay:
        """
        Update an existing Gemini configuration for a user
        
        Args:
            user_id: User's ID
            config_request: Updated Gemini configuration data
            
        Returns:
            UserAIConfigDisplay: Updated configuration with masked API key
        """
        try:
            # Get existing configuration
            existing_config = await self.get_user_ai_config(user_id)
            if not existing_config:
                raise ValueError(f"No AI configuration found for user: {user_id}")
            
            # Get the user's current provider config ID
            user_config_response = self.supabase.table('user_ai_configs').select(
                'active_provider_config_id'
            ).eq('user_id', user_id).execute()
            
            if not user_config_response.data:
                raise ValueError(f"User AI configuration not found for user: {user_id}")
            
            provider_config_id = user_config_response.data[0]['active_provider_config_id']
            
            # Get Gemini provider
            gemini_provider = self.supabase.table('ai_providers').select('id').eq('name', 'gemini').execute()
            if not gemini_provider.data:
                raise ValueError("Gemini provider not found")
            provider_id = gemini_provider.data[0]['id']
            
            # Get the requested model and validate it exists
            model_response = self.supabase.table('ai_models').select('id, name, display_name').eq(
                'provider_id', provider_id
            ).eq('name', config_request.default_model_name).eq('is_active', True).execute()
            
            if not model_response.data:
                # Get available models to show in error message
                available_models = self.supabase.table('ai_models').select('name, display_name').eq(
                    'provider_id', provider_id
                ).eq('is_active', True).execute()
                available_names = [model['name'] for model in available_models.data] if available_models.data else []
                raise ValueError(f"Model '{config_request.default_model_name}' not found for Gemini. Available models: {', '.join(available_names)}")
            model_id = model_response.data[0]['id']
            
            # Update AI provider config with encrypted API key
            update_response = self.supabase.table('ai_provider_configs').update({
                'api_key': encrypt_password(config_request.api_key),
                'default_model_id': model_id,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', provider_config_id).execute()
            
            if not update_response.data:
                raise Exception("Failed to update AI provider configuration")
            
            logger.info(f"Updated Gemini configuration for user: {user_id}")
            
            # Return the updated configuration
            return await self.get_user_ai_config(user_id)
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error updating Gemini config for {user_id}: {str(e)}")
            raise Exception(f"Failed to update Gemini configuration: {str(e)}")
    
    def _mask_api_key(self, api_key: str) -> str:
        """
        Mask an API key for display purposes
        
        Args:
            api_key: The full API key
            
        Returns:
            str: Masked API key (e.g., 'AIza****')
        """
        if not api_key:
            return "****"
        
        if len(api_key) <= 8:
            return "****"
        
        # Show first 4 characters and mask the rest
        return api_key[:4] + "****"


# Global service instance
ai_config_service = AIConfigService()
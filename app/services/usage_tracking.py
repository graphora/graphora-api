"""
Usage tracking service for document processing and LLM costs.

This service tracks and stores usage metrics for pricing and analytics,
including document processing statistics and LLM token consumption.
"""

from typing import Optional, Dict, Any, List, Tuple
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from supabase import create_client, Client
from app.config import get_settings
from app.schemas.usage import (
    DocumentUsage, LLMUsage, UsageAggregate, PricingTier, UserPricingTier,
    ProcessingStatus, PeriodType, ModelProvider, UsageReport, LimitCheckResult,
    DocumentUsageRequest, LLMUsageRequest, ModelProviderSchema, ModelPricingSchema
)
from app.utils.logger import logger

settings = get_settings()


class UsageTrackingService:
    """Service for tracking and managing usage data"""
    
    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured")
        
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        
        # Cache for model pricing to avoid repeated DB queries
        self._pricing_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._cache_last_updated: Optional[datetime] = None
        self._cache_ttl_minutes = 60  # Cache for 1 hour
    
    async def track_document_processing(
        self,
        user_id: str,
        request: DocumentUsageRequest,
        processing_started_at: Optional[datetime] = None
    ) -> DocumentUsage:
        """
        Track document processing usage
        
        Args:
            user_id: User's ID
            request: Document usage request data
            processing_started_at: When processing started (defaults to now)
            
        Returns:
            DocumentUsage: Created usage record
        """
        try:
            # Calculate billable pages (minimum 1)
            billable_pages = max(1, request.page_count)
            
            # Create usage record
            usage_data = {
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                'transform_id': request.transform_id,
                'document_name': request.document_name,
                'document_type': request.document_type.upper(),
                'document_size_bytes': request.document_size_bytes,
                'page_count': request.page_count,
                'processing_status': ProcessingStatus.IN_PROGRESS.value,
                'processing_started_at': (processing_started_at or datetime.now(timezone.utc)).isoformat(),
                'billable_pages': billable_pages,
                'billable_processing_units': 1
            }
            
            response = self.supabase.table('document_usage').insert(usage_data).execute()
            
            if not response.data:
                raise Exception("Failed to create document usage record")
            
            record = response.data[0]
            logger.info(f"Tracked document processing for user {user_id}: {request.document_name}")
            
            return DocumentUsage(**record)
            
        except Exception as e:
            logger.error(f"Error tracking document processing for {user_id}: {str(e)}")
            raise
    
    async def update_document_processing(
        self,
        document_usage_id: str,
        processing_completed_at: Optional[datetime] = None,
        processing_status: ProcessingStatus = ProcessingStatus.SUCCESS,
        chunks_created: int = 0,
        nodes_extracted: int = 0,
        relationships_extracted: int = 0,
        success_rate: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> DocumentUsage:
        """
        Update document processing results
        
        Args:
            document_usage_id: ID of the document usage record
            processing_completed_at: When processing completed
            processing_status: Final processing status
            chunks_created: Number of chunks created
            nodes_extracted: Number of nodes extracted
            relationships_extracted: Number of relationships extracted
            success_rate: Processing success rate (0.0-1.0)
            error_message: Error message if failed
            
        Returns:
            DocumentUsage: Updated usage record
        """
        try:
            completed_at = processing_completed_at or datetime.now(timezone.utc)
            
            # Get the original record to calculate duration
            existing = self.supabase.table('document_usage').select('processing_started_at').eq('id', document_usage_id).execute()
            if not existing.data:
                raise Exception(f"Document usage record {document_usage_id} not found")
            
            started_at = datetime.fromisoformat(existing.data[0]['processing_started_at'].replace('Z', '+00:00'))
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            
            # Update data
            update_data = {
                'processing_status': processing_status.value,
                'processing_completed_at': completed_at.isoformat(),
                'processing_duration_ms': duration_ms,
                'chunks_created': chunks_created,
                'nodes_extracted': nodes_extracted,
                'relationships_extracted': relationships_extracted
            }
            
            if success_rate is not None:
                update_data['success_rate'] = success_rate
            
            response = self.supabase.table('document_usage').update(update_data).eq('id', document_usage_id).execute()
            
            if not response.data:
                raise Exception("Failed to update document usage record")
            
            record = response.data[0]
            logger.info(f"Updated document processing {document_usage_id}: {processing_status.value}")
            
            return DocumentUsage(**record)
            
        except Exception as e:
            logger.error(f"Error updating document processing {document_usage_id}: {str(e)}")
            raise
    
    async def track_llm_usage(
        self,
        user_id: str,
        request: LLMUsageRequest,
        request_timestamp: Optional[datetime] = None,
        response_timestamp: Optional[datetime] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> LLMUsage:
        """
        Track LLM usage and calculate costs
        
        Args:
            user_id: User's ID
            request: LLM usage request data
            request_timestamp: When request was made
            response_timestamp: When response received
            success: Whether the operation succeeded
            error_message: Error message if failed
            
        Returns:
            LLMUsage: Created usage record
        """
        try:
            req_time = request_timestamp or datetime.now(timezone.utc)
            resp_time = response_timestamp or datetime.now(timezone.utc)
            
            # Calculate latency
            latency_ms = request.latency_ms
            if latency_ms is None and response_timestamp:
                latency_ms = int((resp_time - req_time).total_seconds() * 1000)
            
            # Get pricing for the model
            model_costs = await self._get_model_pricing(request.model_provider, request.model_name)
            
            # Calculate costs
            estimated_cost = None
            cost_per_1k_input = None
            cost_per_1k_output = None
            
            if model_costs:
                cost_per_1k_input = Decimal(str(model_costs["input"]))
                cost_per_1k_output = Decimal(str(model_costs["output"]))
                
                input_cost = (Decimal(str(request.input_tokens)) / 1000) * cost_per_1k_input
                output_cost = (Decimal(str(request.output_tokens)) / 1000) * cost_per_1k_output
                estimated_cost = input_cost + output_cost
            
            # Create usage record
            usage_data = {
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                'transform_id': request.transform_id,
                'document_usage_id': request.document_usage_id,
                'model_provider': request.model_provider.value,
                'model_name': request.model_name,
                'input_tokens': request.input_tokens,
                'output_tokens': request.output_tokens,
                'total_tokens': request.input_tokens + request.output_tokens,
                'estimated_cost_usd': str(estimated_cost) if estimated_cost else None,
                'cost_per_1k_input_tokens': str(cost_per_1k_input) if cost_per_1k_input else None,
                'cost_per_1k_output_tokens': str(cost_per_1k_output) if cost_per_1k_output else None,
                'operation_type': request.operation_type,
                'latency_ms': latency_ms,
                'success': success,
                'error_message': error_message,
                'request_timestamp': req_time.isoformat(),
                'response_timestamp': resp_time.isoformat() if response_timestamp else None
            }
            
            response = self.supabase.table('llm_usage').insert(usage_data).execute()
            
            if not response.data:
                raise Exception("Failed to create LLM usage record")
            
            record = response.data[0]
            logger.info(f"Tracked LLM usage for user {user_id}: {request.model_name} ({request.input_tokens + request.output_tokens} tokens)")
            
            return LLMUsage(**record)
            
        except Exception as e:
            logger.error(f"Error tracking LLM usage for {user_id}: {str(e)}")
            raise
    
    async def _load_pricing_from_db(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Load model pricing from database"""
        try:
            # Get all model pricing with provider information
            response = self.supabase.table('model_pricing').select(
                '''
                *,
                provider:provider_id(provider_name)
                '''
            ).eq('is_active', True).execute()
            
            if not response.data:
                logger.warning("No model pricing found in database")
                return {}
            
            # Organize pricing by provider and model
            pricing_data = {}
            for pricing in response.data:
                provider_name = pricing['provider']['provider_name']
                model_name = pricing['model_name']
                
                if provider_name not in pricing_data:
                    pricing_data[provider_name] = {}
                
                pricing_data[provider_name][model_name] = {
                    "input": float(pricing['input_price_per_1k_tokens']),
                    "output": float(pricing['output_price_per_1k_tokens'])
                }
            
            logger.info(f"Loaded pricing for {len(response.data)} models from database")
            return pricing_data
            
        except Exception as e:
            logger.error(f"Error loading pricing from database: {str(e)}")
            return {}
    
    async def _get_cached_pricing(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get cached pricing or load from database if cache is stale"""
        now = datetime.now(timezone.utc)
        
        # Check if cache is empty or stale
        if (not self._pricing_cache or
            not self._cache_last_updated or
            (now - self._cache_last_updated).total_seconds() > (self._cache_ttl_minutes * 60)):
            
            # Reload from database
            self._pricing_cache = await self._load_pricing_from_db()
            self._cache_last_updated = now
            logger.info("Refreshed model pricing cache")
        
        return self._pricing_cache
    
    async def _get_model_pricing(self, provider: ModelProvider, model_name: str) -> Optional[Dict[str, float]]:
        """Get pricing information for a model"""
        try:
            pricing_data = await self._get_cached_pricing()
            provider_pricing = pricing_data.get(provider.value, {})
            return provider_pricing.get(model_name)
        except Exception as e:
            logger.error(f"Error getting model pricing for {provider.value}:{model_name}: {str(e)}")
            return None
    
    async def check_user_limits(self, user_id: str) -> LimitCheckResult:
        """
        Check if user is within their usage limits
        
        Args:
            user_id: User's ID
            
        Returns:
            LimitCheckResult: Limit check results
        """
        try:
            # Get user's pricing tier
            tier_response = self.supabase.table('user_pricing_tiers').select(
                '''
                *,
                pricing_tier:tier_id(*)
                '''
            ).eq('user_id', user_id).eq('is_active', True).execute()
            
            if not tier_response.data:
                # Default to free tier if no assignment
                tier_data = {"tier_name": "Free", "monthly_document_limit": 10, 
                           "monthly_page_limit": 100, "monthly_token_limit": 50000}
                current_usage = {"current_documents": 0, "current_pages": 0, 
                               "current_tokens": 0, "current_cost_usd": Decimal('0')}
            else:
                user_tier = tier_response.data[0]
                tier_data = user_tier['pricing_tier']
                current_usage = {
                    "current_documents": user_tier['current_documents'],
                    "current_pages": user_tier['current_pages'],
                    "current_tokens": user_tier['current_tokens'],
                    "current_cost_usd": Decimal(str(user_tier['current_cost_usd']))
                }
            
            # Check limits
            within_limits = True
            warnings = []
            
            # Document limit
            doc_limit = tier_data.get('monthly_document_limit')
            if doc_limit and current_usage['current_documents'] >= doc_limit:
                within_limits = False
                warnings.append(f"Document limit exceeded ({current_usage['current_documents']}/{doc_limit})")
            
            # Page limit
            page_limit = tier_data.get('monthly_page_limit')
            if page_limit and current_usage['current_pages'] >= page_limit:
                within_limits = False
                warnings.append(f"Page limit exceeded ({current_usage['current_pages']}/{page_limit})")
            
            # Token limit
            token_limit = tier_data.get('monthly_token_limit')
            if token_limit and current_usage['current_tokens'] >= token_limit:
                within_limits = False
                warnings.append(f"Token limit exceeded ({current_usage['current_tokens']}/{token_limit})")
            
            # Cost limit
            cost_limit = tier_data.get('monthly_cost_limit_usd')
            if cost_limit and current_usage['current_cost_usd'] >= Decimal(str(cost_limit)):
                within_limits = False
                warnings.append(f"Cost limit exceeded (${current_usage['current_cost_usd']}/${cost_limit})")
            
            # Calculate remaining capacity
            remaining_documents = None
            remaining_pages = None
            remaining_tokens = None
            remaining_cost = None
            
            if doc_limit:
                remaining_documents = max(0, doc_limit - current_usage['current_documents'])
            if page_limit:
                remaining_pages = max(0, page_limit - current_usage['current_pages'])
            if token_limit:
                remaining_tokens = max(0, token_limit - current_usage['current_tokens'])
            if cost_limit:
                remaining_cost = max(Decimal('0'), Decimal(str(cost_limit)) - current_usage['current_cost_usd'])
            
            # Recommend upgrade if approaching limits
            upgrade_recommended = False
            if remaining_documents is not None and remaining_documents <= 2:
                upgrade_recommended = True
            if remaining_pages is not None and remaining_pages <= 10:
                upgrade_recommended = True
            
            return LimitCheckResult(
                within_limits=within_limits,
                tier_name=tier_data['tier_name'],
                current_documents=current_usage['current_documents'],
                current_pages=current_usage['current_pages'],
                current_tokens=current_usage['current_tokens'],
                current_cost_usd=current_usage['current_cost_usd'],
                document_limit=doc_limit,
                page_limit=page_limit,
                token_limit=token_limit,
                cost_limit_usd=Decimal(str(cost_limit)) if cost_limit else None,
                remaining_documents=remaining_documents,
                remaining_pages=remaining_pages,
                remaining_tokens=remaining_tokens,
                remaining_cost_usd=remaining_cost,
                warnings=warnings,
                upgrade_recommended=upgrade_recommended
            )
            
        except Exception as e:
            logger.error(f"Error checking user limits for {user_id}: {str(e)}")
            raise
    
    async def get_usage_report(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> UsageReport:
        """
        Generate usage report for a user
        
        Args:
            user_id: User's ID
            period_start: Report period start
            period_end: Report period end
            
        Returns:
            UsageReport: Usage report
        """
        try:
            # Get document usage
            doc_response = self.supabase.table('document_usage').select('*').eq('user_id', user_id).gte('created_at', period_start.isoformat()).lte('created_at', period_end.isoformat()).execute()
            
            # Get LLM usage
            llm_response = self.supabase.table('llm_usage').select('*').eq('user_id', user_id).gte('created_at', period_start.isoformat()).lte('created_at', period_end.isoformat()).execute()
            
            doc_data = doc_response.data or []
            llm_data = llm_response.data or []
            
            # Calculate summary statistics
            total_documents = len(doc_data)
            total_pages = sum(doc.get('page_count', 0) for doc in doc_data)
            total_llm_calls = len(llm_data)
            total_tokens = sum(llm.get('total_tokens', 0) for llm in llm_data)
            
            # Calculate total cost
            total_cost = Decimal('0')
            for llm in llm_data:
                if llm.get('estimated_cost_usd'):
                    total_cost += Decimal(str(llm['estimated_cost_usd']))
            
            # Document type breakdown
            document_types = {}
            for doc in doc_data:
                doc_type = doc.get('document_type', 'Unknown')
                document_types[doc_type] = document_types.get(doc_type, 0) + 1
            
            # Model usage breakdown
            model_usage = {}
            for llm in llm_data:
                model_key = f"{llm.get('model_provider', 'unknown')}:{llm.get('model_name', 'unknown')}"
                if model_key not in model_usage:
                    model_usage[model_key] = {
                        'calls': 0,
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0,
                        'estimated_cost_usd': Decimal('0')
                    }
                
                model_usage[model_key]['calls'] += 1
                model_usage[model_key]['input_tokens'] += llm.get('input_tokens', 0)
                model_usage[model_key]['output_tokens'] += llm.get('output_tokens', 0)
                model_usage[model_key]['total_tokens'] += llm.get('total_tokens', 0)
                if llm.get('estimated_cost_usd'):
                    model_usage[model_key]['estimated_cost_usd'] += Decimal(str(llm['estimated_cost_usd']))
            
            # Convert Decimal to string for JSON serialization
            for model in model_usage.values():
                model['estimated_cost_usd'] = str(model['estimated_cost_usd'])
            
            # Daily usage breakdown (simplified)
            daily_usage = []
            current_date = period_start.date()
            while current_date <= period_end.date():
                day_docs = [doc for doc in doc_data 
                           if datetime.fromisoformat(doc['created_at'].replace('Z', '+00:00')).date() == current_date]
                day_llm = [llm for llm in llm_data 
                          if datetime.fromisoformat(llm['created_at'].replace('Z', '+00:00')).date() == current_date]
                
                daily_usage.append({
                    'date': current_date.isoformat(),
                    'documents': len(day_docs),
                    'pages': sum(doc.get('page_count', 0) for doc in day_docs),
                    'llm_calls': len(day_llm),
                    'tokens': sum(llm.get('total_tokens', 0) for llm in day_llm)
                })
                current_date += timedelta(days=1)
            
            # Performance metrics
            successful_docs = [doc for doc in doc_data if doc.get('processing_status') == 'success']
            success_rate = (len(successful_docs) / len(doc_data) * 100) if doc_data else None
            
            processing_times = [doc.get('processing_duration_ms') for doc in successful_docs 
                              if doc.get('processing_duration_ms')]
            avg_processing_time = sum(processing_times) // len(processing_times) if processing_times else None
            
            return UsageReport(
                user_id=user_id,
                period_start=period_start,
                period_end=period_end,
                total_documents=total_documents,
                total_pages=total_pages,
                total_llm_calls=total_llm_calls,
                total_tokens=total_tokens,
                estimated_total_cost_usd=total_cost,
                document_types=document_types,
                model_usage=model_usage,
                daily_usage=daily_usage,
                avg_processing_time_ms=avg_processing_time,
                success_rate=Decimal(str(success_rate)) if success_rate else None
            )
            
        except Exception as e:
            logger.error(f"Error generating usage report for {user_id}: {str(e)}")
            raise
    
    async def get_all_model_providers(self) -> List[ModelProviderSchema]:
        """Get all active model providers"""
        try:
            response = self.supabase.table('model_providers').select('*').eq('is_active', True).execute()
            return [ModelProviderSchema(**provider) for provider in response.data or []]
        except Exception as e:
            logger.error(f"Error getting model providers: {str(e)}")
            raise
    
    async def get_model_pricing_by_provider(self, provider_name: str) -> List[ModelPricingSchema]:
        """Get all pricing for a specific provider"""
        try:
            response = self.supabase.table('model_pricing').select(
                '''
                *,
                provider:provider_id(provider_name)
                '''
            ).eq('provider.provider_name', provider_name).eq('is_active', True).execute()
            
            return [ModelPricingSchema(**pricing) for pricing in response.data or []]
        except Exception as e:
            logger.error(f"Error getting pricing for provider {provider_name}: {str(e)}")
            raise
    
    async def update_model_pricing(
        self,
        pricing_id: str,
        input_price: Optional[Decimal] = None,
        output_price: Optional[Decimal] = None,
        model_description: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> ModelPricingSchema:
        """Update model pricing"""
        try:
            update_data = {}
            if input_price is not None:
                update_data['input_price_per_1k_tokens'] = str(input_price)
            if output_price is not None:
                update_data['output_price_per_1k_tokens'] = str(output_price)
            if model_description is not None:
                update_data['model_description'] = model_description
            if is_active is not None:
                update_data['is_active'] = is_active
            
            if not update_data:
                raise ValueError("No update data provided")
            
            response = self.supabase.table('model_pricing').update(update_data).eq('id', pricing_id).execute()
            
            if not response.data:
                raise Exception(f"Model pricing {pricing_id} not found or update failed")
            
            # Clear cache to force reload
            self._pricing_cache = {}
            self._cache_last_updated = None
            
            logger.info(f"Updated model pricing {pricing_id}")
            return ModelPricingSchema(**response.data[0])
            
        except Exception as e:
            logger.error(f"Error updating model pricing {pricing_id}: {str(e)}")
            raise
    
    async def add_model_pricing(
        self,
        provider_name: str,
        model_name: str,
        input_price: Decimal,
        output_price: Decimal,
        model_version: Optional[str] = None,
        model_description: Optional[str] = None,
        model_context_window: Optional[int] = None
    ) -> ModelPricingSchema:
        """Add new model pricing"""
        try:
            # Get provider ID
            provider_response = self.supabase.table('model_providers').select('id').eq('provider_name', provider_name).execute()
            
            if not provider_response.data:
                raise ValueError(f"Provider {provider_name} not found")
            
            provider_id = provider_response.data[0]['id']
            
            # Create pricing record
            pricing_data = {
                'id': str(uuid.uuid4()),
                'provider_id': provider_id,
                'model_name': model_name,
                'input_price_per_1k_tokens': str(input_price),
                'output_price_per_1k_tokens': str(output_price)
            }
            
            if model_version:
                pricing_data['model_version'] = model_version
            if model_description:
                pricing_data['model_description'] = model_description
            if model_context_window:
                pricing_data['model_context_window'] = model_context_window
            
            response = self.supabase.table('model_pricing').insert(pricing_data).execute()
            
            if not response.data:
                raise Exception("Failed to create model pricing")
            
            # Clear cache to force reload
            self._pricing_cache = {}
            self._cache_last_updated = None
            
            logger.info(f"Added pricing for {provider_name}:{model_name}")
            return ModelPricingSchema(**response.data[0])
            
        except Exception as e:
            logger.error(f"Error adding model pricing for {provider_name}:{model_name}: {str(e)}")
            raise


# Global instance
usage_tracking_service = UsageTrackingService()
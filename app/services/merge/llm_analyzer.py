"""LLM-based conflict analyzer for intelligent merge resolution."""
import logging
import json
from typing import Dict, List, Any
import uuid

from app.schemas.conflicts import (
    Conflict, 
    ConflictType, 
    ResolutionOption
)
from app.config import settings
from app.baml_client import b as baml_client

logger = logging.getLogger(__name__)

class LLMConflictAnalyzer:
    """
    Analyzer that uses LLM to provide intelligent conflict resolution options.
    """
    
    def __init__(self):
        """Initialize the LLM conflict analyzer with configuration settings."""
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.OPENAI_MAX_TOKENS
    
    async def analyze_conflict(self, conflict: Conflict, ontology: Dict[str, Any]) -> List[ResolutionOption]:
        """
        Analyze a conflict and provide resolution options using LLM.
        
        Args:
            conflict: The conflict to analyze
            ontology: The ontology constraints to consider
            
        Returns:
            List of resolution options with confidence scores
        """
        try:
            # Route to the appropriate analysis method based on conflict type
            if conflict.conflict_type == ConflictType.PROPERTY:
                return await self.analyze_property_conflict(conflict, ontology)
            elif conflict.conflict_type == ConflictType.RELATIONSHIP:
                return await self.analyze_relationship_conflict(conflict, ontology)
            elif conflict.conflict_type == ConflictType.ENTITY_MATCH:
                return await self.analyze_entity_match_conflict(conflict, ontology)
            else:
                # For unsupported conflict types, return default options
                logger.warning(f"Unsupported conflict type for LLM analysis: {conflict.conflict_type}")
                return self._get_default_options(conflict)
        except Exception as e:
            logger.error(f"Error analyzing conflict with LLM: {str(e)}")
            # Return default options if analysis fails
            return self._get_default_options(conflict)
    
    async def analyze_property_conflict(self, conflict: Conflict, ontology: Dict[str, Any]) -> List[ResolutionOption]:
        """
        Analyze a property value conflict and provide resolution options.
        
        Args:
            conflict: The property value conflict
            ontology: The ontology constraints to consider
            
        Returns:
            List of resolution options with confidence scores
        """
        # Extract relevant property constraints from ontology
        property_name = conflict.property_name
        property_constraints = ontology.get("property_constraints", {}).get(property_name, {})
        
        # Build the prompt for LLM
        prompt = f"""
        Analyze this property value conflict and recommend the best resolution option.
        
        Property: {property_name}
        Entity Type: {conflict.entity_type}
        Staging Value: {conflict.staging_value}
        Production Value: {conflict.production_value}
        
        Property Constraints:
        {json.dumps(property_constraints, indent=2)}
        
        Provide a ranked list of resolution options with confidence scores and explanations.
        Focus on data quality, consistency with constraints, and likely correctness.
        """
        
        # Get analysis from LLM
        llm_results = await self._analyze_with_llm(prompt)
        
        # Convert LLM results to ResolutionOption objects
        options = []
        for i, result in enumerate(llm_results):
            value = result.get("value", "")
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "")
            
            # Determine resolution type based on which value matches
            if value == conflict.staging_value:
                resolution_type = "keep_staging"
            elif value == conflict.production_value:
                resolution_type = "keep_production"
            else:
                resolution_type = "custom_value"
            
            # Create resolution option
            option = ResolutionOption(
                id=f"option_{i+1}_{uuid.uuid4().hex[:8]}",
                description=f"Use value: {value}",
                resolution_type=resolution_type,
                resolution_data={"property_name": property_name, "value": value},
                confidence=confidence,
                reasoning=explanation,
                requires_review=confidence < 0.7,  # Require review for low confidence
                auto_resolvable=confidence > 0.8   # Auto-resolvable for high confidence
            )
            options.append(option)
        
        return options
    
    async def analyze_relationship_conflict(self, conflict: Conflict, ontology: Dict[str, Any]) -> List[ResolutionOption]:
        """
        Analyze a relationship type conflict and provide resolution options.
        
        Args:
            conflict: The relationship type conflict
            ontology: The ontology constraints to consider
            
        Returns:
            List of resolution options with confidence scores
        """
        # Extract relevant relationship types from ontology
        relationship_types = ontology.get("relationship_types", [])
        
        # Build the prompt for LLM
        prompt = f"""
        Analyze this relationship type conflict and recommend the best resolution option.
        
        Entity ID: {conflict.entity_id}
        Staging Relationship Type: {conflict.staging_value}
        Production Relationship Type: {conflict.production_value}
        
        Available Relationship Types in Ontology:
        {json.dumps(relationship_types, indent=2)}
        
        Provide a ranked list of resolution options with confidence scores and explanations.
        Focus on semantic correctness, ontology compliance, and standard terminology.
        """
        
        # Get analysis from LLM
        llm_results = await self._analyze_with_llm(prompt)
        
        # Convert LLM results to ResolutionOption objects
        options = []
        for i, result in enumerate(llm_results):
            rel_type = result.get("type", "")
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "")
            
            # Determine resolution type based on which type matches
            if rel_type == conflict.staging_value:
                resolution_type = "keep_staging_rel"
            elif rel_type == conflict.production_value:
                resolution_type = "keep_production_rel"
            else:
                resolution_type = "custom_rel_type"
            
            # Create resolution option
            option = ResolutionOption(
                id=f"option_{i+1}_{uuid.uuid4().hex[:8]}",
                description=f"Use relationship type: {rel_type}",
                resolution_type=resolution_type,
                resolution_data={"relationship_type": rel_type},
                confidence=confidence,
                reasoning=explanation,
                requires_review=confidence < 0.7,
                auto_resolvable=confidence > 0.8
            )
            options.append(option)
        
        return options
    
    async def analyze_entity_match_conflict(self, conflict: Conflict, ontology: Dict[str, Any]) -> List[ResolutionOption]:
        """
        Analyze an entity match conflict and provide resolution options.
        
        Args:
            conflict: The entity match conflict
            ontology: The ontology constraints to consider
            
        Returns:
            List of resolution options with confidence scores
        """
        # Build the prompt for LLM
        prompt = f"""
        Analyze this potential entity match conflict and recommend the best resolution option.
        
        Source Entity ID: {conflict.entity_id}
        Target Entity ID: {conflict.staging_value}
        Similarity Score: {conflict.production_value or 0.0}
        
        Provide a ranked list of resolution options with confidence scores and explanations.
        Consider whether these entities should be merged or kept separate.
        Focus on entity identity, data quality, and avoiding duplication.
        """
        
        # Get analysis from LLM
        llm_results = await self._analyze_with_llm(prompt)
        
        # Convert LLM results to ResolutionOption objects
        options = []
        for i, result in enumerate(llm_results):
            action = result.get("action", "")
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "")
            
            # Determine resolution type based on action
            if action.lower() == "merge":
                resolution_type = "merge_entities"
                description = "Merge entities"
            else:
                resolution_type = "keep_separate"
                description = "Keep entities separate"
            
            # Create resolution option
            option = ResolutionOption(
                id=f"option_{i+1}_{uuid.uuid4().hex[:8]}",
                description=description,
                resolution_type=resolution_type,
                resolution_data={
                    "source_entity_id": conflict.entity_id,
                    "target_entity_id": conflict.staging_value,
                    "similarity": conflict.production_value or 0.0
                },
                confidence=confidence,
                reasoning=explanation,
                requires_review=confidence < 0.8,  # Higher threshold for entity merges
                auto_resolvable=confidence > 0.9   # Higher threshold for auto-resolution
            )
            options.append(option)
        
        return options
    
    def _get_default_options(self, conflict: Conflict) -> List[ResolutionOption]:
        """
        Generate default resolution options when LLM analysis fails.
        
        Args:
            conflict: The conflict to generate default options for
            
        Returns:
            List of basic resolution options
        """
        options = []
        
        if conflict.conflict_type == ConflictType.PROPERTY:
            # Default options for property conflicts
            options.append(ResolutionOption(
                id=f"default_staging_{uuid.uuid4().hex[:8]}",
                description=f"Keep staging value: {conflict.staging_value}",
                resolution_type="keep_staging",
                resolution_data={"property_name": conflict.property_name, "value": conflict.staging_value},
                confidence=0.5,
                reasoning="Default option to keep staging value",
                requires_review=True,
                auto_resolvable=False
            ))
            options.append(ResolutionOption(
                id=f"default_production_{uuid.uuid4().hex[:8]}",
                description=f"Keep production value: {conflict.production_value}",
                resolution_type="keep_production",
                resolution_data={"property_name": conflict.property_name, "value": conflict.production_value},
                confidence=0.5,
                reasoning="Default option to keep production value",
                requires_review=True,
                auto_resolvable=False
            ))
        elif conflict.conflict_type == ConflictType.RELATIONSHIP:
            # Default options for relationship conflicts
            options.append(ResolutionOption(
                id=f"default_staging_rel_{uuid.uuid4().hex[:8]}",
                description=f"Keep staging relationship type: {conflict.staging_value}",
                resolution_type="keep_staging_rel",
                resolution_data={"relationship_type": conflict.staging_value},
                confidence=0.5,
                reasoning="Default option to keep staging relationship type",
                requires_review=True,
                auto_resolvable=False
            ))
            options.append(ResolutionOption(
                id=f"default_production_rel_{uuid.uuid4().hex[:8]}",
                description=f"Keep production relationship type: {conflict.production_value}",
                resolution_type="keep_production_rel",
                resolution_data={"relationship_type": conflict.production_value},
                confidence=0.5,
                reasoning="Default option to keep production relationship type",
                requires_review=True,
                auto_resolvable=False
            ))
        elif conflict.conflict_type == ConflictType.ENTITY_MATCH:
            # Default options for entity match conflicts
            similarity = conflict.production_value or 0.0
            
            options.append(ResolutionOption(
                id=f"default_merge_{uuid.uuid4().hex[:8]}",
                description="Merge entities",
                resolution_type="merge_entities",
                resolution_data={
                    "source_entity_id": conflict.entity_id,
                    "target_entity_id": conflict.staging_value,
                    "similarity": similarity
                },
                confidence=similarity,  # Use similarity score as confidence
                reasoning="Default option to merge entities based on similarity",
                requires_review=True,
                auto_resolvable=False
            ))
            options.append(ResolutionOption(
                id=f"default_separate_{uuid.uuid4().hex[:8]}",
                description="Keep entities separate",
                resolution_type="keep_separate",
                resolution_data={
                    "source_entity_id": conflict.entity_id,
                    "target_entity_id": conflict.staging_value,
                    "similarity": similarity
                },
                confidence=1.0 - similarity,  # Inverse of similarity score
                reasoning="Default option to keep entities separate",
                requires_review=True,
                auto_resolvable=False
            ))
        else:
            # Generic options for other conflict types
            options.append(ResolutionOption(
                id=f"default_no_action_{uuid.uuid4().hex[:8]}",
                description="No automatic resolution available",
                resolution_type="manual_review",
                resolution_data={},
                confidence=0.0,
                reasoning="This conflict type requires manual review",
                requires_review=True,
                auto_resolvable=False
            ))
        
        return options
    
    async def _analyze_with_llm(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Call the LLM for analysis using BAML client.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            List of parsed options from LLM response
        """
        try:
            # Use BAML client to call the LLM
            response = await baml_client.run(
                "ConflictAnalysis",
                prompt=prompt
            )
            
            # Parse the response
            if isinstance(response, list):
                return response
            
            # If the response contains a "options" key, return that
            if isinstance(response, dict) and "options" in response:
                return response["options"]
                
            # If the response contains a "recommendations" key, return that
            if isinstance(response, dict) and "recommendations" in response:
                return response["recommendations"]
                
            # Otherwise, wrap the response in a list
            return [response] if response else []
            
        except Exception as e:
            logger.error(f"Error calling LLM via BAML: {str(e)}")
            raise

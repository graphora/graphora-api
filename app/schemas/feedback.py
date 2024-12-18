from typing import Dict, Optional, List
from pydantic import BaseModel, Field

class PydanticFeedbackInput(BaseModel):
    """Pydantic model for REST API feedback validation"""
    feedback_type: str = Field(description="Type of feedback being provided")
    entity_updates: Optional[List[Dict]] = Field(default_factory=list, description="List of entity updates")
    relationship_updates: Optional[List[Dict]] = Field(default_factory=list, description="List of relationship updates")
    notes: Optional[str] = Field(default=None, description="Additional notes about the feedback")
    
    class Config:
        json_schema_extra = {
            "example": {
                "feedback_type": "entity_correction",
                "entity_updates": [{
                    "id": "123",
                    "type": "PERSON",
                    "value": "John Smith"
                }],
                "relationship_updates": [],
                "notes": "Correcting entity name"
            }
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for service layer"""
        return {
            "feedback_type": self.feedback_type,
            "entity_updates": self.entity_updates,
            "relationship_updates": self.relationship_updates,
            "notes": self.notes
        }
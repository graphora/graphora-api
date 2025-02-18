import pytest
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from pydantic import BaseModel

from app.services.llm.client import (
    call_llm_gemini,
    create_extraction_prompt,
)

class TestEntity(BaseModel):
    """Test entity for extraction evaluation"""
    name: str
    age: int
    email: str
    description: str | None = None

class Person(BaseModel):
    """Person entity for real-world testing"""
    name: str
    age: int | None = None
    occupation: str | None = None
    location: str | None = None
    education: List[str] | None = None

class Company(BaseModel):
    """Company entity for real-world testing"""
    name: str
    industry: str | None = None
    founded_year: int | None = None
    location: str | None = None
    employees: int | None = None

@pytest.fixture
def test_models():
    """Test models for extraction"""
    return {
        "TestEntity": TestEntity,
        "Person": Person,
        "Company": Company
    }

@pytest.fixture
def test_data():
    """Test data with ground truth annotations"""
    return [
        {
            "text": """
            John Smith is a 35-year-old software engineer living in San Francisco.
            He graduated from MIT and Stanford University.
            He works at TechCorp, a leading AI company founded in 2010 with 500 employees
            in the technology sector.
            """,
            "ground_truth": {
                "Person": [{
                    "name": "John Smith",
                    "age": 35,
                    "occupation": "software engineer",
                    "location": "San Francisco",
                    "education": ["MIT", "Stanford University"]
                }],
                "Company": [{
                    "name": "TechCorp",
                    "industry": "technology",
                    "founded_year": 2010,
                    "employees": 500
                }]
            }
        },
        # Add more test cases...
    ]

@pytest.fixture
def llm_configs():
    """Different LLM configurations to test"""
    return [
        {
            "name": "gemini-pro",
            "provider": "google",
            "temperature": 0.1
        },
        # Add more LLM configs...
    ]

class ExtractionMetrics:
    """Metrics for evaluating extraction quality"""
    
    def __init__(self):
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.latency_ms = 0.0
        self.success_rate = 0.0
        self.error_rate = 0.0
        self.token_usage = 0
        self.extraction_count = 0
        self.field_accuracy = {}
        self.confidence_correlation = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "latency_ms": self.latency_ms,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "token_usage": self.token_usage,
            "extraction_count": self.extraction_count,
            "field_accuracy": self.field_accuracy,
            "confidence_correlation": self.confidence_correlation
        }

def calculate_field_match(
    extracted: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> float:
    """Calculate field-level match score"""
    if not extracted or not ground_truth:
        return 0.0
    
    matches = 0
    total_fields = len(ground_truth)
    
    for field, truth_value in ground_truth.items():
        if field not in extracted:
            continue
            
        extracted_value = extracted[field]
        
        # Handle different value types
        if isinstance(truth_value, (int, float)):
            # Numeric tolerance
            matches += abs(truth_value - extracted_value) <= 0.1
        elif isinstance(truth_value, list):
            # List overlap
            truth_set = set(truth_value)
            extracted_set = set(extracted_value)
            overlap = len(truth_set.intersection(extracted_set))
            matches += overlap / max(len(truth_set), len(extracted_set))
        else:
            # String similarity
            matches += truth_value.lower() == str(extracted_value).lower()
    
    return matches / total_fields

async def evaluate_extraction(
    extractor_fn,
    test_data: List[Dict[str, Any]],
    models: Dict[str, BaseModel]
) -> ExtractionMetrics:
    """Evaluate extraction quality and performance"""
    metrics = ExtractionMetrics()
    start_time = datetime.now()
    
    try:
        for test_case in test_data:
            text = test_case["text"]
            ground_truth = test_case["ground_truth"]
            
            # Perform extraction
            extraction_start = datetime.now()
            try:
                extraction = await extractor_fn(text, models)
                metrics.extraction_count += 1
                metrics.success_rate = (
                    metrics.extraction_count / len(test_data)
                )
            except Exception as e:
                metrics.error_rate += 1 / len(test_data)
                continue
            
            # Calculate latency
            latency = (
                datetime.now() - extraction_start
            ).total_seconds() * 1000
            metrics.latency_ms = (
                (metrics.latency_ms * (metrics.extraction_count - 1) + latency) /
                metrics.extraction_count
            )
            
            # Calculate accuracy metrics
            for entity_type, truth_instances in ground_truth.items():
                if entity_type not in extraction:
                    continue
                
                extracted_instances = extraction[entity_type]
                
                # Match instances
                matched_pairs = []
                for truth in truth_instances:
                    best_match = max(
                        extracted_instances,
                        key=lambda x: calculate_field_match(
                            x.properties,
                            truth
                        ),
                        default=None
                    )
                    if best_match:
                        matched_pairs.append((truth, best_match))
                
                # Calculate metrics
                true_positives = len(matched_pairs)
                false_positives = len(extracted_instances) - true_positives
                false_negatives = len(truth_instances) - true_positives
                
                # Update precision and recall
                precision = (
                    true_positives / 
                    (true_positives + false_positives)
                    if true_positives + false_positives > 0
                    else 0
                )
                recall = (
                    true_positives /
                    (true_positives + false_negatives)
                    if true_positives + false_negatives > 0
                    else 0
                )
                
                # Calculate field-level accuracy
                for truth, extracted in matched_pairs:
                    field_scores = {}
                    for field in truth:
                        if field in extracted.properties:
                            match_score = calculate_field_match(
                                {field: extracted.properties[field]},
                                {field: truth[field]}
                            )
                            field_scores[field] = match_score
                    
                    # Update field accuracy
                    for field, score in field_scores.items():
                        if field not in metrics.field_accuracy:
                            metrics.field_accuracy[field] = []
                        metrics.field_accuracy[field].append(score)
                
                # Calculate confidence correlation
                if matched_pairs:
                    truth_scores = [
                        calculate_field_match(
                            extracted.properties,
                            truth
                        )
                        for truth, extracted in matched_pairs
                    ]
                    confidence_scores = [
                        extracted.confidence_score
                        for _, extracted in matched_pairs
                    ]
                    
                    metrics.confidence_correlation = np.corrcoef(
                        truth_scores,
                        confidence_scores
                    )[0, 1]
                
                # Update overall metrics
                metrics.precision = (
                    metrics.precision * (metrics.extraction_count - 1) +
                    precision
                ) / metrics.extraction_count
                
                metrics.recall = (
                    metrics.recall * (metrics.extraction_count - 1) +
                    recall
                ) / metrics.extraction_count
                
                if metrics.precision + metrics.recall > 0:
                    metrics.f1_score = (
                        2 * metrics.precision * metrics.recall /
                        (metrics.precision + metrics.recall)
                    )
    
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        raise
    
    # Average field accuracy
    metrics.field_accuracy = {
        field: sum(scores) / len(scores)
        for field, scores in metrics.field_accuracy.items()
    }
    
    return metrics

async def test_extraction_quality(test_data, test_models):
    """Test extraction quality with different LLMs"""
    metrics = await evaluate_extraction(
        call_llm_gemini,
        test_data,
        test_models
    )
    
    # Quality checks
    assert metrics.precision >= 0.7, "Precision below threshold"
    assert metrics.recall >= 0.7, "Recall below threshold"
    assert metrics.f1_score >= 0.7, "F1 score below threshold"
    assert metrics.success_rate >= 0.9, "Success rate below threshold"
    assert metrics.error_rate <= 0.1, "Error rate above threshold"
    
    # Performance checks
    assert metrics.latency_ms <= 5000, "Latency above threshold"
    
    # Field accuracy checks
    for field, accuracy in metrics.field_accuracy.items():
        assert accuracy >= 0.7, f"Field accuracy below threshold: {field}"
    
    # Confidence correlation check
    assert metrics.confidence_correlation >= 0.5, (
        "Low confidence correlation"
    )

async def test_extraction_edge_cases(test_models):
    """Test extraction with edge cases"""
    edge_cases = [
        # Empty text
        "",
        # Very short text
        "John",
        # Very long text
        "a" * 10000,
        # Special characters
        "John@#$%^&* Smith is 35 years old",
        # Multiple entities
        """
        John Smith (35) and Jane Doe (28) both work at TechCorp.
        Another company, DataCorp, was founded in 2015.
        """,
        # Ambiguous information
        "Someone who might be John or Jim works somewhere",
        # Nested information
        """
        TechCorp, founded by John Smith (who previously worked at
        DataCorp from 2010 to 2015), is growing rapidly.
        """
    ]
    
    for text in edge_cases:
        extraction = await call_llm_gemini(text, test_models)
        
        # Validate extraction
        assert isinstance(extraction, dict), "Invalid extraction format"
        for entity_type, instances in extraction.items():
            assert entity_type in test_models, "Unknown entity type"
            assert isinstance(instances, list), "Invalid instances format"

async def test_prompt_generation(test_models):
    """Test prompt generation"""
    prompt = create_extraction_prompt(
        "Sample text",
        test_models
    )
    
    # Validate prompt structure
    assert "Entity Definitions" in prompt
    assert "Rules" in prompt
    assert "Text" in prompt
    
    # Check entity definitions
    for model_name in test_models:
        assert model_name in prompt
        model = test_models[model_name]
        for field in model.model_fields:
            if field not in ['id', 'type', 'provenance']:
                assert field in prompt

if __name__ == "__main__":
    pytest.main([__file__])

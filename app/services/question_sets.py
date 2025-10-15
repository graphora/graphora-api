from typing import List
from app.schemas.schema import QuestionSet, Question, QuestionType

# Configurable question sets for schema generation
QUESTION_SETS: List[QuestionSet] = [
    QuestionSet(
        id="basic_info",
        title="Basic Information",
        description="Understanding your data and use case",
        questions=[
            Question(
                id="use_case",
                type=QuestionType.TEXTAREA,
                prompt="What is your primary use case for this knowledge graph?",
                required=True,
                placeholder="e.g., Analyzing customer relationships, tracking research citations, mapping business processes...",
                help_text="Describe what insights you want to gain from your graph",
                validation={"min_length": 10, "max_length": 1000},
            ),
            Question(
                id="data_sources",
                type=QuestionType.TEXTAREA,
                prompt="What types of data will you be processing?",
                required=True,
                placeholder="e.g., Customer records, financial reports, research papers, emails...",
                help_text="List the main data sources and document types",
                validation={"min_length": 10, "max_length": 1000},
            ),
            Question(
                id="domain",
                type=QuestionType.SELECT,
                prompt="Which domain best describes your use case?",
                required=True,
                options=[
                    "Business/Enterprise",
                    "Healthcare/Medical",
                    "Financial/Banking",
                    "Research/Academic",
                    "Legal/Compliance",
                    "Technology/Software",
                    "E-commerce/Retail",
                    "Manufacturing/Supply Chain",
                    "Government/Public Sector",
                    "Other",
                ],
            ),
        ],
    ),
    QuestionSet(
        id="graph_requirements",
        title="Graph Requirements",
        description="Understanding your graph structure needs",
        questions=[
            Question(
                id="key_entities",
                type=QuestionType.TEXTAREA,
                prompt="What are the main entities (people, objects, concepts) in your data?",
                required=True,
                placeholder="e.g., Customer, Product, Order, Company, Person, Document...",
                help_text="List the main things you want to represent as nodes",
                validation={"min_length": 10, "max_length": 1000},
            ),
            Question(
                id="relationships",
                type=QuestionType.TEXTAREA,
                prompt="What are the key relationships between these entities?",
                required=True,
                placeholder="e.g., Customer PURCHASED Product, Person WORKS_FOR Company, Document CITES Document...",
                help_text="Describe how your entities connect to each other",
                validation={"min_length": 10, "max_length": 1000},
            ),
            Question(
                id="query_patterns",
                type=QuestionType.TEXTAREA,
                prompt="What types of questions will you ask your graph?",
                required=True,
                placeholder="e.g., Which customers bought similar products? How are these companies connected? What documents cite this research?",
                help_text="Examples of queries you plan to run",
                validation={"min_length": 10, "max_length": 1000},
            ),
        ],
    ),
    QuestionSet(
        id="data_specifics",
        title="Data Specifics",
        description="Details about your data characteristics",
        questions=[
            Question(
                id="data_volume",
                type=QuestionType.SELECT,
                prompt="What is the approximate scale of your data?",
                required=True,
                options=[
                    "Small (< 1K records)",
                    "Medium (1K - 100K records)",
                    "Large (100K - 1M records)",
                    "Very Large (> 1M records)",
                ],
            ),
            Question(
                id="data_complexity",
                type=QuestionType.SELECT,
                prompt="How complex is your data structure?",
                required=True,
                options=[
                    "Simple (Few entity types, basic relationships)",
                    "Moderate (Multiple entity types, some complex relationships)",
                    "Complex (Many entity types, hierarchical relationships, metadata)",
                    "Very Complex (Highly interconnected, temporal data, multiple contexts)",
                ],
            ),
            Question(
                id="temporal_requirements",
                type=QuestionType.SELECT,
                prompt="Do you need to track changes over time?",
                required=True,
                options=[
                    "No temporal tracking needed",
                    "Basic timestamps (created/updated)",
                    "Version tracking (track all changes)",
                    "Time-series data (events over time)",
                    "Bi-temporal (valid time + transaction time)",
                ],
            ),
            Question(
                id="sample_data",
                type=QuestionType.FILE,
                prompt="Upload a sample of your data (optional)",
                required=False,
                help_text="This helps us understand your data format and structure",
            ),
        ],
    ),
]


def get_question_sets_for_domain(
    domain: str = None, include_optional: bool = True
) -> List[QuestionSet]:
    """Get question sets filtered by domain and optional settings"""

    # For now, return all question sets
    # TODO: Implement domain-specific filtering when needed
    question_sets = QUESTION_SETS.copy()

    if not include_optional:
        # Filter out optional questions
        filtered_sets = []
        for question_set in question_sets:
            filtered_questions = [q for q in question_set.questions if q.required]
            if filtered_questions:
                filtered_set = question_set.model_copy()
                filtered_set.questions = filtered_questions
                filtered_sets.append(filtered_set)
        question_sets = filtered_sets

    return question_sets


def get_question_by_id(question_id: str) -> Question | None:
    """Get a specific question by its ID"""

    for question_set in QUESTION_SETS:
        for question in question_set.questions:
            if question.id == question_id:
                return question

    return None


def validate_user_responses(responses: List[dict]) -> List[str]:
    """Validate user responses against question requirements"""

    errors = []

    for response in responses:
        question_id = response.get("question_id")
        value = response.get("value")

        question = get_question_by_id(question_id)
        if not question:
            errors.append(f"Unknown question ID: {question_id}")
            continue

        # Check required fields
        if question.required and (
            not value or (isinstance(value, str) and not value.strip())
        ):
            errors.append(f"Question '{question.prompt}' is required")
            continue

        # Validate based on question type and validation rules
        if question.validation and value:
            validation = question.validation

            if isinstance(value, str):
                if (
                    validation.get("min_length")
                    and len(value) < validation["min_length"]
                ):
                    errors.append(
                        f"Answer for '{question.prompt}' is too short (minimum {validation['min_length']} characters)"
                    )

                if (
                    validation.get("max_length")
                    and len(value) > validation["max_length"]
                ):
                    errors.append(
                        f"Answer for '{question.prompt}' is too long (maximum {validation['max_length']} characters)"
                    )

                if validation.get("pattern"):
                    import re

                    if not re.match(validation["pattern"], value):
                        errors.append(
                            f"Answer for '{question.prompt}' does not match required format"
                        )

        # Validate select options
        if (
            question.type in [QuestionType.SELECT, QuestionType.MULTISELECT]
            and question.options
        ):
            if question.type == QuestionType.SELECT:
                if value not in question.options:
                    errors.append(f"Invalid option selected for '{question.prompt}'")
            elif question.type == QuestionType.MULTISELECT:
                if isinstance(value, list):
                    invalid_options = [v for v in value if v not in question.options]
                    if invalid_options:
                        errors.append(
                            f"Invalid options selected for '{question.prompt}': {invalid_options}"
                        )
                else:
                    errors.append(
                        f"Multiple selection expected for '{question.prompt}'"
                    )

    return errors

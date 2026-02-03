"""Service for managing ontology templates."""

from pathlib import Path
from typing import List, Optional
import yaml

from pydantic import BaseModel

from app.utils.logger import logger


class OntologyTemplate(BaseModel):
    """Ontology template metadata."""

    id: str
    name: str
    description: str
    entities: List[str]
    relationships: List[str]
    use_cases: List[str]


class OntologyTemplateDetail(OntologyTemplate):
    """Full ontology template with content."""

    content: str


# Map template IDs to use case descriptions
TEMPLATE_USE_CASES = {
    "company_person": [
        "Corporate filings and SEC documents",
        "Organizational charts",
        "Business news articles",
        "LinkedIn profiles and resumes",
    ],
    "product_catalog": [
        "E-commerce product listings",
        "Inventory documentation",
        "Product specifications",
        "Supplier catalogs",
    ],
    "research_papers": [
        "Academic papers and journals",
        "Literature reviews",
        "Research proposals",
        "Conference proceedings",
    ],
    "legal_contracts": [
        "NDAs and service agreements",
        "Employment contracts",
        "Licensing agreements",
        "Terms of service",
    ],
    "financial_analysis": [
        "SEC 10-K and 10-Q filings",
        "Earnings reports",
        "Investment research",
        "Risk assessments",
    ],
}


class OntologyTemplateService:
    """Service for managing ontology templates."""

    def __init__(self):
        self.templates_dir = Path(__file__).parent.parent / "templates" / "ontologies"

    def list_templates(self) -> List[OntologyTemplate]:
        """List all available ontology templates."""
        templates = []

        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return templates

        for file_path in self.templates_dir.glob("*.yaml"):
            try:
                template = self._parse_template_metadata(file_path)
                if template:
                    templates.append(template)
            except Exception as e:
                logger.error(f"Error parsing template {file_path}: {e}")

        return sorted(templates, key=lambda t: t.name)

    def get_template(self, template_id: str) -> Optional[OntologyTemplateDetail]:
        """Get a specific template by ID."""
        file_path = self.templates_dir / f"{template_id}.yaml"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                content = f.read()
                data = yaml.safe_load(content)

            entities = list(data.get("entities", {}).keys())
            relationships = self._extract_relationships(data)

            return OntologyTemplateDetail(
                id=template_id,
                name=data.get("name", template_id),
                description=data.get("description", ""),
                entities=entities,
                relationships=relationships,
                use_cases=TEMPLATE_USE_CASES.get(template_id, []),
                content=content,
            )
        except Exception as e:
            logger.error(f"Error loading template {template_id}: {e}")
            return None

    def get_template_content(self, template_id: str) -> Optional[str]:
        """Get the raw YAML content of a template."""
        file_path = self.templates_dir / f"{template_id}.yaml"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading template {template_id}: {e}")
            return None

    def _parse_template_metadata(self, file_path: Path) -> Optional[OntologyTemplate]:
        """Parse template metadata from a YAML file."""
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            template_id = file_path.stem
            entities = list(data.get("entities", {}).keys())
            relationships = self._extract_relationships(data)

            return OntologyTemplate(
                id=template_id,
                name=data.get("name", template_id),
                description=data.get("description", ""),
                entities=entities,
                relationships=relationships,
                use_cases=TEMPLATE_USE_CASES.get(template_id, []),
            )
        except Exception as e:
            logger.error(f"Error parsing template metadata from {file_path}: {e}")
            return None

    def _extract_relationships(self, data: dict) -> List[str]:
        """Extract relationship names from ontology data."""
        relationships = set()

        for entity_name, entity_data in data.get("entities", {}).items():
            if isinstance(entity_data, dict):
                for rel_name in entity_data.get("relationships", {}).keys():
                    relationships.add(rel_name)

        return sorted(list(relationships))


# Singleton instance
ontology_template_service = OntologyTemplateService()

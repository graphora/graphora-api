import yaml
from pydantic import BaseModel

class OntologyParser:
    async def parse(self, ontology_text: str) -> dict:
        """Parse ontology into standardized structure"""
        return await self._parse_plaintext(ontology_text)

    async def _parse_plaintext(self, ontology_text: str) -> dict:
        return dict(**yaml.safe_load(ontology_text))
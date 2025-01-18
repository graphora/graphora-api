import yaml

class KnowledgeGraphYAMLExporter:
   @staticmethod
   def to_yaml(kg) -> str:

       def _model_to_dict(obj):
           if isinstance(obj, BaseModel):
               data = obj.model_dump(by_alias=True)
               return {k: _model_to_dict(v) for k, v in data.items()}
           elif isinstance(obj, list):
               return [_model_to_dict(item) for item in obj]
           elif isinstance(obj, dict):
               return {k: _model_to_dict(v) for k, v in obj.items()}
           return obj

       kg_dict = _model_to_dict(kg)
       return yaml.dump(kg_dict, sort_keys=False, allow_unicode=True)
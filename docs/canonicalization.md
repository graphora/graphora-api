# Canonicalisation Metadata

To tune entity resolution for different domains, you can specify per-property canonicalisation hints inside your ontology YAML under each property definition. Example:

```yaml
entities:
  Company:
    properties:
      name:
        type: string
        unique: true
        canonicalization:
          strip_punctuation: true
          strip_suffixes: ["Inc", "LLC", "Ltda"]
          preserve_case: false
      ticker:
        type: string
        unique: true
      founded_on:
        type: datetime
        canonicalization:
          preserve_case: true
```

Supported options:
- `strip_punctuation` (`bool`): remove punctuation characters.
- `remove_non_alnum` (`bool`): keep letters / digits only.
- `strip_suffixes` (`list[str]`): explicit suffixes to remove.
- `strip_company_suffixes` (`bool`): reuse built‑in company suffix list.
- `preserve_case` (`bool`): skip lower-casing.

Additional options and custom canonicalisers can be plugged in by calling `register_canonicalizer("Entity.property", func)` during server startup.

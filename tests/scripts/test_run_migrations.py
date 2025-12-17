from pathlib import Path

from scripts.run_migrations import _migration_sort_key


def test_migration_sort_key_orders_versions_numerically() -> None:
    migrations = [
        "10_add_doc_usage_error_message.sql",
        "11_add_created_at_to_change_logs.sql",
        "1_db_setup.sql",
        "2_config_schema.sql",
        "3_gemini_provider_schema.sql",
        "3_usage_tracking_schema.sql",
        "4_model_pricing_schema.sql",
        "5_create_schema_tables.sql",
        "5_quality_feedback_schema.sql",
        "6_refresh_gemini_models.sql",
        "7_add_session_id_to_document_usage.sql",
        "8_add_merge_metadata.sql",
        "9_create_entity_ledger.sql",
    ]

    sorted_migrations = sorted(map(Path, migrations), key=_migration_sort_key)
    sorted_names = [path.name for path in sorted_migrations]

    assert sorted_names == [
        "1_db_setup.sql",
        "2_config_schema.sql",
        "3_gemini_provider_schema.sql",
        "3_usage_tracking_schema.sql",
        "4_model_pricing_schema.sql",
        "5_create_schema_tables.sql",
        "5_quality_feedback_schema.sql",
        "6_refresh_gemini_models.sql",
        "7_add_session_id_to_document_usage.sql",
        "8_add_merge_metadata.sql",
        "9_create_entity_ledger.sql",
        "10_add_doc_usage_error_message.sql",
        "11_add_created_at_to_change_logs.sql",
    ]

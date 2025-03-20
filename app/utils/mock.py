transform_id = "20250206201729"
transform_graph = {"nodes":[{"id":"03613df7-734e-49fc-aea9-f397121c03a5","label":"BusinessSegment","properties":{"_merged_ids":"BusinessSegment_CrudeOilMarketing","name":"Crude Oil Marketing","_uid_":"03613df7-734e-49fc-aea9-f397121c03a5"},"type":"BusinessSegment"},{"id":"155a216b-580f-4774-9cd5-e2cc35193e93","label":"RawMaterial","properties":{"_merged_ids":"RawMaterial_CrudeOil","name":"Crude Oil","_uid_":"155a216b-580f-4774-9cd5-e2cc35193e93"},"type":"RawMaterial"},{"id":"40e8fe81-eec8-4b86-b680-0de4e5aa8fef","label":"RiskFactor","properties":{"_merged_ids":"cybersecurity_risk","name":"Cybersecurity Risk","description":"Potential cyber threats and their impacts on the organization","_uid_":"40e8fe81-eec8-4b86-b680-0de4e5aa8fef","potential_impact":"Compromise of system availability and security"},"type":"RiskFactor"},{"id":"47e535ba-aef8-401f-8007-356b560d710e","label":"PartI","properties":{"_merged_ids":"PartI,PartI_RiskFactors","_uid_":"47e535ba-aef8-401f-8007-356b560d710e"},"type":"PartI"},{"id":"4fe2c48d-5501-48a8-a8e3-3f7e22eb18c5","label":"RiskCategory","properties":{"_merged_ids":"risk_category_operational","name":"Operational Risks","_uid_":"4fe2c48d-5501-48a8-a8e3-3f7e22eb18c5"},"type":"RiskCategory"},{"id":"6af525b8-4e80-4784-b83e-e18ccd891a7c","label":"RiskFactor","properties":{"_merged_ids":"driver_attraction_risk","name":"Driver Attraction Risk","description":"Difficulty in attracting and retaining drivers","_uid_":"6af525b8-4e80-4784-b83e-e18ccd891a7c","potential_impact":"Could negatively affect operations and limit growth"},"type":"RiskFactor"},{"id":"933853a6-1066-4571-b8e8-12d775187d46","label":"Product","properties":{"_merged_ids":"Product_CrudeOil","name":"Crude Oil","_uid_":"933853a6-1066-4571-b8e8-12d775187d46"},"type":"Product"},{"id":"96b486ac-cff6-4188-8396-b8797b0f1a97","label":"Company","properties":{"cik":"<UNKNOWN>","name":"Adams Resources & Energy, Inc.","_uid_":"96b486ac-cff6-4188-8396-b8797b0f1a97"},"type":"Company"},{"id":"9c545335-4c0f-4cdb-9735-da2808070731","label":"Metadata","properties":{"filingDate":"<UNKNOWN>","name":"Form13_2023","about":"Form 13 Annual report for Adams Resources & Energy, Inc.","context":"Annual report describing business operations of Adams Resources & Energy, Inc. in crude oil marketing, transportation, and related services","type":"financial_report","_uid_":"9c545335-4c0f-4cdb-9735-da2808070731"},"type":"Metadata"},{"id":"aa411001-62ea-4123-aa9d-f627c72b93c4","label":"Business","properties":{"_merged_ids":"Business","regulatory_environment":"Regulated by FERC, DOT, and PHMSA","description":"Adams Resources  Energy, Inc. operates in crude oil marketing, transportation, and related services across the lower 48 states of the United States","employees":741,"seasonality":"Not explicitly mentioned","_uid_":"aa411001-62ea-4123-aa9d-f627c72b93c4"},"type":"Business"},{"id":"cbf92efe-5126-436b-9134-aef503708483","label":"Company","properties":{"_merged_ids":"Company_AdamsResourcesAndEnergy","name":"Adams Resources  Energy, Inc.","_uid_":"cbf92efe-5126-436b-9134-aef503708483"},"type":"Company"},{"id":"e2532f8b-5ce8-409c-bf17-518ec407eb3f","label":"RiskCategory","properties":{"_merged_ids":"risk_category_technology","name":"Technology Risks","_uid_":"e2532f8b-5ce8-409c-bf17-518ec407eb3f"},"type":"RiskCategory"}],"edges":[{"id":"e162cb30-904a-426e-8594-0f501b708c95","source":"aa411001-62ea-4123-aa9d-f627c72b93c4","target":"03613df7-734e-49fc-aea9-f397121c03a5","type":"HAS_SEGMENT","properties":{"_uid_":"e162cb30-904a-426e-8594-0f501b708c95"}},{"id":"dc0131bc-618f-4cc7-ba66-db70a3832e96","source":"aa411001-62ea-4123-aa9d-f627c72b93c4","target":"155a216b-580f-4774-9cd5-e2cc35193e93","type":"HAS_RAW_MATERIAL","properties":{"_uid_":"dc0131bc-618f-4cc7-ba66-db70a3832e96"}},{"id":"7e862b51-535e-439c-91db-d890128938ba","source":"47e535ba-aef8-401f-8007-356b560d710e","target":"40e8fe81-eec8-4b86-b680-0de4e5aa8fef","type":"HAS_RISK_FACTOR","properties":{"_uid_":"7e862b51-535e-439c-91db-d890128938ba"}},{"id":"0dba467e-4b2d-48c1-8987-f680a891eb21","source":"40e8fe81-eec8-4b86-b680-0de4e5aa8fef","target":"e2532f8b-5ce8-409c-bf17-518ec407eb3f","type":"HAS_RISK_CATEGORY","properties":{"_uid_":"0dba467e-4b2d-48c1-8987-f680a891eb21"}},{"id":"17f43c1e-9879-44d0-9192-60f8b1ff74a1","source":"47e535ba-aef8-401f-8007-356b560d710e","target":"aa411001-62ea-4123-aa9d-f627c72b93c4","type":"HAS_BUSINESS","properties":{"_uid_":"17f43c1e-9879-44d0-9192-60f8b1ff74a1"}},{"id":"c0943e76-8883-4ebd-b7a8-94afd16db90d","source":"47e535ba-aef8-401f-8007-356b560d710e","target":"6af525b8-4e80-4784-b83e-e18ccd891a7c","type":"HAS_RISK_FACTOR","properties":{"_uid_":"c0943e76-8883-4ebd-b7a8-94afd16db90d"}},{"id":"458197c0-c51b-410f-aada-bd836e53b2b9","source":"9c545335-4c0f-4cdb-9735-da2808070731","target":"47e535ba-aef8-401f-8007-356b560d710e","type":"HAS","properties":{"section":"PartI","_uid_":"458197c0-c51b-410f-aada-bd836e53b2b9"}},{"id":"e6f4916c-d885-4174-baed-674bf42ac446","source":"6af525b8-4e80-4784-b83e-e18ccd891a7c","target":"4fe2c48d-5501-48a8-a8e3-3f7e22eb18c5","type":"HAS_RISK_CATEGORY","properties":{"_uid_":"e6f4916c-d885-4174-baed-674bf42ac446"}},{"id":"1951d7cb-cdd4-40a7-af65-9e81ed592410","source":"aa411001-62ea-4123-aa9d-f627c72b93c4","target":"933853a6-1066-4571-b8e8-12d775187d46","type":"HAS_PRODUCT","properties":{"_uid_":"1951d7cb-cdd4-40a7-af65-9e81ed592410"}},{"id":"9be60adc-8590-4a0e-ac92-6df6da722885","source":"9c545335-4c0f-4cdb-9735-da2808070731","target":"96b486ac-cff6-4188-8396-b8797b0f1a97","type":"ABOUT_COMPANY","properties":{"_uid_":"9be60adc-8590-4a0e-ac92-6df6da722885"}},{"id":"76a8c710-eb98-4846-8685-e74a6343e0a0","source":"9c545335-4c0f-4cdb-9735-da2808070731","target":"9c545335-4c0f-4cdb-9735-da2808070731","type":"HAS","properties":{"section":"Metadata","_uid_":"76a8c710-eb98-4846-8685-e74a6343e0a0"}},{"id":"499dc1c7-808e-4c67-aee7-a3db5423eb63","source":"aa411001-62ea-4123-aa9d-f627c72b93c4","target":"cbf92efe-5126-436b-9134-aef503708483","type":"HAS_COMPETITION","properties":{"_uid_":"499dc1c7-808e-4c67-aee7-a3db5423eb63"}}],"total_nodes":13,"total_edges":12}

def get_mock_subgraph(transform_id: str = transform_id):
  nodes, edges = transform_graph['nodes'], transform_graph['edges']
  for node in nodes:
    node['labels'] = [node['label'], f'Staging_{transform_id}']
  for edge in edges:
    edge['source_id'] = edge['source']
    edge['target_id'] = edge['target']
  return nodes, edges

merge_graph = {
    "status": "success",
    "data": {
        "nodes": [
            {
                "id": "03613df7-734e-49fc-aea9-f397121c03a5",
                "labels": [
                    "BusinessSegment",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "_merged_ids": "BusinessSegment_CrudeOilMarketing",
                    "name": "Crude Oil Marketing",
                    "_uid_": "03613df7-734e-49fc-aea9-f397121c03a5",
                    "__status": "needs_review",
                    "__type": "BusinessSegment"
                },
                "status": "needs_review"
            },
            {
                "id": "155a216b-580f-4774-9cd5-e2cc35193e93",
                "labels": [
                    "RawMaterial",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "_merged_ids": "RawMaterial_CrudeOil",
                    "name": "Crude Oil",
                    "_uid_": "155a216b-580f-4774-9cd5-e2cc35193e93",
                    "__status": "needs_review",
                    "__type": "RawMaterial"
                },
                "status": "needs_review"
            },
            {
                "id": "40e8fe81-eec8-4b86-b680-0de4e5aa8fef",
                "labels": [
                    "RiskFactor",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "_merged_ids": "cybersecurity_risk",
                    "name": "Cybersecurity Risk",
                    "description": "Potential cyber threats and their impacts on the organization",
                    "_uid_": "40e8fe81-eec8-4b86-b680-0de4e5aa8fef",
                    "potential_impact": "Compromise of system availability and security",
                    "__status": "needs_review",
                    "__type": "RiskFactor"
                },
                "status": "needs_review"
            },
            {
                "id": "47e535ba-aef8-401f-8007-356b560d710e",
                "labels": [
                    "PartI",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "_merged_ids": "PartI,PartI_RiskFactors",
                    "_uid_": "47e535ba-aef8-401f-8007-356b560d710e",
                    "__status": "needs_review",
                    "__type": "PartI"
                },
                "status": "needs_review"
            },
            {
                "id": "4fe2c48d-5501-48a8-a8e3-3f7e22eb18c5",
                "labels": [
                    "RiskCategory",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "_merged_ids": "risk_category_operational",
                    "name": "Operational Risks",
                    "_uid_": "4fe2c48d-5501-48a8-a8e3-3f7e22eb18c5",
                    "__status": "needs_review",
                    "__type": "RiskCategory"
                },
                "status": "needs_review"
            },
            {
                "id": "6af525b8-4e80-4784-b83e-e18ccd891a7c",
                "labels": [
                    "RiskFactor",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "_merged_ids": "driver_attraction_risk",
                    "name": "Driver Attraction Risk",
                    "description": "Difficulty in attracting and retaining drivers",
                    "_uid_": "6af525b8-4e80-4784-b83e-e18ccd891a7c",
                    "potential_impact": "Could negatively affect operations and limit growth",
                    "__status": "needs_review",
                    "__type": "RiskFactor"
                },
                "status": "needs_review"
            },
            {
                "id": "933853a6-1066-4571-b8e8-12d775187d46",
                "labels": [
                    "Product",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "_merged_ids": "Product_CrudeOil",
                    "name": "Crude Oil",
                    "_uid_": "933853a6-1066-4571-b8e8-12d775187d46",
                    "__status": "needs_review",
                    "__type": "Product"
                },
                "status": "needs_review"
            },
            {
                "id": "96b486ac-cff6-4188-8396-b8797b0f1a97",
                "labels": [
                    "Company",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "cik": "<UNKNOWN>",
                    "name": "Adams Resources & Energy, Inc.",
                    "_uid_": "96b486ac-cff6-4188-8396-b8797b0f1a97",
                    "__status": "needs_review",
                    "__type": "Company"
                },
                "status": "needs_review"
            },
            {
                "id": "9c545335-4c0f-4cdb-9735-da2808070731",
                "labels": [
                    "Metadata",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "filingDate": "<UNKNOWN>",
                    "name": "Form13_2023",
                    "about": "Form 13 Annual report for Adams Resources & Energy, Inc.",
                    "context": "Annual report describing business operations of Adams Resources & Energy, Inc. in crude oil marketing, transportation, and related services",
                    "type": "financial_report",
                    "_uid_": "9c545335-4c0f-4cdb-9735-da2808070731",
                    "__status": "needs_review",
                    "__type": "Metadata"
                },
                "status": "needs_review"
            },
            {
                "id": "aa411001-62ea-4123-aa9d-f627c72b93c4",
                "labels": [
                    "Business",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "_merged_ids": "Business",
                    "regulatory_environment": "Regulated by FERC, DOT, and PHMSA",
                    "description": "Adams Resources  Energy, Inc. operates in crude oil marketing, transportation, and related services across the lower 48 states of the United States",
                    "employees": 741,
                    "seasonality": "Not explicitly mentioned",
                    "_uid_": "aa411001-62ea-4123-aa9d-f627c72b93c4",
                    "__status": "needs_review",
                    "__type": "Business"
                },
                "status": "needs_review"
            },
            {
                "id": "cbf92efe-5126-436b-9134-aef503708483",
                "labels": [
                    "Company",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "_merged_ids": "Company_AdamsResourcesAndEnergy",
                    "name": "Adams Resources  Energy, Inc.",
                    "_uid_": "cbf92efe-5126-436b-9134-aef503708483",
                    "__status": "needs_review",
                    "__type": "Company"
                },
                "status": "needs_review"
            },
            {
                "id": "e2532f8b-5ce8-409c-bf17-518ec407eb3f",
                "labels": [
                    "RiskCategory",
                    "Staging_20250206201729"
                ],
                "properties": {
                    "_merged_ids": "risk_category_technology",
                    "name": "Technology Risks",
                    "_uid_": "e2532f8b-5ce8-409c-bf17-518ec407eb3f",
                    "__status": "needs_review",
                    "__type": "RiskCategory"
                },
                "status": "needs_review"
            }
        ],
        "edges": [
            {
                "source": "aa411001-62ea-4123-aa9d-f627c72b93c4",
                "target": "03613df7-734e-49fc-aea9-f397121c03a5",
                "type": "HAS_SEGMENT",
                "properties": {
                    "_uid_": "e162cb30-904a-426e-8594-0f501b708c95",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "aa411001-62ea-4123-aa9d-f627c72b93c4",
                "target": "155a216b-580f-4774-9cd5-e2cc35193e93",
                "type": "HAS_RAW_MATERIAL",
                "properties": {
                    "_uid_": "dc0131bc-618f-4cc7-ba66-db70a3832e96",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "47e535ba-aef8-401f-8007-356b560d710e",
                "target": "40e8fe81-eec8-4b86-b680-0de4e5aa8fef",
                "type": "HAS_RISK_FACTOR",
                "properties": {
                    "_uid_": "7e862b51-535e-439c-91db-d890128938ba",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "40e8fe81-eec8-4b86-b680-0de4e5aa8fef",
                "target": "e2532f8b-5ce8-409c-bf17-518ec407eb3f",
                "type": "HAS_RISK_CATEGORY",
                "properties": {
                    "_uid_": "0dba467e-4b2d-48c1-8987-f680a891eb21",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "47e535ba-aef8-401f-8007-356b560d710e",
                "target": "aa411001-62ea-4123-aa9d-f627c72b93c4",
                "type": "HAS_BUSINESS",
                "properties": {
                    "_uid_": "17f43c1e-9879-44d0-9192-60f8b1ff74a1",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "47e535ba-aef8-401f-8007-356b560d710e",
                "target": "6af525b8-4e80-4784-b83e-e18ccd891a7c",
                "type": "HAS_RISK_FACTOR",
                "properties": {
                    "_uid_": "c0943e76-8883-4ebd-b7a8-94afd16db90d",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "9c545335-4c0f-4cdb-9735-da2808070731",
                "target": "47e535ba-aef8-401f-8007-356b560d710e",
                "type": "HAS",
                "properties": {
                    "section": "PartI",
                    "_uid_": "458197c0-c51b-410f-aada-bd836e53b2b9",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "6af525b8-4e80-4784-b83e-e18ccd891a7c",
                "target": "4fe2c48d-5501-48a8-a8e3-3f7e22eb18c5",
                "type": "HAS_RISK_CATEGORY",
                "properties": {
                    "_uid_": "e6f4916c-d885-4174-baed-674bf42ac446",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "aa411001-62ea-4123-aa9d-f627c72b93c4",
                "target": "933853a6-1066-4571-b8e8-12d775187d46",
                "type": "HAS_PRODUCT",
                "properties": {
                    "_uid_": "1951d7cb-cdd4-40a7-af65-9e81ed592410",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "9c545335-4c0f-4cdb-9735-da2808070731",
                "target": "96b486ac-cff6-4188-8396-b8797b0f1a97",
                "type": "ABOUT_COMPANY",
                "properties": {
                    "_uid_": "9be60adc-8590-4a0e-ac92-6df6da722885",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "9c545335-4c0f-4cdb-9735-da2808070731",
                "target": "9c545335-4c0f-4cdb-9735-da2808070731",
                "type": "HAS",
                "properties": {
                    "section": "Metadata",
                    "_uid_": "76a8c710-eb98-4846-8685-e74a6343e0a0",
                    "__status": "new"
                },
                "status": "new"
            },
            {
                "source": "aa411001-62ea-4123-aa9d-f627c72b93c4",
                "target": "cbf92efe-5126-436b-9134-aef503708483",
                "type": "HAS_COMPETITION",
                "properties": {
                    "_uid_": "499dc1c7-808e-4c67-aee7-a3db5423eb63",
                    "__status": "new"
                },
                "status": "new"
            }
        ],
        "conflicts": [],
        "summary": {
            "total_nodes": 12,
            "new_nodes": 12,
            "updated_nodes": 0,
            "conflicts": 0,
            "status": {
                "new": 12,
                "resolved": 0,
                "needs_review": 0
            }
        }
    }
}
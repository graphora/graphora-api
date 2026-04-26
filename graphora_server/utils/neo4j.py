# DEPRECATED: These functions use hardcoded database settings
# Use UserDatabaseService.get_user_config(user_id) and Neo4jStorage instead

# def run_cypher_staging(query: str, **kwargs):
#   with GraphDatabase.driver(settings.STAGING_NEO4J_URI,
#                             auth=(settings.STAGING_NEO4J_USER, settings.STAGING_NEO4J_PASSWORD)) as driver:
#     driver.verify_connectivity()
#     with driver.session(database=settings.STAGING_NEO4J_DB) as session:
#       result = session.run(query, **kwargs).to_df()
#       return result


# def run_cypher_batch_staging(queries: List[str], **kwargs) -> List[str]:
#   results = []
#   for query in queries:
#     print(query)
#     res = run_cypher_staging(query, **kwargs)
#     results.append(res)
#   return results

# def run_cypher(query: str, **kwargs):
#   with GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)) as driver:
#     driver.verify_connectivity()
#     with driver.session(database=settings.NEO4J_DB) as session:
#       result = session.run(query, **kwargs).to_df()
#       return result


# def run_cypher_batch(queries: List[str], **kwargs) -> List[str]:
#   results = []
#   for query in queries:
#     print(query)
#     res = run_cypher(query, **kwargs)
#     results.append(res)
#   return results

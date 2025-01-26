from typing import List, Dict
from neo4j import GraphDatabase
import pandas as pd
from contextlib import contextmanager

class DBConnector:
    def __init__(self, NEO4J_HOST: str, NEO4J_PASSWORD: str, NEO4J_USER='neo4j', NEO4J_DB='neo4j') -> None:
        self.NEO4J_HOST = NEO4J_HOST
        self.NEO4J_USER = NEO4J_USER
        self.NEO4J_PASSWORD = NEO4J_PASSWORD
        self.NEO4J_DB = NEO4J_DB
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.NEO4J_HOST,
                auth=(self.NEO4J_USER, self.NEO4J_PASSWORD)
            )
        return self._driver

    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        self.driver.verify_connectivity()
        session = self.driver.session(database=self.NEO4J_DB)
        try:
            yield session
        finally:
            session.close()

    def session(self):
        return self.driver.session()

    def run_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame"""
        with self.get_session() as session:
            result = session.run(query, parameters=params or {})
            return result.to_df()

    def run_query_return_list(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute query and return results as list of dictionaries"""
        with self.get_session() as session:
            result = session.run(query, parameters=params or {})
            return [dict(record) for record in result]

    async def run_query_async(self, query: str, params: Dict = None) -> List[Dict]:
        """Async wrapper for query execution"""
        # Since neo4j-python-driver doesn't support async,
        # we'll wrap the synchronous call
        return self.run_query_return_list(query, params)

    def close(self):
        """Close the database connection"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
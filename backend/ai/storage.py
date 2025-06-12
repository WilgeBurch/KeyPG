import hashlib
import logging
from collections import defaultdict
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KeyPGStorage:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def batch_add_patterns(self, patterns):
        """
        patterns: список словарей с полями hash, data (list), text (str)
        """
        with self.driver.session() as session:
            session.write_transaction(self._batch_create_nodes, patterns)

    @staticmethod
    def _batch_create_nodes(tx, patterns):
        query = """
        UNWIND $patterns AS pat
        MERGE (p:Pattern {hash: pat.hash})
        SET p.data = pat.data, p.text = pat.text
        """
        tx.run(query, patterns=patterns)

    def batch_add_restore_keys(self, file_pattern_hashes):
        """
        file_pattern_hashes: список словарей вида {'file_id': int, 'pattern_hashes': [str, ...]}
        """
        with self.driver.session() as session:
            session.write_transaction(self._batch_save_restore_keys, file_pattern_hashes)

    @staticmethod
    def _batch_save_restore_keys(tx, file_pattern_hashes):
        # Сохраняем маршруты восстановления для файлов (Document) и связи NEXT между паттернами
        query = """
        UNWIND $file_pattern_hashes AS fileinfo
        MERGE (d:Document {id: fileinfo.file_id})
        SET d.pattern_hashes = fileinfo.pattern_hashes
        WITH fileinfo
        UNWIND range(0, size(fileinfo.pattern_hashes)-2) AS idx
          MATCH (p1:Pattern {hash: fileinfo.pattern_hashes[idx]})
          MATCH (p2:Pattern {hash: fileinfo.pattern_hashes[idx+1]})
          MERGE (p1)-[:NEXT_IN_DOC {document: fileinfo.file_id}]->(p2)
        """
        tx.run(query, file_pattern_hashes=file_pattern_hashes)
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
        with self.driver.session() as session:
            session.write_transaction(self._batch_create_nodes, patterns)

    @staticmethod
    def _batch_create_nodes(tx, patterns):
        query = """
        UNWIND $patterns AS pat
        MERGE (p:Pattern {hash: pat.hash})
        SET p.data = pat.data,
            p.text = pat.text,
            p.meta = pat.meta
        """
        tx.run(query, patterns=patterns)

    def batch_add_restore_keys(self, file_pattern_hashes):
        with self.driver.session() as session:
            session.write_transaction(self._batch_save_restore_keys, file_pattern_hashes)

    @staticmethod
    def _batch_save_restore_keys(tx, file_pattern_hashes):
        query = """
        UNWIND $file_pattern_hashes AS fileinfo
        MERGE (d:Document {id: fileinfo.file_id})
        SET d.pattern_hashes = fileinfo.pattern_hashes,
            d.meta = fileinfo.meta

        WITH fileinfo
        UNWIND range(0, size(fileinfo.pattern_hashes)-2) AS idx
            MATCH (p1:Pattern {hash: fileinfo.pattern_hashes[idx]})
            MATCH (p2:Pattern {hash: fileinfo.pattern_hashes[idx+1]})
            MERGE (p1)-[r:NEXT_IN_DOC {document: fileinfo.file_id}]->(p2)
            ON CREATE SET r.weight = 1, r.meta = {order: idx}
            ON MATCH SET r.weight = r.weight + 1
        """
        tx.run(query, file_pattern_hashes=file_pattern_hashes)

    def get_restore_path(self, file_id):
        with self.driver.session() as session:
            result = session.read_transaction(self._get_restore_path, file_id)
        return result

    @staticmethod
    def _get_restore_path(tx, file_id):
        query = """
        MATCH (d:Document {id: $file_id})
        UNWIND range(0, size(d.pattern_hashes)-1) AS idx
        MATCH (p:Pattern {hash: d.pattern_hashes[idx]})
        RETURN p.hash AS hash, p.data AS data, p.text AS text, p.meta AS meta, idx
        ORDER BY idx ASC
        """
        result = tx.run(query, file_id=file_id)
        return [dict(r) for r in result]

    def get_most_probable_path(self, start_hash, max_length=100):
        """
        Поиск наиболее вероятного (частого) пути восстановления документа из графа паттернов,
        начиная с указанного паттерна. Используется модифицированный алгоритм поиска путей с максимальным весом.
        """
        with self.driver.session() as session:
            result = session.read_transaction(self._get_most_probable_path, start_hash, max_length)
        return result

    @staticmethod
    def _get_most_probable_path(tx, start_hash, max_length):
        # Используем жадный алгоритм: на каждом шаге выбираем следующее ребро с максимальным весом.
        # Можно заменить на более сложный (например, динамическое программирование), если потребуется.
        query = """
        MATCH (start:Pattern {hash: $start_hash})
        WITH start
        CALL {
            WITH start
            MATCH path = (start)-[:NEXT_IN_DOC*1..$max_length]->(end)
            WITH path,
                 reduce(weight = 0, r in relationships(path) | weight + r.weight) AS total_weight,
                 [n IN nodes(path) | n.hash] AS hashes
            ORDER BY total_weight DESC, length(path) DESC
            RETURN hashes, total_weight
            LIMIT 1
        }
        RETURN hashes, total_weight
        """
        result = tx.run(query, start_hash=start_hash, max_length=max_length)
        record = result.single()
        if record:
            return {
                "hashes": record["hashes"],
                "weight": record["total_weight"]
            }
        return None

    def restore_document_by_most_probable_path(self, file_id, max_length=100):
        """
        Восстановить текст документа на основе наиболее вероятного маршрута паттернов
        (учитывается частотность связей в графе, а не только исходный порядок из конкретного файла).
        """
        # Получаем первый паттерн файла
        with self.driver.session() as session:
            doc = session.run(
                "MATCH (d:Document {id: $file_id}) RETURN d.pattern_hashes AS pattern_hashes", file_id=file_id
            ).single()
            if not doc or not doc["pattern_hashes"]:
                return None
            start_hash = doc["pattern_hashes"][0]

        # Находим наиболее вероятный путь
        path_result = self.get_most_probable_path(start_hash, max_length)
        if not path_result or not path_result["hashes"]:
            return None

        # Получаем тексты паттернов вдоль найденного пути
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $hashes AS h
                MATCH (p:Pattern {hash: h})
                RETURN p.text AS text, p.hash AS hash
                ORDER BY apoc.coll.indexOf($hashes, h)
                """, hashes=path_result["hashes"]
            )
            return [r["text"] for r in result if r["text"]]

    def analyze_graph_statistics(self):
        """
        Анализ статистики графа: частота паттернов, распределение весов, выявление наиболее центральных паттернов.
        Можно использовать для адаптации системы, визуализации, оптимизации.
        """
        with self.driver.session() as session:
            # Пример: топ-10 самых частых паттернов (по количеству входящих и исходящих связей)
            result = session.run("""
            MATCH (p:Pattern)
            OPTIONAL MATCH (p)-[r:NEXT_IN_DOC]->()
            WITH p, count(r) AS out_links
            OPTIONAL MATCH ()-[r2:NEXT_IN_DOC]->(p)
            WITH p, out_links, count(r2) AS in_links
            RETURN p.hash AS hash, out_links, in_links, p.meta AS meta
            ORDER BY out_links + in_links DESC
            LIMIT 10
            """)
            return [dict(r) for r in result]
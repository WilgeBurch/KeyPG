import hashlib
import logging
from collections import defaultdict
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KeyPGStorage:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.pattern_usage = defaultdict(int)
        self.hash_to_node = {}

    def close(self):
        self.driver.close()

    def add_pattern(self, tx, pattern, data):
        pattern_hash = hashlib.sha256(pattern.tobytes()).hexdigest()
        result = tx.run("MATCH (n:Pattern {hash: $hash}) RETURN n", hash=pattern_hash)
        node = result.single()
        if node:
            node_id = node[0].id
            self.pattern_usage[node_id] += 1
        else:
            result = tx.run(
                "CREATE (n:Pattern {hash: $hash, data: $data}) RETURN id(n)",
                hash=pattern_hash, data=data.tolist()
            )
            node_id = result.single()[0]
            self.hash_to_node[pattern_hash] = node_id
            self.pattern_usage[node_id] += 1
        return node_id

    def add_edge(self, tx, node1, node2):
        tx.run(
            "MATCH (a:Pattern), (b:Pattern) WHERE id(a) = $node1 AND id(b) = $node2 "
            "MERGE (a)-[:NEXT]->(b)",
            node1=node1, node2=node2
        )

    def optimize_graph(self, threshold=2):
        with self.driver.session() as session:
            nodes_to_remove = [node for node, count in self.pattern_usage.items() if count < threshold]
            for node in nodes_to_remove:
                session.run("MATCH (n:Pattern) WHERE id(n) = $node_id DETACH DELETE n", node_id=node)
                hash_value = next((h for h, n in self.hash_to_node.items() if n == node), None)
                if hash_value is not None:
                    del self.hash_to_node[hash_value]
                del self.pattern_usage[node]
            logging.info(f"Removed {len(nodes_to_remove)} nodes with usage < {threshold}")

    def adapt_to_usage(self):
        with self.driver.session() as session:
            sorted_nodes = sorted(self.pattern_usage.items(), key=lambda x: x[1], reverse=True)
            root_id = session.run("MERGE (r:Root) RETURN id(r)").single()[0]
            for node, _ in sorted_nodes[:len(sorted_nodes)//2]:
                session.run(
                    "MATCH (n:Pattern), (r:Root) WHERE id(n) = $node_id AND id(r) = $root_id "
                    "MERGE (r)-[:FREQUENT]->(n)",
                    node_id=node, root_id=root_id
                )
            logging.info("Adapted graph to frequent usage patterns")
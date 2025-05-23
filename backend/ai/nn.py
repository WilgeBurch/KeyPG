import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from neo4j import GraphDatabase
import hashlib
from collections import defaultdict
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Автоэнкодер для выявления паттернов
class KeyPGAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(KeyPGAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 2. Класс для управления графом в Neo4j
class KeyPGStorage:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.pattern_usage = defaultdict(int)
        self.hash_to_node = {}
    
    def close(self):
        self.driver.close()
    
    def add_pattern(self, tx, pattern, data):
        """Добавляет паттерн в Neo4j с контентным хешированием"""
        pattern_hash = hashlib.sha256(pattern.tobytes()).hexdigest()
        
        # Проверка существования узла
        result = tx.run("MATCH (n:Pattern {hash: $hash}) RETURN n", hash=pattern_hash)
        if result.single():
            node_id = result.single()[0].id
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
        """Добавляет ребро между узлами"""
        tx.run(
            "MATCH (a:Pattern), (b:Pattern) WHERE id(a) = $node1 AND id(b) = $node2 "
            "MERGE (a)-[:NEXT]->(b)",
            node1=node1, node2=node2
        )
    
    def optimize_graph(self, threshold=2):
        """Удаляет редко используемые узлы (прунинг)"""
        with self.driver.session() as session:
            nodes_to_remove = [node for node, count in self.pattern_usage.items() if count < threshold]
            for node in nodes_to_remove:
                session.run("MATCH (n:Pattern) WHERE id(n) = $node_id DETACH DELETE n", node_id=node)
                hash_value = next(h for h, n in self.hash_to_node.items() if n == node)
                del self.hash_to_node[hash_value]
                del self.pattern_usage[node]
            logging.info(f"Removed {len(nodes_to_remove)} nodes with usage < {threshold}")
    
    def adapt_to_usage(self):
        """Перемещает часто используемые узлы ближе к корню"""
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

# 3. Предобработка данных
def preprocess_data(file_path, chunk_size=1024):
    """Читает файл и разбивает на чанки"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        processed_chunks = []
        for chunk in chunks:
            chunk_array = np.frombuffer(chunk, dtype=np.uint8) / 255.0
            if len(chunk_array) < chunk_size:
                chunk_array = np.pad(chunk_array, (0, chunk_size - len(chunk_array)), 'constant')
            processed_chunks.append(chunk_array)
        return processed_chunks
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return []

# 4. Обучение автоэнкодера
def train_autoencoder(model, data, epochs=10, batch_size=32, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            batch = batch.float().to(device)
            optimizer.zero_grad()
            encoded, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader):.4f}')
    
    return model

# 5. Основной процесс
if __name__ == "__main__":
    # Параметры
    input_dim = 1024  # Размер чанка
    hidden_dim = 512
    latent_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "your_password"  # Замените на ваш пароль Neo4j
    
    # Инициализация
    model = KeyPGAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
    storage = KeyPGStorage(neo4j_uri, neo4j_user, neo4j_password)
    
    # Пример данных
    data_files = ['sample1.txt', 'sample2.txt']  # Замените на ваши файлы
    data = []
    for file in data_files:
        if os.path.exists(file):
            chunks = preprocess_data(file, chunk_size=input_dim)
            data.extend(chunks)
        else:
            logging.warning(f"File {file} not found")
    
    if not data:
        logging.error("No valid data to process")
        exit()
    
    data = torch.tensor(data, dtype=torch.float32)
    
    # Обучение автоэнкодера
    model = train_autoencoder(model, data, epochs=10, batch_size=32)
    
    # Сохранение паттернов в Neo4j
    with storage.driver.session() as session:
        with torch.no_grad():
            for chunk in data:
                chunk = chunk.float().to(device)
                encoded, _ = model(chunk)
                node_id = session.write(storage.add_pattern, encoded.cpu().numpy(), chunk.cpu().numpy())
                storage.pattern_usage[node_id] += 1
    
    # Оптимизация и адаптация графа
    storage.optimize_graph(threshold=2)
    storage.adapt_to_usage()
    
    # Закрытие соединения
    storage.close()
    
    logging.info("Processing completed")
import torch
import os
import logging
from storage import KeyPGStorage
from nn import KeyPGAutoencoder, preprocess_data
from text_utils import get_fasttext_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_dim = 300
hidden_dim = 256
latent_dim = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры Neo4j (Эти строки должны быть в этом файле!)
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "your_password"

model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
model = KeyPGAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

storage = KeyPGStorage(neo4j_uri, neo4j_user, neo4j_password)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
ft_model = get_fasttext_model()
data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt') or f.endswith('.docx') or f.endswith('.pdf')]

with storage.driver.session() as session:
    with torch.no_grad():
        for file in data_files:
            chunks = preprocess_data(file, ft_model, chunk_size_words=64)
            if not chunks:
                continue
            for chunk in chunks:
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).to(device)
                encoded, _ = model(chunk_tensor)
                node_id = session.write(storage.add_pattern, encoded.cpu().numpy(), chunk_tensor.cpu().numpy())
                storage.pattern_usage[node_id] += 1

storage.optimize_graph(threshold=2)
storage.adapt_to_usage()
storage.close()

logging.info("Сохранение паттернов и оптимизация графа завершены.")
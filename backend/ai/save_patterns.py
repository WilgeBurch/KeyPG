import torch
import os
import logging
import hashlib
from storage import KeyPGStorage
from nn import KeyPGAutoencoder
from text_utils import get_fasttext_model, preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_dim = 300
hidden_dim = 256
latent_dim = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры Neo4j
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "your_password"  # ЗАМЕНИ на свой реальный пароль!

model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
model = KeyPGAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

storage = KeyPGStorage(neo4j_uri, neo4j_user, neo4j_password)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
ft_model = get_fasttext_model()
data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt') or f.endswith('.docx') or f.endswith('.pdf')]

BATCH_SIZE = 64  # можно менять для ускорения

def hash_pattern(vec):
    return hashlib.sha256(vec.tobytes()).hexdigest()

all_patterns = []
file_pattern_hashes = []

with torch.no_grad():
    for file_idx, file in enumerate(data_files):
        chunks = preprocess_data(file, ft_model, min_words=40, max_words=90)
        if not chunks:
            continue
        hashes_this_file = []
        for chunk_vec in chunks:
            chunk_tensor = torch.tensor(chunk_vec, dtype=torch.float32).to(device)
            encoded = model.encoder(chunk_tensor).cpu().numpy()
            pat_hash = hash_pattern(encoded)
            # Можно добавить оригинальный текст чанка, если нужно восстановление текстом
            all_patterns.append({
                "hash": pat_hash,
                "data": encoded.tolist(),
                "text": ""  # сюда можно добавить текст чанка, если нужно
            })
            hashes_this_file.append(pat_hash)
        file_pattern_hashes.append({
            "file_id": file_idx,
            "pattern_hashes": hashes_this_file
        })

# Batch insert patterns
# Чтобы не было дублей, фильтруем по hash (только уникальные паттерны)
unique_patterns = {p['hash']: p for p in all_patterns}
storage.batch_add_patterns(list(unique_patterns.values()))
storage.batch_add_restore_keys(file_pattern_hashes)

storage.close()
logging.info("Batch-запись паттернов и ключей восстановления завершена.")
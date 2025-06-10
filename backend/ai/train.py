import os
import torch
import logging
from nn import KeyPGAutoencoder, preprocess_data, train_autoencoder
from text_utils import get_fasttext_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_dim = 300   # FastText vector size
hidden_dim = 256
latent_dim = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ft_model = get_fasttext_model()

data_dir = os.path.join(os.path.dirname(__file__), 'data')
data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
data = []
for file in data_files:
    chunks = preprocess_data(file, ft_model, chunk_size_words=64)
    data.extend(chunks)
if not data:
    logging.error("Нет данных для обучения.")
    exit(1)
data = torch.tensor(data, dtype=torch.float32)

model = KeyPGAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
logging.info("Начинается обучение...")
model = train_autoencoder(model, data, epochs=10, batch_size=32, device=device)
torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'model.pt'))
logging.info("Модель обучена и сохранена в model.pt")
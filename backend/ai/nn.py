import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from text_utils import get_fasttext_model, clean_text, text_to_chunks, chunk_to_vector
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Автоэнкодер
class KeyPGAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 2. Предобработка для обучения (смысловые чанки, fasttext)
def preprocess_data(file_path, ft_model, chunk_size_words=64):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = clean_text(text)
        chunks = text_to_chunks(text, chunk_size_words)
        processed_chunks = [chunk_to_vector(chunk, ft_model) for chunk in chunks if chunk]
        return processed_chunks
    except Exception as e:
        logging.error(f"Ошибка при обработке файла {file_path}: {e}")
        return []

def train_autoencoder(model, data, epochs=10, batch_size=32, lr=0.001, device=None):
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
import os
import numpy as np
import torch
import torch.nn as nn

class KeyPGAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, vec):
        """
        Получить латентный вектор (код) из входного вектора.
        vec: np.ndarray или list длины input_dim
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(np.array(vec), dtype=torch.float32).unsqueeze(0)
            z = self.encoder(x)
            return z.squeeze(0).cpu().numpy()

def load_autoencoder_model(path="best_model.pt", input_dim=300, hidden_dim=128, latent_dim=64):
    """
    Загружает KeyPGAutoencoder из .pt-файла. Если файла нет — создаёт новую модель.
    """
    model = KeyPGAutoencoder(input_dim, hidden_dim, latent_dim)
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
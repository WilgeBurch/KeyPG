import sys
import os
import torch
import numpy as np

from nn import KeyPGAutoencoder
from text_utils import preprocess_data, get_fasttext_model

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# == Аргумент: путь к файлу для инференса ==
if len(sys.argv) < 2:
    print("Использование: python infer.py путь_к_файлу")
    exit(1)

input_path = sys.argv[1]
if not os.path.exists(input_path):
    print(f"Файл не найден: {input_path}")
    exit(1)

# == Параметры модели ==
input_dim = 300
hidden_dim = 256
latent_dim = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# == Загрузка FastText ==
ft_model = get_fasttext_model()

# == Предобработка файла ==
chunks = preprocess_data(input_path, ft_model, chunk_size_words=128)
if not chunks:
    print("Не удалось обработать файл или он пустой.")
    exit(1)

data = np.array(chunks)
data = torch.tensor(data, dtype=torch.float32).to(device)

# == Загрузка обученной модели ==
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
if not os.path.exists(model_path):
    print(f"Файл модели не найден: {model_path}")
    exit(1)

model = KeyPGAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# == Инференс ==
with torch.no_grad():
    latent_vectors = model.encoder(data)

print(f"Обработано {len(chunks)} чанков текста.")
print("Латентные вектора (первые 2):")
print(latent_vectors[:2].cpu().numpy())

with torch.no_grad():
    reconstructed = model.decoder(latent_vectors)
print("Пример реконструкции (первые 5 чисел):")
print("Вход:", data[0][:5].cpu().numpy())
print("Восстановлено:", reconstructed[0][:5].cpu().numpy())
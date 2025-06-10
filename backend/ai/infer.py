import torch
import sys
import os
from nn import KeyPGAutoencoder, preprocess_data
from text_utils import get_fasttext_model

input_dim = 300
hidden_dim = 256
latent_dim = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if len(sys.argv) < 2:
    print("Использование: python infer.py путь_к_файлу.txt")
    exit(1)

file_path = sys.argv[1]
if not os.path.exists(file_path):
    print(f"Файл {file_path} не найден")
    exit(1)

ft_model = get_fasttext_model()
chunks = preprocess_data(file_path, ft_model, chunk_size_words=64)
if not chunks:
    print("Нет данных для обработки")
    exit(1)
data = torch.tensor(chunks, dtype=torch.float32).to(device)
model = KeyPGAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'model.pt'), map_location=device))
model.eval()

with torch.no_grad():
    for i, chunk in enumerate(data):
        encoded, _ = model(chunk)
        print(f"Chunk {i+1}: Encoded vector (первые 10): {encoded.cpu().numpy()[:10]}")
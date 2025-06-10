import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from nn import KeyPGAutoencoder
from text_utils import preprocess_data, get_fasttext_model

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ==== Гиперпараметры ====
input_dim = 300
hidden_dim = 256
latent_dim = 64
chunk_size_words = 128    # Можно менять 64/128/256
batch_size = 32
learning_rate = 1e-3
num_epochs = 10           # Оптимально для большого датасета
checkpoint_period = 10    # Сохранять чекпоинт каждые N эпох
max_checkpoints = 5       # Хранить только N последних чекпоинтов
patience = 7              # Для Early Stopping
val_split = 0.2           # 80% train / 20% val

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Загрузка FastText ====
print("Загрузка FastText word vectors...")
ft_model = get_fasttext_model()

# ==== Предобработка данных ====
data_dir = 'data'
all_chunks = []
for fname in os.listdir(data_dir):
    if fname.lower().endswith(('.txt', '.docx', '.pdf')):
        print(f"Обработка файла: {fname}")
        chunks = preprocess_data(os.path.join(data_dir, fname), ft_model, chunk_size_words=chunk_size_words)
        all_chunks.extend(chunks)
if not all_chunks:
    print("Нет данных для обучения.")
    exit(1)

data = np.array(all_chunks)
data = torch.tensor(data, dtype=torch.float32)

# ==== Деление на train/val ====
val_size = int(len(data) * val_split)
train_size = len(data) - val_size
train_dataset, val_dataset = random_split(data, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ==== Модель и оптимизатор ====
model = KeyPGAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

print("Начинается обучение...")
print(model)
print(f"Всего обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

best_val_loss = float('inf')
epochs_no_improve = 0
checkpoint_paths = []

for epoch in range(1, num_epochs + 1):
    model.train()
    train_losses = []
    for xb in train_loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        x_hat = model(xb)
        loss = criterion(x_hat, xb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    train_loss = np.mean(train_losses)

    # == Validation ==
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb in val_loader:
            xb = xb.to(device)
            x_hat = model(xb)
            loss = criterion(x_hat, xb)
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)

    print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # == Сохраняем чекпоинт каждые N эпох ==
    if epoch % checkpoint_period == 0:
        cp_path = f'model_epoch_{epoch}.pt'
        torch.save(model.state_dict(), cp_path)
        checkpoint_paths.append(cp_path)
        if len(checkpoint_paths) > max_checkpoints:
            os.remove(checkpoint_paths.pop(0))

    # == Лучшая модель по val_loss ==
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        epochs_no_improve = 0
        print("Лучшая модель обновлена!")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping: val loss не улучшается {patience} эпох подряд.")
            break

# Сохраняем последнюю модель
torch.save(model.state_dict(), 'model.pt')
print("Обучение завершено. Модель сохранена в model.pt (и best_model.pt — лучший чекпоинт по val loss).")

# == Пример реконструкции ==
model.eval()
with torch.no_grad():
    xb = data[0].unsqueeze(0).to(device)
    x_hat = model(xb)
    print("Пример реконструкции (первые 5 чисел):")
    print("Вход:", xb[0][:5].cpu().numpy())
    print("Восстановлено:", x_hat[0][:5].cpu().numpy())
import os
import re
import fasttext
import fasttext.util
import numpy as np
import nltk

# Дополнительные импорты для поддержки docx и pdf
try:
    import docx
except ImportError:
    docx = None
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

nltk.download('punkt')

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cc.ru.300.bin")

def download_fasttext_model():
    import urllib.request
    import gzip
    import shutil
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz"
    gz_path = MODEL_PATH + ".gz"
    print("Скачиваем FastText русскую модель (1.2 ГБ)...")
    urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, 'rb') as f_in:
        with open(MODEL_PATH, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print("FastText модель скачана и распакована.")

def get_fasttext_model():
    if not os.path.exists(MODEL_PATH):
        download_fasttext_model()
    return fasttext.load_model(MODEL_PATH)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9ё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_chunks(text, chunk_size_words=64):
    """Разбивает текст на чанки по N слов"""
    words = nltk.word_tokenize(text, language='russian')
    chunks = [words[i:i+chunk_size_words] for i in range(0, len(words), chunk_size_words)]
    return chunks

def chunk_to_vector(chunk_words, ft_model):
    """Преобразует список слов чанка в средний fasttext-вектор"""
    if not chunk_words:
        return np.zeros(ft_model.get_dimension(), dtype=np.float32)
    vectors = [ft_model.get_word_vector(word) for word in chunk_words]
    return np.mean(vectors, axis=0)

def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.docx':
        if docx is None:
            raise ImportError("Модуль python-docx не установлен. Установите его с помощью 'pip install python-docx'")
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == '.pdf':
        if pdfplumber is None:
            raise ImportError("Модуль pdfplumber не установлен. Установите его с помощью 'pip install pdfplumber'")
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or '' for page in pdf.pages)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def preprocess_data(file_path, ft_model, chunk_size_words=64):
    import logging
    try:
        text = extract_text(file_path)
        text = clean_text(text)
        chunks = text_to_chunks(text, chunk_size_words)
        processed_chunks = [chunk_to_vector(chunk, ft_model) for chunk in chunks if chunk]
        return processed_chunks
    except Exception as e:
        logging.error(f"Ошибка при обработке файла {file_path}: {e}")
        return []
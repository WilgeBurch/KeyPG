# Краткая инструкция

## 1. Установка зависимостей
```sh
cd backend/ai
pip install -r requirements.txt
```

## 2. Скачивание FastText модели
Скачивается автоматически при первом запуске train.py/infer.py/save_patterns.py.

## 3. Подготовка данных
Положите русскоязычные текстовые файлы в папку data/.  
Поддерживаются .txt, .docx, .pdf

## 4. Обучение модели
```sh
python train.py
```

## 5. Инференс (проверка нового файла)
```sh
python infer.py data/sample1.txt
```

## 6. Сохранение паттернов в Neo4j
Перед запуском убедитесь, что Neo4j работает и параметры доступа указаны верно в save_patterns.py.
```sh
python save_patterns.py
```
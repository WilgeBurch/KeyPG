import os
import shutil
from datetime import datetime, timedelta
from typing import List
from uuid import uuid4

import aiofiles
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from apscheduler.schedulers.background import BackgroundScheduler
import fastapi.responses

# Neo4j
from neo4j import GraphDatabase, basic_auth

# ai modules
from ai.text_utils import extract_text_from_bytes, clean_text, dynamic_fragment, chunk_to_vector, get_fasttext_model
from ai.nn import load_autoencoder_model

# Конфигурация через переменные окружения
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/filesdb")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploaded_files")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_neo4j_password")

# SQLAlchemy
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Neo4j driver (singleton)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))

# AI models
ft_model = get_fasttext_model()
autoencoder = load_autoencoder_model()

# Модели
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    files = relationship("FileInfo", back_populates="owner")

class FileInfo(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    extension = Column(String)
    upload_date = Column(DateTime, default=datetime.utcnow)
    uuid = Column(String, unique=True, index=True)
    size = Column(Integer, default=0)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="files")

Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Не удалось проверить учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=username)
    if user is None:
        raise credentials_exception
    return user

# Регистрация пользователя
@app.post("/register")
def register(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    if get_user(db, form_data.username):
        raise HTTPException(status_code=400, detail="Пользователь уже существует")
    user = User(
        username=form_data.username,
        hashed_password=get_password_hash(form_data.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"msg": "Пользователь успешно зарегистрирован"}

# Авторизация пользователя
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Неправильное имя пользователя или пароль")
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Загрузка файла (без сохранения на диск, обработка и запись в Neo4j)
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ext = os.path.splitext(file.filename)[1]
    name = os.path.splitext(file.filename)[0]
    unique_id = str(uuid4())
    content = await file.read()
    # --- Новый пайплайн: сразу в AI и Neo4j ---
    # 1. Извлекаем текст из файла (работа с bytes)
    text = extract_text_from_bytes(content, file.filename)
    if not text:
        raise HTTPException(status_code=400, detail="Не удалось извлечь текст из файла")
    # 2. Предобработка и разбиение на чанки
    clean = clean_text(text)
    chunks = dynamic_fragment(clean)
    # 3. Векторизация и кодирование автоэнкодером
    vectors = [chunk_to_vector(chunk.split(), ft_model) for chunk in chunks]
    encoded = [autoencoder.encode(vec) for vec in vectors]
    # 4. Сохраняем паттерны в Neo4j
    with neo4j_driver.session() as session:
        for idx, code in enumerate(encoded):
            session.run(
                "MERGE (p:Pattern {hash: $hash}) "
                "SET p.text = $text, p.file_uuid = $file_uuid, p.chunk_idx = $idx",
                hash=str(hash(bytes(code))), text=chunks[idx], file_uuid=unique_id, idx=idx
            )
        session.run(
            "MERGE (d:Document {uuid: $file_uuid}) "
            "SET d.filename = $filename, d.owner_id = $owner_id, d.upload_date = $upload_date, "
            "d.pattern_hashes = $hashes",
            file_uuid=unique_id,
            filename=name,
            owner_id=current_user.id,
            upload_date=datetime.utcnow().isoformat(),
            hashes=[str(hash(bytes(code))) for code in encoded]
        )
    # 5. Сохраняем метаинформацию о файле в Postgres
    db_file = FileInfo(
        filename=name,
        extension=ext,
        uuid=unique_id,
        owner_id=current_user.id,
        size=len(content)
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return {"uuid": unique_id, "filename": file.filename, "size": len(content)}

# Получение списка файлов пользователя
@app.get("/files")
def list_files(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    files = db.query(FileInfo).filter(FileInfo.owner_id == current_user.id).all()
    return [
        {
            "uuid": file.uuid,
            "filename": file.filename,
            "extension": file.extension,
            "upload_date": file.upload_date,
            "size": file.size
        } for file in files
    ]

# Восстановление файла по ключу (выгрузка в папку загрузки)
@app.post("/restore/{file_uuid}")
async def restore_file(file_uuid: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Проверяем права пользователя на файл
    db_file = db.query(FileInfo).filter(FileInfo.uuid == file_uuid, FileInfo.owner_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="Файл не найден")
    # 1. Получаем порядок паттернов для документа
    with neo4j_driver.session() as session:
        doc = session.run(
            "MATCH (d:Document {uuid: $file_uuid}) RETURN d.pattern_hashes AS pattern_hashes",
            file_uuid=file_uuid
        ).single()
        if not doc or not doc["pattern_hashes"]:
            raise HTTPException(status_code=404, detail="Ключи паттернов для файла не найдены")
        hashes = doc["pattern_hashes"]
        # 2. Получаем текстовые чанки по паттернам
        result = session.run(
            "UNWIND $hashes AS h "
            "MATCH (p:Pattern {hash: h}) "
            "RETURN p.text AS text, p.chunk_idx AS idx "
            "ORDER BY apoc.coll.indexOf($hashes, h)",
            hashes=hashes
        )
        chunks = [r["text"] for r in result if r["text"]]
        if not chunks:
            raise HTTPException(status_code=404, detail="Не найдено ни одного фрагмента для восстановления")
    # 3. Собираем финальный текст
    restored_text = "\n".join(chunks)
    # 4. Сохраняем в папку загрузок пользователя
    downloads_dir = os.path.expanduser("~/Downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    out_path = os.path.join(downloads_dir, f"restored_{db_file.filename}{db_file.extension}")
    async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
        await f.write(restored_text)
    return {"path": out_path}

# Удаление файла пользователя по uuid
@app.delete("/delete/{file_uuid}")
async def delete_file(file_uuid: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_file = db.query(FileInfo).filter(FileInfo.uuid == file_uuid, FileInfo.owner_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="Файл не найден")
    # Удаляем из Neo4j
    with neo4j_driver.session() as session:
        session.run("MATCH (d:Document {uuid: $file_uuid}) DETACH DELETE d", file_uuid=file_uuid)
        session.run("MATCH (p:Pattern {file_uuid: $file_uuid}) DETACH DELETE p", file_uuid=file_uuid)
    db.delete(db_file)
    db.commit()
    return {"msg": "Файл успешно удалён"}

# Очистка БД от отсутствующих файлов (не требуется в новой архитектуре, но оставим для Postgres)
def clean_missing_files():
    db = SessionLocal()
    try:
        files = db.query(FileInfo).all()
        for file in files:
            # Проверяем только PostgreSQL, физические файлы не храним
            pass
        db.commit()
    finally:
        db.close()

# Планировщик для периодической проверки файлов
scheduler = BackgroundScheduler()
scheduler.add_job(clean_missing_files, "interval", minutes=5)
scheduler.start()

# Для корректного завершения планировщика
import atexit
atexit.register(lambda: scheduler.shutdown())
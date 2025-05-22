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

# Конфигурация через переменные окружения
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/filesdb")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploaded_files")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SQLAlchemy
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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
    size = Column(Integer, default=0)  # Новое поле для размера файла
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

# Загрузка файла
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ext = os.path.splitext(file.filename)[1]
    name = os.path.splitext(file.filename)[0]
    unique_id = str(uuid4())
    save_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}{ext}")

    async with aiofiles.open(save_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    size = os.path.getsize(save_path)  # Получаем размер файла на диске

    db_file = FileInfo(
        filename=name,
        extension=ext,
        uuid=unique_id,
        owner_id=current_user.id,
        size=size  # Сохраняем размер файла
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return {"uuid": unique_id, "filename": file.filename, "size": size}

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
            "size": file.size  # Возвращаем размер файла
        } for file in files
    ]

# Скачивание файла
@app.get("/download/{file_uuid}")
async def download_file(file_uuid: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_file = db.query(FileInfo).filter(FileInfo.uuid == file_uuid, FileInfo.owner_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="Файл не найден")
    file_path = os.path.join(UPLOAD_FOLDER, f"{db_file.uuid}{db_file.extension}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл отсутствует на сервере")
    return fastapi.responses.FileResponse(path=file_path, filename=f"{db_file.filename}{db_file.extension}", media_type="application/octet-stream")

# Удаление файла пользователя по uuid
@app.delete("/delete/{file_uuid}")
async def delete_file(file_uuid: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_file = db.query(FileInfo).filter(FileInfo.uuid == file_uuid, FileInfo.owner_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="Файл не найден")
    file_path = os.path.join(UPLOAD_FOLDER, f"{db_file.uuid}{db_file.extension}")
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Не удалось удалить файл с диска")
    db.delete(db_file)
    db.commit()
    return {"msg": "Файл успешно удалён"}

# Очистка БД от отсутствующих файлов
def clean_missing_files():
    db = SessionLocal()
    try:
        files = db.query(FileInfo).all()
        for file in files:
            path = os.path.join(UPLOAD_FOLDER, f"{file.uuid}{file.extension}")
            if not os.path.exists(path):
                db.delete(file)
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


import os
from datetime import datetime, timedelta
from typing import Annotated, Literal, Optional, Union

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

from source_code.api.db.mongo import get_db

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = os.environ["ALGORITHM"]
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"])


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    tel: str | None = None
    full_name: str | None = None
    role: Optional[Literal["admin", "expert", "user"]] = "user"
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


class Users(BaseModel):
    users: Union[list[User], list] = []


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)  # 必填，长度限制
    password: str = Field(..., min_length=6)  # 密码需要加密存储
    email: Optional[EmailStr] = None  # 使用Pydantic的邮箱格式验证
    tel: Optional[str] = Field(None, pattern=r"^1[3-9]\d{9}$")  # 简单的手机号正则
    full_name: Optional[str] = None
    role: Optional[Literal["admin", "expert", "user"]] = "user"


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


async def get_user(client, username: str):
    user_dict = await client.find_one("users", {"username": username})
    if not user_dict is None:
        return UserInDB(**user_dict)
    else:
        return None


async def authenticate_user(client, username: str, password: str):
    user = await get_user(client, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    db = await get_db()
    user = await get_user(client=db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
        current_user: Annotated[User, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

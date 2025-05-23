import logging
import os
import uuid
from datetime import timedelta
from typing import Annotated, Optional, Literal

from fastapi import Depends, FastAPI, HTTPException, status, Request, UploadFile, Body, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm

from source_code.api.config import static_dir, pdf_dir
from source_code.api.core.auth import Token, authenticate_user, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, \
    User, get_current_active_user, UserRegister, get_password_hash, Users
from source_code.api.db.mongo import AsyncMongoDB, get_db
from source_code.api.utils import transform_pdf_to_txt, save_embed_text, add_embed_text, convert_pdf_to_images_fitz

logger = logging.getLogger(__name__)

app = FastAPI()


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "msg": exc.args[0], "data": None},
        )


@app.get("/hello")
def read_root():
    return {"Hello": "World"}


@app.post("/image")
async def post_image(
        image: UploadFile
):
    contents = await image.read()
    new_file_name = uuid.uuid4().hex + ".jpg"
    image_save_path = os.path.join(static_dir, new_file_name)
    with open(image_save_path, "wb") as f:
        f.write(contents)
    return {"code": 0, "msg": "success", "data": {"filepath": image_save_path}}


@app.get("/image")
async def get_image(
        file_path: str
):
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )
    return FileResponse(file_path)


@app.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(
        user: UserRegister,
        db: AsyncMongoDB = Depends(get_db)
):
    # 检查用户名是否已存在
    count = await db.count_documents("users", {"username": user.username})
    if count > 0:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )

    user_dict = user.dict()
    user_dict["hashed_password"] = get_password_hash(user_dict["password"])
    user_dict.pop("password")
    user_dict["disabled"] = False

    # 存储到"数据库"
    await db.insert_one("users", user_dict)

    # 返回时移除密码字段
    return user_dict


@app.post("/login")
async def login_for_access_token(
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
        db: AsyncMongoDB = Depends(get_db)
) -> Token:
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@app.get("/users/me", response_model=User)
async def read_users_me(
        current_user: Annotated[User, Depends(get_current_active_user)],
):
    return current_user


@app.get("/users", response_model=Users)
async def read_users(
        current_user: Annotated[User, Depends(get_current_active_user)],
        db: AsyncMongoDB = Depends(get_db),
        role: Optional[Literal["admin", "expert", "user"]] = None,
):
    # if current_user.role != "admin":
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="您没有权限访问此资源",
    #     )
    query = {}
    if role is not None:
        query.update({"role": role})
    user_list = await db.find("users", query)
    response_user_list = [User(**user) for user in user_list]
    return {"users": response_user_list}


@app.delete("/users/{username}")
async def delete_user(
        username: str,
        current_user: Annotated[User, Depends(get_current_active_user)],
        db: AsyncMongoDB = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="您没有权限访问此资源",
        )
    await db.delete_one("users", {"username": username})
    return {"code": 0, "msg": "delete success", "data": {"username": username}}


@app.post("/rag_expert")
async def post_rag_expert(
        expert_name: str = Body(..., description="专家名称"),
        expert_description: str = Body(..., description="专家描述"),
        db: AsyncMongoDB = Depends(get_db)
):
    await db.insert_one(
        "rag_expert",
        {
            "name": expert_name,
            "description": expert_description,
            "uuid": uuid.uuid4().hex,
            "rag_data": []
        }
    )
    init_doc_list = [
        {
            "text": expert_description,
            "page_idx": None,
            "source": None
        }
    ]
    save_embed_text(init_doc_list, expert_name)
    return {"code": 0, "msg": "success", "data": {"expert_name": expert_name}}


@app.get("/rag_expert")
async def get_rag_expert(
        db: AsyncMongoDB = Depends(get_db)
):
    expert_list = await db.find("rag_expert", query={}, projection={"_id": 0})
    return {"code": 0, "msg": "success", "data": expert_list}


@app.delete("/rag_expert/{expert_name}")
async def delete_rag_expert(
        expert_name: str,
        db: AsyncMongoDB = Depends(get_db)
):
    await db.delete_one("rag_expert", {"name": expert_name})
    return {"code": 0, "msg": "success", "data": {"expert_name": expert_name}}


@app.post("/rag_data")
async def post_rag_data(
        file: UploadFile,
        expert_name: str = Form(..., description="专家名称"),
        db: AsyncMongoDB = Depends(get_db)
):
    os.makedirs(pdf_dir, exist_ok=True)
    if file.filename.endswith(".pdf"):
        # update embedding
        contents = await file.read()
        file_save_path = os.path.join(pdf_dir, file.filename)
        with open(file_save_path, "wb") as f:
            f.write(contents)
        doc_list = transform_pdf_to_txt(file_save_path)
        add_embed_text(doc_list, expert_name)
        # update db
        await db.update_one(
            "rag_expert",
            {"name": expert_name},
            {"$push": {"rag_data": file_save_path}}
        )
        # update images
        save_pdf_images_dir = os.path.join(pdf_dir, os.path.splitext(file.filename)[0])
        os.makedirs(save_pdf_images_dir, exist_ok=True)
        convert_pdf_to_images_fitz(file_save_path, save_pdf_images_dir)
    else:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
    return {"code": 0, "msg": "success", "data": {"filepath": file_save_path}}

@app.get("/rag_data")
async def get_rag_data(
        file_name: str,
):
    file_save_path = os.path.join(pdf_dir, file_name)
    if not os.path.exists(file_save_path):
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )
    return FileResponse(file_save_path)

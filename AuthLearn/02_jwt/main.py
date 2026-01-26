"""
Stateless JWT authentication demo with FastAPI + SQLite.
Run: uvicorn main:app --reload --port 8002
Default user: alice / 123456
"""

from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jose import JWTError, jwt
from passlib.context import CryptContext

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "auth.db"
SECRET_KEY = "jwt-secret"
ALGORITHM = "HS256"
ACCESS_TTL_MIN = 15
REFRESH_TTL_DAYS = 7

app = FastAPI(title="JWT Demo")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as db:
        db.execute(
            """
            create table if not exists users(
                id integer primary key autoincrement,
                username text unique,
                password_hash text
            )
            """
        )
        row = db.execute("select 1 from users where username=?", ("alice",)).fetchone()
        if not row:
            db.execute(
                "insert into users(username, password_hash) values (?, ?)",
                ("alice", pwd_ctx.hash("123456")),
            )
            db.commit()


@app.on_event("startup")
def startup():
    init_db()


def create_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + expires_delta
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def get_current_user(
    request: Request, authorization: str | None = Header(default=None)
):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    elif "access_token" in request.cookies:
        token = request.cookies.get("access_token")

    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    user_id = int(payload.get("sub"))
    with get_db() as db:
        row = db.execute(
            "select id, username from users where id=?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def login_page():
    return HTMLResponse(
        """
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>JWT Demo</title></head>
<body>
  <h3>JWT 登录示例</h3>
  <form id="login-form">
    <label>用户名 <input name="username" value="alice"></label><br>
    <label>密码 <input type="password" name="password" value="123456"></label><br>
    <button type="submit">获取 Token</button>
  </form>
  <button id="me-btn">调用 /me</button>
  <pre id="out"></pre>
<script>
const form = document.querySelector('#login-form');
form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fd = new FormData(form);
  const res = await fetch('/api/login', {method:'POST', body: fd});
  const data = await res.json();
  window.access = data.access;
  document.querySelector('#out').textContent = JSON.stringify(data, null, 2);
});
document.querySelector('#me-btn').onclick = async ()=>{
  const res = await fetch('/me', {headers: {'Authorization': 'Bearer '+(window.access||'')}});
  document.querySelector('#out').textContent = await res.text();
};
</script>
</body>
</html>
"""
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return login_page()


@app.post("/api/login")
def login(username: str = Form(...), password: str = Form(...)):
    with get_db() as db:
        row = db.execute(
            "select id, username, password_hash from users where username=?",
            (username,),
        ).fetchone()
    if not row or not pwd_ctx.verify(password, row["password_hash"]):
        raise HTTPException(status_code=400, detail="用户名或密码错误")

    access = create_token(
        {"sub": str(row["id"]), "name": row["username"]},
        expires_delta=timedelta(minutes=ACCESS_TTL_MIN),
    )
    refresh = create_token(
        {"sub": str(row["id"]), "type": "refresh"},
        expires_delta=timedelta(days=REFRESH_TTL_DAYS),
    )
    resp = JSONResponse({"access": access, "refresh": refresh})
    resp.set_cookie(
        "access_token",
        access,
        httponly=True,
        samesite="lax",
        max_age=int(timedelta(minutes=ACCESS_TTL_MIN).total_seconds()),
    )
    return resp


@app.post("/api/refresh")
def refresh(refresh_token: str = Form(...)):
    payload = decode_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="刷新 token 无效")
    new_access = create_token(
        {"sub": payload["sub"]},
        expires_delta=timedelta(minutes=ACCESS_TTL_MIN),
    )
    return {"access": new_access}


@app.get("/me")
def me(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="未认证")
    return user

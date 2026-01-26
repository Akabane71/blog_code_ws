"""
Simple session-cookie authentication demo with FastAPI + SQLite.
Run: uvicorn main:app --reload --port 8001
Default user: alice / 123456
"""

from datetime import datetime, timedelta
import secrets
import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from itsdangerous import URLSafeSerializer, BadSignature
from passlib.context import CryptContext

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "auth.db"
SESSION_TTL = timedelta(hours=1)

app = FastAPI(title="Session Cookie Demo")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
signer = URLSafeSerializer("session-secret")
sessions: dict[str, tuple[int, datetime]] = {}  # sid -> (user_id, expires_at)


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
        # seed demo user
        row = db.execute("select 1 from users where username=?", ("alice",)).fetchone()
        if not row:
            db.execute(
                "insert into users(username, password_hash) values (?, ?)",
                ("alice", pwd_ctx.hash("123456"[:72])),
            )
            db.commit()


@app.on_event("startup")
def startup():
    init_db()


def get_current_user(request: Request):
    raw_cookie = request.cookies.get("sid")
    if not raw_cookie:
        return None
    try:
        sid = signer.loads(raw_cookie)
    except BadSignature:
        return None
    stored = sessions.get(sid)
    if not stored:
        return None
    user_id, expires_at = stored
    if datetime.utcnow() > expires_at:
        sessions.pop(sid, None)
        return None
    with get_db() as db:
        user = db.execute(
            "select id, username from users where id=?", (user_id,)
        ).fetchone()
    return dict(user) if user else None


def render_page(body: str):
    return HTMLResponse(
        f"""
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>Session Demo</title></head>
<body>
{body}
<p style="margin-top:1rem">
  <a href="/">首页</a> | <a href="/protected">受保护接口(JSON)</a>
</p>
</body>
</html>
"""
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request, user=Depends(get_current_user)):
    if user:
        return render_page(
            f"<h3>Hi, {user['username']} 已登录</h3>"
            '<form action="/logout" method="post"><button>退出登录</button></form>'
        )
    return render_page(
        """
<h3>请登录</h3>
<form action="/login" method="post">
  <label>用户名 <input name="username" value="alice"></label><br>
  <label>密码 <input type="password" name="password" value="123456"></label><br>
  <button type="submit">登录</button>
</form>
"""
    )


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    with get_db() as db:
        row = db.execute(
            "select id, password_hash from users where username=?", (username,)
        ).fetchone()
    if not row or not pwd_ctx.verify(password[:72], row["password_hash"]):
        raise HTTPException(status_code=400, detail="用户名或密码错误")

    sid = secrets.token_urlsafe(32)
    sessions[sid] = (row["id"], datetime.utcnow() + SESSION_TTL)
    signed_cookie = signer.dumps(sid)

    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie(
        "sid",
        signed_cookie,
        httponly=True,
        samesite="lax",
        max_age=int(SESSION_TTL.total_seconds()),
    )
    return resp


@app.post("/logout")
def logout(request: Request):
    raw_cookie = request.cookies.get("sid")
    if raw_cookie:
        try:
            sid = signer.loads(raw_cookie)
            sessions.pop(sid, None)
        except BadSignature:
            pass
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie("sid")
    return resp


@app.get("/protected")
def protected(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    return JSONResponse({"message": "You are in", "user": user})

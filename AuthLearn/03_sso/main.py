"""
SSO / OAuth2 Authorization Code demo with FastAPI + Authlib.
Uses GitHub as IdP by default. Requires env vars:
  GITHUB_CLIENT_ID
  GITHUB_CLIENT_SECRET
Run: uvicorn main:app --reload --port 8003
"""

import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(title="SSO Demo")
oauth = OAuth()

# Needed for Authlib to store state in session between redirect and callback
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SSO_SESSION_SECRET", "dev-sso-session-secret"),
    same_site="lax",
)


def require_env():
    if not (os.getenv("GITHUB_CLIENT_ID") and os.getenv("GITHUB_CLIENT_SECRET")):
        raise HTTPException(
            status_code=500,
            detail="缺少 GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET 环境变量，无法演示。",
        )


oauth.register(
    name="github",
    client_id=os.getenv("GITHUB_CLIENT_ID") or "missing",
    client_secret=os.getenv("GITHUB_CLIENT_SECRET") or "missing",
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "read:user user:email"},
)


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>SSO Demo</title></head>
<body>
  <h3>GitHub OAuth2 授权码登录</h3>
  <a href="/login">跳转 GitHub 登录</a>
</body>
</html>
"""

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url='/')


@app.get("/login")
async def login(request: Request):
    require_env()
    redirect_uri = request.url_for("auth_callback")
    return await oauth.github.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.github.authorize_access_token(request)
    except OAuthError as e:
        raise HTTPException(status_code=400, detail=f"OAuth 错误: {e.error}")
    user = await oauth.github.get("user", token=token)
    profile = user.json()
    # 将用户信息与 access_token 简单存入 session，用于受保护页面
    request.session["user"] = {
        "login": profile.get("login"),
        "id": profile.get("id"),
        "avatar_url": profile.get("avatar_url"),
        "access_token": token["access_token"],
    }
    return RedirectResponse(url="/protected")


def current_user(request: Request):
    return request.session.get("user")


@app.get("/protected", response_class=HTMLResponse)
async def protected_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse(url="/login")
    masked = (user.get("access_token") or "")[:6] + "..."
    return f"""
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>Protected</title></head>
<body>
  <h3>受保护页面（需 SSO 登录）</h3>
  <p>GitHub 用户名：{user.get('login')} (id: {user.get('id')})</p>
  <p>Access Token（截断）：{masked}</p>
  <p><img src="{user.get('avatar_url')}" width="100" /></p>
  <p><a href="/">首页</a> | <a href="/logout">退出</a></p>
</body>
</html>
"""

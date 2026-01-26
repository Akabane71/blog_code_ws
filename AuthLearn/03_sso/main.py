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
    print("登录用户资料：", profile)
    html = f"""
    <h3>登录成功</h3>
    <p>GitHub 用户名：{profile.get('login')}</p>
    <p>ID：{profile.get('id')}</p>
    <p>Token（访问 GitHub API）：{token['access_token']}...</p>
    <img src="{profile.get('avatar_url')}" width="100" />
    """
    return HTMLResponse(html)

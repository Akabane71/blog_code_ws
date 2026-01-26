import uvicorn
from dotenv import load_dotenv

load_dotenv()
def jwt_server():
    uvicorn.run("02_jwt.main:app", host="localhost", port=8000)    
    
def session_cookie_server():
    uvicorn.run("01_seasion_cookie.main:app", host="localhost", port=8000)
    
def sso_server():
    uvicorn.run("03_sso.main:app", host="localhost", port=8000)
    

# jwt_server()
# session_cookie_server()
sso_server()
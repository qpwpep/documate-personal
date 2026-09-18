from fastapi import FastAPI
from pydantic import BaseModel, Field
app = FastAPI()
class PublicAccount(BaseModel):
    name: str
    quota: int = Field(ge=0, le=500)
@app.get("/account", response_model=PublicAccount)
def account():
    return {"name": "Mina", "quota": 75, "password_hash": "synthetic-not-a-secret"}

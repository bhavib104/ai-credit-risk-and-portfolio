from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.db import init_db
from backend.routers import predict, model, portfolio, drift, health

app = FastAPI(title="Credit Risk Decisioning API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    init_db()

app.include_router(health.router)
app.include_router(predict.router, prefix="/api")
app.include_router(model.router, prefix="/api")
app.include_router(portfolio.router, prefix="/api")
app.include_router(drift.router, prefix="/api")

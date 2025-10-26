from fastapi import FastAPI
from api.diffusion import router as diffusion_router
from api.ebm import router as ebm_router

app = FastAPI(title='Assignment 4 API ✅')

@app.get('/')
def home():
    return {'message': 'Assignment 4 API is running ✅'}

app.include_router(diffusion_router)
app.include_router(ebm_router)

# Assignment 4 — Advanced Image Generation (Diffusion + EBM)

This project is **separate** from Assignment 3 (GAN). It serves two endpoints via FastAPI and can be run locally or with Docker.

## Run locally
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn torch torchvision pillow
uvicorn app.main:app --host 0.0.0.0 --port 8003
# Test endpoints
curl -o diff.png http://localhost:8003/sample/diffusion
curl -o ebm.png  http://localhost:8003/sample/ebm


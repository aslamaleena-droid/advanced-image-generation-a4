# Assignment 4 — Diffusion + EBM API

## Setup
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn torch torchvision pillow

## Train (currently placeholders)
python train_diffusion_cifar.py
python train_ebm_cifar.py

## Serve
uvicorn app.main:app --host 0.0.0.0 --port 8003

## Sample
curl -o diff.png http://localhost:8003/sample/diffusion
curl -o ebm.png  http://localhost:8003/sample/ebm

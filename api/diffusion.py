from fastapi import APIRouter, Response

router = APIRouter()

@router.get('/sample/diffusion')
def sample_diffusion():
    # This will use the real diffusion model later!
    content = b'Dummy diffusion image data'
    return Response(content=content, media_type='application/octet-stream')

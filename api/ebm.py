from fastapi import APIRouter, Response

router = APIRouter()

@router.get('/sample/ebm')
def sample_ebm():
    # This will use the real EBM model later!
    content = b'Dummy ebm image data'
    return Response(content=content, media_type='application/octet-stream')

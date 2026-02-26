from fastapi.responses import StreamingResponse

def stream_response(generator_func):
    """Wrapper لإرسال الرد chunk by chunk"""
    return StreamingResponse(generator_func(), media_type="text/plain")

# from pathlib import Path
# import sys

# from fastapi import FastAPI

# ROOT = Path(__file__).resolve().parent.parent
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from app.api.multimodal_analysis import router as multimodal_router
# from app.api.qna import router as qna_router
# from app.api.report import router as report_router
# from app.api.request_outbound import router as request_outbound_router

# app = FastAPI(title="AI Response Server")
# app.include_router(multimodal_router)
# app.include_router(report_router)
# app.include_router(request_outbound_router)
# app.include_router(qna_router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8888, reload=True)

# # 실행 방법 python -m app.main app 폴더 경로에서


from pathlib import Path
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.multimodal_analysis import router as multimodal_router
from app.api.qna import router as qna_router
from app.api.report import router as report_router
from app.api.request_outbound import router as request_outbound_router

app = FastAPI(title="AI Response Server")

# ✅ 422(Validation Error) 상세를 콘솔에 찍기
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    print("\n=== 422 VALIDATION ERROR ===")
    print("URL:", request.url)
    print("BODY:", body.decode("utf-8", "ignore"))
    print("ERRORS:", exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

app.include_router(multimodal_router)
app.include_router(report_router)
app.include_router(request_outbound_router)
app.include_router(qna_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8888, reload=True)

# 실행 방법 python -m app.main  (app 폴더 경로에서)

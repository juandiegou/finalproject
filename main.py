"""_summary_
    Fast Api main File

    To start APP execute: uvicorn main:app --reload
    and open naviator in URL: 
"""
from fastapi import FastAPI, HTTPException, Request, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
#from fastapi.exceptions import RequestValidationError
from database import db
from router import endpoints
#from exceptions.UnAuthorizedException import UnAuthorizedException

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(endpoints.router)

@app.get("/")
async def root():
    """_summary_
        Main path for the app.
    """
    return {"message": f"Hello World. Welcome to FastAPI! {db['oauth_tokens']}"}


# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, exc):
#     return {"detail": exc.errors(), "body": exc.body}


# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"detail": exc.detail},
#     )

# @app.exception_handler(UnAuthorizedException)
# async def unauthorized_exception_handler(request, exc):
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"detail": exc.detail},
#     )


# @app.middleware("http")
# async def custom_exception_middleware(request: Request, call_next):
#     try:
#         return await call_next(request)
#     except Exception as exc:
#         return JSONResponse(
#             status_code=500,
#             content={"detail": "Internal Server Error"},
#         )

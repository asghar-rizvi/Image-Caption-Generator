from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os
from pathlib import Path
from prediction import preprocess_img, load_models
import uuid

app = FastAPI()

@app.on_event("startup")
async def load_models_on_startup():
    load_models()
    print("Models loaded successfully")

# templates and static file setup
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, 'templates')))
# app.mount("/static", StaticFiles(directory=str(Path(BASE_DIR, 'static'))), name="static")
#
# Create upload directory if it doesn't exist
UPLOAD_DIR = Path(BASE_DIR, "uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split(".")[-1]
        file_name = f"{uuid.uuid4()}.{file_ext}"
        file_path = Path(UPLOAD_DIR, file_name)
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process image and get caption
        caption = preprocess_img(file_path)
        
        # Delete the temporary file
        file_path.unlink()
        
        return JSONResponse(content={
            "success": True,
            "caption": caption,
            "filename": file.filename
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
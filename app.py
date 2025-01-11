from typing import Annotated
from fastapi import FastAPI, File, UploadFile, Request
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from src.utils import load_model, predict
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="template")
app.mount("/static", StaticFiles(directory="static"), name="static")


model, preprocess = load_model(
    "models/pretrained_vit16_32_batch-size_0.001_learning-rate_4_epochs.pt"
)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-image/")
async def predict_image(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")]
):
    image_data = await file.read()
    predictions = predict(model, image_data, preprocess)
    return JSONResponse(
        content={
            "label": predictions["label"].capitalize(),
            "probability": predictions["probability"],
        }
    )

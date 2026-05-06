from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib

app = FastAPI(title="Nepali Fake News Detector")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

MODEL = None
VECTORIZER = None

UI_DIR = Path(__file__).resolve().parent / "ui"

# Update this mapping if your model uses different labels.
LABEL_MAP = {
	0: "fake",
	1: "real",
}


class PredictRequest(BaseModel):
	text: str = Field(..., min_length=1)


def load_artifacts():
	base_dir = Path(__file__).resolve().parent
	model_path = base_dir / "model.pkl"
	vectorizer_path = base_dir / "vectorize.pkl"

	if not model_path.exists():
		raise FileNotFoundError(f"Missing model file: {model_path}")
	if not vectorizer_path.exists():
		raise FileNotFoundError(f"Missing vectorizer file: {vectorizer_path}")

	model = joblib.load(model_path)
	vectorizer = joblib.load(vectorizer_path)
	return model, vectorizer


def to_python_value(value):
	if hasattr(value, "item"):
		return value.item()
	return value


@app.on_event("startup")
def startup_event():
	global MODEL, VECTORIZER
	try:
		MODEL, VECTORIZER = load_artifacts()
	except Exception as exc:
		print("Failed to load model artifacts:", exc)


@app.get("/health")
def health_check():
	return {
		"status": "ok",
		"model_loaded": MODEL is not None,
		"vectorizer_loaded": VECTORIZER is not None,
	}


@app.post("/predict")
def predict(payload: PredictRequest):
	if MODEL is None or VECTORIZER is None:
		raise HTTPException(status_code=500, detail="Model artifacts not loaded")

	text = payload.text.strip()
	if not text:
		raise HTTPException(status_code=400, detail="Text is required")

	features = VECTORIZER.transform([text])
	raw_pred = to_python_value(MODEL.predict(features)[0])

	label = LABEL_MAP.get(raw_pred, raw_pred)
	label = to_python_value(label)
	probability = None

	if hasattr(MODEL, "predict_proba"):
		proba = MODEL.predict_proba(features)
		if hasattr(MODEL, "classes_"):
			classes = list(MODEL.classes_)
		else:
			classes = []

		if classes:
			try:
				class_index = classes.index(raw_pred)
				probability = float(proba[0][class_index])
			except ValueError:
				probability = float(max(proba[0]))
		else:
			probability = float(max(proba[0]))

	return {
		"label": label,
		"probability": probability,
		"raw_prediction": raw_pred,
	}


if UI_DIR.exists():
	app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")

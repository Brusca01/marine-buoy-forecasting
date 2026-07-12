import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from fastapi import FastAPI, HTTPException
from schemas import (
    PredictResponse, TargetPrediction, HistoryPoint,
    StaticClusterResponse, DynamicClusterResponse,
)
import predictor

app = FastAPI(title="Marine Buoy API")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictResponse)
@app.get("/api/predict", response_model=PredictResponse)
def predict(station: str = "42002", n_history: int = 168):
    try:
        res = predictor.predict(station, n_history)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    predictions = {}
    for target, v in res.items():
        predictions[target] = TargetPrediction(
            target=v["target"], unit=v["unit"],
            model_name=v["model_name"],
            last_timestamp=v["last_timestamp"],
            predicted_timestamp=v["predicted_timestamp"],
            predicted_value=v["predicted_value"],
            persistence_value=v["persistence_value"],
            history=[HistoryPoint(**p) for p in v["history"]],
        )
    return PredictResponse(station=station, predictions=predictions)


@app.get("/api/comparison")
def comparison():
    try:
        return predictor.get_metrics()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/api/clusters/static", response_model=StaticClusterResponse)
def clusters_static():
    try:
        return predictor.get_static_clusters()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/api/clusters/dynamic", response_model=DynamicClusterResponse)
def clusters_dynamic():
    try:
        return predictor.get_dynamic_clusters()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

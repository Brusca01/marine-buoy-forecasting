import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BACKEND = "http://localhost:8080"
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parents[1] / "static"

app = FastAPI(title="Marine Buoy Frontend")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _get(path, params=None):
    try:
        r = httpx.get(f"{BACKEND}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def _get_stations():
    data, _ = _get("/api/stations")
    return data or []


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    stations = _get_stations()
    return templates.TemplateResponse(request, "index.html", {"stations": stations})


@app.get("/prediction", response_class=HTMLResponse)
def prediction(request: Request, station: str = "42002", n_history: int = 168):
    data, err = _get("/api/predict", {"station": station, "n_history": n_history})
    stations = _get_stations()
    return templates.TemplateResponse(request, "prediction.html",
        {"data": data, "error": err, "station": station,
         "n_history": n_history, "stations": stations})


@app.get("/comparison", response_class=HTMLResponse)
def comparison(request: Request):
    data, err = _get("/api/comparison")
    return templates.TemplateResponse(request, "comparison.html",
        {"data": data, "error": err})


@app.get("/clustering", response_class=HTMLResponse)
def clustering(request: Request):
    static, serr = _get("/api/clusters/static")
    dynamic, derr = _get("/api/clusters/dynamic")
    return templates.TemplateResponse(request, "clustering.html",
        {"static": static, "dynamic": dynamic, "serr": serr, "derr": derr})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

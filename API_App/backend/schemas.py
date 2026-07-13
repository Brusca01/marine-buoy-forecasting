from pydantic import BaseModel


class HistoryPoint(BaseModel):
    timestamp: str
    value: float


class TargetPrediction(BaseModel):
    target: str
    unit: str
    model_name: str
    last_timestamp: str
    predicted_timestamp: str
    predicted_value: float
    persistence_value: float
    history: list[HistoryPoint]


class PredictResponse(BaseModel):
    station: str
    predictions: dict[str, TargetPrediction]


class StationInfo(BaseModel):
    id: str
    label: str


class StaticPoint(BaseModel):
    station_id: str
    cluster: int
    pca_x: float
    pca_y: float


class StaticClusterResponse(BaseModel):
    k: int
    silhouette: float
    silhouette_by_k: dict[str, float]
    inertia_by_k: dict[str, float]
    n_stations: int
    clusters: dict[str, list[str]]
    points: list[StaticPoint]


class DynamicPoint(BaseModel):
    station_id: str
    year: int
    cluster: int
    pca_x: float
    pca_y: float


class StationSequence(BaseModel):
    years: list[int]
    clusters: list[int]
    changes: int


class DynamicClusterResponse(BaseModel):
    k: int
    silhouette: float
    n_points: int
    sequences: dict[str, StationSequence]
    points: list[DynamicPoint]

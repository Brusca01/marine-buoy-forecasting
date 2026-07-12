from pydantic import BaseModel


class PredictRequest(BaseModel):
    station: str = "42002"
    n_history: int = 168


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


class MetricSet(BaseModel):
    MAE: float
    RMSE: float
    R2: float


class ModelMetrics(BaseModel):
    val: MetricSet
    test: MetricSet


class StaticPoint(BaseModel):
    year_month: str
    cluster: int
    pca_x: float
    pca_y: float


class StaticClusterResponse(BaseModel):
    k: int
    silhouette: float
    silhouette_by_k: dict[str, float]
    inertia_by_k: dict[str, float]
    n_points: int
    clusters: dict[str, list[str]]
    points: list[StaticPoint]


class DynamicPoint(BaseModel):
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

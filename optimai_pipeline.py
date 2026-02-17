# Runtime patch: avoid threadpoolctl/OpenBLAS introspection crash on macOS.

import contextlib



class _DummyTPL(contextlib.AbstractContextManager):

    def __init__(self, *args, **kwargs):

        pass

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc, tb):

        return False



def _no_threadpool_limits(*args, **kwargs):

    return _DummyTPL()



try:

    import sklearn.utils.fixes as _fixes

    _fixes.threadpool_limits = _no_threadpool_limits

except Exception:

    pass



try:

    import sklearn.cluster._kmeans as _sk_kmeans

    _sk_kmeans.threadpool_limits = _no_threadpool_limits

except Exception:

    pass



try:

    import threadpoolctl

    threadpoolctl.threadpool_limits = _no_threadpool_limits

except Exception:

    pass



try:

    from threadpoolctl import _OpenBLASModule

    def _safe_get_version(self):

        cfg_fn = getattr(self._dynlib, "openblas_get_config", lambda: None)

        cfg = cfg_fn()

        if not cfg:

            return "unknown"

        try:

            cfg_parts = cfg.split()

            if cfg_parts and cfg_parts[0] == b"OpenBLAS":

                return cfg_parts[1].decode("utf-8")

        except Exception:

            return "unknown"

        return "unknown"

    _OpenBLASModule.get_version = _safe_get_version

except Exception:

    pass

"""

OptimAI – Enterprise Sales Force & Omnichannel Automation Pipeline

-----------------------------------------------------------------

Author: dnAI Javier Castro (July 2025)



Highlights:

- Barcelona-like synthetic geography with configurable polygon/land masks.

- Gradient boosting churn/potential models plus Isolation Forest anomaly guardrails.

- Quantile segmentation + omnichannel NBA scoring (field + CC + ecommerce).

- Channel allocator balancing preferences, digital engagement, and SLA capacity.

- Preparation script builder for each visit/call aligned to NBA rationale.

- Field routing with greedy NBA-aware TSP + optional Folium visualisations.

- Contact-centre scheduler with SLA/backlog summary and monitoring telemetry.

- Enterprise-grade quality gates, logging, and monitoring hooks.

"""



import json

import logging

import os

import textwrap

from dataclasses import dataclass, field, replace

from datetime import datetime

from pathlib import Path

from typing import Callable, Dict, List, Optional, Tuple, Any

from urllib import error as urlerror

from urllib import request as urlrequest



import numpy as np

import pandas as pd



from sklearn.cluster import KMeans

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest

from sklearn.metrics.pairwise import haversine_distances

from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



from scipy.optimize import linear_sum_assignment
from scipy import stats



import matplotlib.pyplot as plt

try:

    import folium

    from folium.plugins import AntPath, Fullscreen, MousePosition, MeasureControl, BeautifyIcon

    FOLIUM_AVAILABLE = True

except ModuleNotFoundError:

    folium = None

    AntPath = Fullscreen = MousePosition = MeasureControl = BeautifyIcon = None

    FOLIUM_AVAILABLE = False

    logging.warning("Folium not installed; map visualisations disabled. Install with `pip install folium`.")

try:

    from IPython.display import display

except ModuleNotFoundError:

    def display(*args, **kwargs):

        return None



try:

    from ortools.linear_solver import pywraplp  # noqa: F401

    from ortools.constraint_solver import routing_enums_pb2, pywrapcp  # noqa: F401

except ModuleNotFoundError:

    pywraplp = routing_enums_pb2 = pywrapcp = None

except Exception as exc:  # version mismatches, etc.

    logging.warning("OR-Tools import failed; forcing greedy fallback: %s", exc)

    pywraplp = routing_enums_pb2 = pywrapcp = None



try:

    from shapely.geometry import Point, Polygon

    SHAPELY_AVAILABLE = True

except ModuleNotFoundError:

    SHAPELY_AVAILABLE = False



logging.basicConfig(level=logging.INFO,

                    format="%(asctime)s ▸ %(levelname)s ▸ %(message)s")



FORCE_GREEDY = True

if FORCE_GREEDY:

    pywrapcp = None





# ---------------------------------------------------------------------------

# Configuration & orchestration helpers

# ---------------------------------------------------------------------------

@dataclass

class Config:

    model_dir: Path = Path("models")

    n_territories: int = 20

    n_value_buckets: int = 4

    n_potential_buckets: int = 4

    random_state: int = 42

    nba_lambda_km: float = 5.0

    day_start_min: int = 9 * 60

    avg_speed_kmph: float = 40.0

    service_duration_min: int = 30

    tw_padding_min: int = 60

    gb_classifier_params: Dict[str, Any] = field(default_factory=lambda: {

        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3})

    gb_regressor_params: Dict[str, Any] = field(default_factory=lambda: {

        "n_estimators": 500, "learning_rate": 0.05, "max_depth": 3})

    channels: Tuple[str, ...] = ("field",)

    channel_daily_capacity: Dict[str, int] = field(default_factory=lambda: {"field": 120})

    channel_sla_minutes: Dict[str, int] = field(default_factory=lambda: {"field": 1440})

    script_sections: Tuple[str, ...] = (

        "Context",

        "NBA Rationale",

        "Action Items",

        "Objection Handling",

        "Compliance Reminder"

    )

    prep_script_bullets: int = 4

    max_daily_touchpoints_per_account: int = 2

    kpi_targets: Dict[str, float] = field(default_factory=lambda: {

        "churn_auc": 0.78,

        "potential_rmse": 750.0

    })

    enterprise_checks_enabled: bool = True
    allowed_numeric_nan_cols: Tuple[str, ...] = ("amount",)
    max_allowed_nan_ratio: float = 0.05
    pilot_salespersons: int = 2
    daily_visit_quota: int = 10
    monte_carlo_runs: int = 300
    pilot_days: int = 40
    bootstrap_iterations: int = 2000
    permutation_iterations: int = 5000





@dataclass

class PipelineArtifacts:

    df: pd.DataFrame

    router: Any

    contact_center_schedule: pd.DataFrame

    channel_summary: pd.DataFrame

    channel_tracking: pd.DataFrame

    monitoring: pd.DataFrame

    execution_trace: pd.DataFrame

    validator_issues: List[str]




@dataclass
class PilotExperimentArtifacts:

    selected_reps: List[int]

    route_plan: pd.DataFrame

    unit_metrics: pd.DataFrame

    kpi_summary: pd.DataFrame

    stats_summary: pd.DataFrame





# ---------------------------------------------------------------------------

# Barcelona geographic helper

# ---------------------------------------------------------------------------

def approx_coast_lon(lat: float) -> float:

    a = (2.36 - 2.16) / (41.7 - 41.2)

    b = 2.16 - a * 41.2

    return a * lat + b - 0.01





BARCELONA_POLY = Polygon([

    (41.20, 1.85), (41.20, 2.14), (41.30, 2.18),

    (41.42, 2.22), (41.55, 2.32), (41.68, 2.30),

    (41.70, 2.15), (41.60, 1.90), (41.40, 1.85)

]) if SHAPELY_AVAILABLE else None





# ---------------------------------------------------------------------------

# Synthetic data generation

# ---------------------------------------------------------------------------

def generate_dummy_customers(

    n_customers=2500,

    n_reps=20,

    cfg: Optional[Config] = None,

    mode: str = "gaussian",

    land_only: bool = True,

    use_polygon: bool = False

) -> pd.DataFrame:
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.random_state)

    if mode == "uniform":
        lat = rng.uniform(41.2, 41.7, n_customers * 2)
        lon = rng.uniform(1.8, 2.4, n_customers * 2)
    else:
        centers = np.array([
            [41.385, 2.173], [41.448, 2.208], [41.519, 2.199],
            [41.617, 2.287], [41.360, 2.100], [41.480, 2.082],
            [41.565, 2.022], [41.610, 1.855], [41.300, 2.000],
            [41.400, 1.950]
        ])
        chosen = rng.choice(len(centers), size=n_customers * 3, replace=True)
        lat = centers[chosen, 0] + rng.normal(0, 0.025, chosen.size)
        lon = centers[chosen, 1] + rng.normal(0, 0.03, chosen.size)

    if land_only:
        keep = lon <= np.vectorize(approx_coast_lon)(lat)
        lat, lon = lat[keep], lon[keep]

    if land_only and use_polygon and SHAPELY_AVAILABLE:
        mask_poly = [Point(la, lo).within(BARCELONA_POLY) for la, lo in zip(lat, lon)]
        lat, lon = lat[mask_poly], lon[mask_poly]

    if len(lat) < n_customers:
        logging.warning("Only %d candidates after filtering; trimming.", len(lat))
        n_customers = len(lat)

    lat, lon = lat[:n_customers], lon[:n_customers]

    industries = ["Retail", "Healthcare", "SaaS", "Education", "Manufacturing", "Logistics", "Energy"]
    tiers = list("ABC") + ["Strategic"]
    products = ["CX Suite", "Payments", "Connectivity", "MarTech", "POS", "Analytics"]
    market_clusters = ["NorthAmerica", "Europe", "LATAM", "APAC", "MiddleEastAfrica"]

    company_size = np.clip(np.round(rng.lognormal(mean=5.0, sigma=0.7, size=n_customers)), 20, 8000).astype(int)
    digital_engagement_score = 100.0 * rng.beta(2.0, 2.4, n_customers)
    onsite_engagement_preference = 100.0 * rng.beta(2.8, 1.9, n_customers)
    budget_fit_score = rng.beta(2.4, 2.0, n_customers)
    authority_access_score = rng.beta(2.0, 2.2, n_customers)
    need_intensity_score = rng.beta(2.3, 1.9, n_customers)
    timeline_urgency_score = rng.beta(2.1, 2.0, n_customers)
    strategic_fit_score = rng.beta(2.5, 1.8, n_customers)
    relationship_strength = np.clip(
        100 * (0.50 * rng.beta(2.8, 2.0, n_customers) + 0.50 * budget_fit_score) + rng.normal(0, 4, n_customers),
        0, 100
    )
    value_realization_score = np.clip(
        100 * (0.45 * strategic_fit_score + 0.35 * budget_fit_score + 0.20 * rng.beta(2.0, 2.2, n_customers)),
        0, 100
    )
    competitor_pressure_index = np.clip(
        100 * (0.35 * rng.beta(2.0, 2.0, n_customers) + 0.35 * (1.0 - relationship_strength / 100.0) +
               0.30 * rng.beta(1.8, 2.4, n_customers)),
        0, 100
    )
    account_complexity_index = np.clip(100 * rng.beta(2.2, 1.8, n_customers), 0, 100)
    implementation_risk_score = np.clip(
        100 * (0.45 * rng.beta(2.1, 2.1, n_customers) + 0.30 * account_complexity_index / 100.0 +
               0.25 * (1.0 - value_realization_score / 100.0)),
        0, 100
    )
    product_adoption_index = np.clip(
        100 * (0.50 * digital_engagement_score / 100.0 + 0.30 * relationship_strength / 100.0 +
               0.20 * value_realization_score / 100.0) + rng.normal(0, 4, n_customers),
        0, 100
    )
    meeting_accept_rate = np.clip(
        0.12 + 0.55 * relationship_strength / 100.0 + 0.25 * onsite_engagement_preference / 100.0 -
        0.20 * competitor_pressure_index / 100.0 + rng.normal(0, 0.06, n_customers),
        0.03, 0.98
    )
    multithread_contacts = np.clip(np.round(
        1 + 7 * (0.45 * relationship_strength / 100.0 + 0.35 * strategic_fit_score + 0.20 * rng.random(n_customers))
    ), 1, 8).astype(int)
    buying_committee_size = np.clip(np.round(
        2 + 10 * (0.45 * account_complexity_index / 100.0 + 0.30 * rng.random(n_customers) + 0.25 * strategic_fit_score)
    ), 2, 12).astype(int)
    decision_cycle_days = np.clip(np.round(
        18 + 115 * (0.50 * account_complexity_index / 100.0 + 0.30 * (1.0 - authority_access_score) +
                    0.20 * rng.random(n_customers))
    ), 10, 180).astype(int)
    renewal_window_days = np.clip(np.round(rng.normal(170, 85, n_customers)), 15, 365).astype(int)
    global_market_growth_pct = np.clip(rng.normal(4.5, 4.0, n_customers), -6.0, 16.0)
    nps_score = np.clip(
        15 + 60 * (value_realization_score / 100.0) + 10 * (relationship_strength / 100.0) -
        18 * (competitor_pressure_index / 100.0) + rng.normal(0, 8, n_customers),
        -100, 100
    )
    referenceability_score = np.clip(
        100 * (0.55 * value_realization_score / 100.0 + 0.25 * relationship_strength / 100.0 +
               0.20 * rng.beta(2.0, 2.0, n_customers)),
        0, 100
    )

    support_tickets_90d = rng.poisson(
        np.clip(
            0.4 + 2.2 * implementation_risk_score / 100.0 + 0.8 * account_complexity_index / 100.0,
            0.1,
            6.0
        )
    ).astype(int)
    inbound_calls_30d = rng.poisson(
        np.clip(
            0.6 + 1.4 * support_tickets_90d / 6.0 + 0.5 * competitor_pressure_index / 100.0,
            0.1,
            5.0
        )
    ).astype(int)
    touches_last_30d = rng.poisson(
        np.clip(
            1.2 + 3.8 * meeting_accept_rate + 0.8 * need_intensity_score,
            0.2,
            10.0
        )
    ).astype(int)
    web_sessions_30d = rng.poisson(
        np.clip(
            5.0 + 18.0 * digital_engagement_score / 100.0 + 1.4 * need_intensity_score,
            1.0,
            40.0
        )
    ).astype(int)
    abandoned_carts_30d = rng.poisson(np.clip(0.3 + 0.08 * web_sessions_30d, 0.1, 6.0)).astype(int)
    orders_ecom_30d = rng.poisson(np.clip(0.5 + 0.05 * web_sessions_30d, 0.1, 10.0)).astype(int)
    outbound_connect_rate = np.clip(
        0.08 + 0.70 * meeting_accept_rate + 0.12 * (relationship_strength / 100.0) + rng.normal(0, 0.04, n_customers),
        0.05,
        0.98
    )

    churn_linear = (
        -1.8
        + 1.4 * (competitor_pressure_index / 100.0)
        + 1.0 * (implementation_risk_score / 100.0)
        + 0.45 * (support_tickets_90d / (support_tickets_90d.max() + 1e-9))
        - 0.95 * (relationship_strength / 100.0)
        - 0.70 * (value_realization_score / 100.0)
        - 0.45 * budget_fit_score
        - 0.20 * meeting_accept_rate
        + rng.normal(0, 0.35, n_customers)
    )
    churn_prob_true = 1.0 / (1.0 + np.exp(-churn_linear))
    churn_label = (rng.random(n_customers) < churn_prob_true).astype(int)

    future_spend = (
        500
        + 0.95 * company_size
        + 1800 * budget_fit_score
        + 1700 * need_intensity_score
        + 1200 * strategic_fit_score
        + 850 * (relationship_strength / 100.0)
        + 95 * (global_market_growth_pct + 6.0)
        - 1100 * churn_prob_true
        - 950 * (competitor_pressure_index / 100.0)
        - 700 * (implementation_risk_score / 100.0)
        + rng.normal(0, 480, n_customers)
    )
    future_spend = np.clip(future_spend, 250, 22000)

    amount = np.clip(
        future_spend * rng.uniform(0.10, 0.38, n_customers) + rng.normal(0, 70, n_customers),
        80,
        None
    )
    recency_days = np.clip(np.round(
        25 + 270 * churn_prob_true + 110 * (1.0 - relationship_strength / 100.0) + rng.normal(0, 20, n_customers)
    ), 1, 545).astype(int)
    last_purchase_date = pd.Timestamp.today() - pd.to_timedelta(recency_days, unit="D")
    recent_issue_flag = ((support_tickets_90d >= 2) | (implementation_risk_score > 70)).astype(int)

    csat_score = np.clip(
        4.8
        + 2.2 * (value_realization_score / 100.0)
        + 1.0 * (relationship_strength / 100.0)
        - 0.9 * (support_tickets_90d / (support_tickets_90d.max() + 1e-9))
        - 0.8 * recent_issue_flag
        - 0.5 * (competitor_pressure_index / 100.0)
        + rng.normal(0, 0.55, n_customers),
        0,
        10
    )
    lifetime_value = np.clip(
        future_spend * (1.8 + 1.4 * relationship_strength / 100.0) + rng.normal(0, 900, n_customers),
        300,
        None
    )

    tier_probs = np.column_stack([
        0.45 - 0.25 * strategic_fit_score,
        0.35 + 0.05 * strategic_fit_score,
        0.15 + 0.15 * strategic_fit_score,
        0.05 + 0.05 * strategic_fit_score
    ])
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    tier_idx = [rng.choice(4, p=tier_probs[i]) for i in range(n_customers)]

    df = pd.DataFrame({
        "client_id": np.arange(n_customers),
        "last_purchase_date": last_purchase_date,
        "invoice_id": np.arange(n_customers),
        "amount": amount,
        "churn_label": churn_label,
        "future_spend": future_spend,
        "lat": lat,
        "lon": lon,
        "rep_id": rng.integers(0, n_reps, n_customers),
        "preferred_channel": "field",
        "digital_engagement_score": digital_engagement_score,
        "inbound_calls_30d": inbound_calls_30d,
        "outbound_connect_rate": outbound_connect_rate,
        "web_sessions_30d": web_sessions_30d,
        "abandoned_carts_30d": abandoned_carts_30d,
        "orders_ecom_30d": orders_ecom_30d,
        "support_tickets_90d": support_tickets_90d,
        "csat_score": csat_score,
        "recent_issue_flag": recent_issue_flag,
        "account_tier": [tiers[i] for i in tier_idx],
        "industry": rng.choice(industries, n_customers),
        "company_size": company_size,
        "lifetime_value": lifetime_value,
        "last_contact_channel": "field",
        "touches_last_30d": touches_last_30d,
        "product_interest": rng.choice(products, n_customers),
        "market_cluster": rng.choice(market_clusters, n_customers),
        "budget_fit_score": budget_fit_score,
        "authority_access_score": authority_access_score,
        "need_intensity_score": need_intensity_score,
        "timeline_urgency_score": timeline_urgency_score,
        "multithread_contacts": multithread_contacts,
        "value_realization_score": value_realization_score,
        "relationship_strength": relationship_strength,
        "meeting_accept_rate": meeting_accept_rate,
        "competitor_pressure_index": competitor_pressure_index,
        "implementation_risk_score": implementation_risk_score,
        "global_market_growth_pct": global_market_growth_pct,
        "renewal_window_days": renewal_window_days,
        "product_adoption_index": product_adoption_index,
        "onsite_engagement_preference": onsite_engagement_preference,
        "account_complexity_index": account_complexity_index,
        "decision_cycle_days": decision_cycle_days,
        "buying_committee_size": buying_committee_size,
        "nps_score": nps_score,
        "referenceability_score": referenceability_score,
        "strategic_fit_score": strategic_fit_score
    })

    na_mask = rng.random(n_customers) < 0.02
    df.loc[na_mask, "amount"] = np.nan
    return df





def generate_rep_homes(df: pd.DataFrame, cfg: Optional[Config] = None) -> pd.DataFrame:

    cfg = cfg or Config()

    rng = np.random.default_rng(cfg.random_state + 123)

    reps = sorted(df.rep_id.unique())

    sample_idx = rng.choice(df.index, size=len(reps), replace=False)

    return pd.DataFrame({

        "rep_id": reps,

        "rep_home_lat": df.loc[sample_idx, "lat"].values,

        "rep_home_lon": df.loc[sample_idx, "lon"].values

    })





# ---------------------------------------------------------------------------

# Quality gates & monitoring

# ---------------------------------------------------------------------------

class DataValidator:

    def __init__(
        self,
        required_cols: List[str],
        allowed_numeric_nan_cols: Optional[Tuple[str, ...]] = None,
        max_allowed_nan_ratio: float = 0.0
    ):

        self.required_cols = required_cols
        self.allowed_numeric_nan_cols = set(allowed_numeric_nan_cols or ())
        self.max_allowed_nan_ratio = max_allowed_nan_ratio



    def run(self, df: pd.DataFrame) -> List[str]:

        issues: List[str] = []

        missing = [c for c in self.required_cols if c not in df.columns]

        if missing:

            issues.append(f"Missing columns: {missing}")

        if df.client_id.duplicated().any():

            issues.append("Duplicate client_id detected")

        numeric_cols = df.select_dtypes(include=[np.number])

        nan_counts = numeric_cols.isnull().sum()
        nan_counts = nan_counts[nan_counts > 0]
        if not nan_counts.empty:
            for col, count in nan_counts.items():
                ratio = float(count) / float(len(df))
                if col not in self.allowed_numeric_nan_cols:
                    issues.append(f"Unexpected NaN values in numeric column '{col}' ({ratio:.2%})")
                    continue
                if ratio > self.max_allowed_nan_ratio:
                    issues.append(
                        f"Column '{col}' NaN ratio {ratio:.2%} exceeds allowed "
                        f"{self.max_allowed_nan_ratio:.2%}"
                    )

        if {"lat", "lon"}.issubset(df.columns):

            if (df["lat"].isnull() | df["lon"].isnull()).any():

                issues.append("Lat/Lon missing values")

        if issues:

            logging.warning("Quality gate raised %d issues", len(issues))

        else:

            logging.info("Quality gate passed")

        return issues





class MonitoringSuite:

    def __init__(self, cfg: Config):

        self.cfg = cfg

        self.records: List[Dict[str, Any]] = []



    def record(self, metric: str, value: float, details: Optional[str] = None) -> None:

        self.records.append({

            "timestamp": datetime.utcnow(),

            "metric": metric,

            "value": value,

            "target": self.cfg.kpi_targets.get(metric),

            "details": details or ""

        })



    def as_frame(self) -> pd.DataFrame:

        if not self.records:

            return pd.DataFrame(columns=["timestamp", "metric", "value", "target", "details"])

        return pd.DataFrame(self.records)





class ExecutionTracer:

    """Collects lightweight execution events for notebook inspection."""



    def __init__(self):

        self.records: List[Dict[str, Any]] = []



    def log(self, phase: str, message: str, **extra: Any) -> None:

        payload = {"timestamp": datetime.utcnow(), "phase": phase, "message": message}

        payload.update(extra)

        self.records.append(payload)

        logging.info("[%s] %s %s", phase, message, extra if extra else "")



    def as_frame(self) -> pd.DataFrame:

        if not self.records:

            return pd.DataFrame(columns=["timestamp", "phase", "message"])

        return pd.DataFrame(self.records)





# ---------------------------------------------------------------------------

# Predictive models & scoring

# ---------------------------------------------------------------------------

class PredictiveModels:

    def __init__(self, cfg: Config, monitor: Optional[MonitoringSuite] = None):

        self.cfg = cfg

        self.monitor = monitor

        self.churn_model = GradientBoostingClassifier(**cfg.gb_classifier_params,

                                                      random_state=cfg.random_state)

        self.potential_model = GradientBoostingRegressor(**cfg.gb_regressor_params,

                                                         random_state=cfg.random_state)

        self.churn_features_: List[str] = []

        self.potential_features_: List[str] = []

        self.imputer_ = SimpleImputer(strategy="median")



    def _num(self, df: pd.DataFrame) -> pd.DataFrame:

        return df.select_dtypes(include=[np.number])



    def _prep_Xy(self, df: pd.DataFrame, target: str, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:

        num = self._num(df)

        X = num.drop(columns=drop_cols, errors="ignore")

        y = num[target]

        return X, y



    def train_churn(self, df: pd.DataFrame) -> float:

        X, y = self._prep_Xy(df, "churn_label",

                             ["client_id", "churn_label", "future_spend"])

        self.churn_features_ = X.columns.tolist()

        X_imp = pd.DataFrame(self.imputer_.fit_transform(X), columns=X.columns)

        X_tr, X_val, y_tr, y_val = train_test_split(

            X_imp, y, test_size=0.2, random_state=self.cfg.random_state)

        self.churn_model.fit(X_tr, y_tr)

        auc = roc_auc_score(y_val, self.churn_model.predict_proba(X_val)[:, 1])

        if self.monitor:

            self.monitor.record("churn_auc", auc)

        logging.info("Churn model AUC = %.3f", auc)

        return auc



    def train_potential(self, df: pd.DataFrame) -> float:

        X, y = self._prep_Xy(df, "future_spend",

                             ["client_id", "future_spend", "churn_label"])

        self.potential_features_ = X.columns.tolist()

        X_imp = pd.DataFrame(self.imputer_.fit_transform(X), columns=X.columns)

        X_tr, X_val, y_tr, y_val = train_test_split(

            X_imp, y, test_size=0.2, random_state=self.cfg.random_state)

        self.potential_model.fit(X_tr, y_tr)

        rmse = mean_squared_error(y_val, self.potential_model.predict(X_val), squared=False)

        if self.monitor:

            self.monitor.record("potential_rmse", rmse)

        logging.info("Potential model RMSE = %.2f", rmse)

        return rmse



    def predict(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        Xc = df[self.churn_features_]

        Xc_imp = pd.DataFrame(self.imputer_.transform(Xc), columns=Xc.columns)

        df["churn_prob"] = self.churn_model.predict_proba(Xc_imp)[:, 1]



        Xp = df[self.potential_features_]

        Xp_imp = pd.DataFrame(self.imputer_.transform(Xp), columns=Xp.columns)

        df["predicted_potential"] = self.potential_model.predict(Xp_imp)

        return df





class Segmentation:

    def __init__(self, cfg: Config):

        self.cfg = cfg



    def segment(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        if "monetary_6m" not in df.columns:

            df["monetary_6m"] = df.groupby("client_id")["amount"].transform("sum")

        df["value_bucket"] = pd.qcut(df["monetary_6m"],

                                      self.cfg.n_value_buckets,

                                      labels=False, duplicates="drop")

        df["potential_bucket"] = pd.qcut(df["predicted_potential"],

                                          self.cfg.n_potential_buckets,

                                          labels=False, duplicates="drop")

        df["segment"] = df["value_bucket"].astype(str) + df["potential_bucket"].astype(str)

        return df





class NextBestAction:

    def __init__(self, cfg: Config):

        self.cfg = cfg

        self.weights = {

            "segment": 0.10,

            "churn_prob": 0.30,

            "recency_days": 0.10,

            "predicted_potential": 0.30,

            "digital_engagement_score": 0.10,

            "support_tickets_90d": 0.10

        }



    def _ensure_recency(self, df: pd.DataFrame) -> pd.DataFrame:

        if "recency_days" not in df.columns and "last_purchase_date" in df.columns:

            df = df.copy()

            df["recency_days"] = (pd.Timestamp.today() - pd.to_datetime(df["last_purchase_date"])).dt.days

        elif "recency_days" not in df.columns:

            df["recency_days"] = 0

        return df



    def score(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self._ensure_recency(df).copy()

        z_cols = ["churn_prob", "recency_days", "predicted_potential",

                  "digital_engagement_score", "support_tickets_90d"]

        for col in z_cols:

            df[f"{col}_z"] = StandardScaler().fit_transform(df[[col]])

        df["segment_int"] = df["segment"].astype(int)

        df["segment_z"] = StandardScaler().fit_transform(df[["segment_int"]])

        df["nba_score"] = (

            self.weights["segment"] * df["segment_z"] +

            self.weights["churn_prob"] * df["churn_prob_z"] +

            self.weights["recency_days"] * df["recency_days_z"] +

            self.weights["predicted_potential"] * df["predicted_potential_z"] +

            self.weights["digital_engagement_score"] * df["digital_engagement_score_z"] +

            self.weights["support_tickets_90d"] * df["support_tickets_90d_z"]

        )

        return df





# ---------------------------------------------------------------------------

# Territory design & anomaly detection

# ---------------------------------------------------------------------------

class TerritoryDesign:

    def __init__(self, cfg: Config):

        self.cfg = cfg



    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:

        km = KMeans(n_clusters=self.cfg.n_territories, random_state=self.cfg.random_state)

        df = df.copy()

        df["territory_cluster"] = km.fit_predict(df[["lat", "lon"]].to_numpy())

        self.kmeans = km

        return df



    def match_reps_to_centroids(self, rep_homes: pd.DataFrame) -> pd.DataFrame:

        homes = rep_homes[["rep_home_lat", "rep_home_lon"]].to_numpy()

        cents = self.kmeans.cluster_centers_

        rad = haversine_distances(np.radians(homes), np.radians(cents))

        cost = rad * 6371.0

        row_ind, col_ind = linear_sum_assignment(cost)

        return pd.DataFrame({

            "rep_id": rep_homes["rep_id"].values[row_ind],

            "territory_cluster": col_ind,

            "distance_km": cost[row_ind, col_ind]

        })





def get_territory_customers(df: pd.DataFrame, rep_id: int) -> pd.DataFrame:

    rep_territory = df.loc[df.rep_id == rep_id, "rep_territory"].iloc[0]

    return df[df.territory_cluster == rep_territory]





class AnomalyDetection:

    def __init__(self, features: List[str], contamination=0.02, random_state=42):

        self.features = features

        self.model = IsolationForest(contamination=contamination,

                                     random_state=random_state)



    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        X = df[self.features].to_numpy()

        self.model.fit(X)

        scores = -self.model.decision_function(X)

        df["anomaly_score"] = scores

        df["is_anomaly"] = self.model.predict(X) == -1

        return df





# ---------------------------------------------------------------------------

# Channel allocator & prep scripts

# ---------------------------------------------------------------------------

class ChannelAllocator:

    def __init__(self, cfg: Config):

        self.cfg = cfg



    @staticmethod

    def _norm(series: pd.Series) -> pd.Series:

        return (series - series.min()) / (series.max() - series.min() + 1e-9)



    def assign(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        onsite = self._norm(df.get("onsite_engagement_preference", pd.Series(50, index=df.index)))
        relationship = self._norm(df.get("relationship_strength", pd.Series(50, index=df.index)))
        df["field_score"] = (
            0.35 * self._norm(df["predicted_potential"]) +
            0.25 * self._norm(df["nba_score"]) +
            0.20 * onsite +
            0.15 * relationship +
            0.05 * (1 - self._norm(df["digital_engagement_score"]))
        )
        df["recommended_channel"] = "field"
        df["channel_confidence"] = df["field_score"]
        df["channel_reason"] = (
            "Field-only strategy | NBA=" + df["nba_score"].round(2).astype(str) +
            ", potential=" + df["predicted_potential"].round(0).astype(int).astype(str)
        )
        return df





def build_channel_tracking(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:

    """Summarise NBA load per channel before scheduling."""

    base_cols = ["channel", "clients", "avg_nba", "avg_churn",

                 "avg_potential", "avg_confidence", "anomalies"]

    if df.empty or "recommended_channel" not in df.columns:

        return pd.DataFrame({

            "channel": list(cfg.channels),

            "clients": 0,

            "avg_nba": 0.0,

            "avg_churn": 0.0,

            "avg_potential": 0.0,

            "avg_confidence": 0.0,

            "anomalies": 0

        })[base_cols]



    summary = (

        df.groupby("recommended_channel")

          .agg(clients=("client_id", "count"),

               avg_nba=("nba_score", "mean"),

               avg_churn=("churn_prob", "mean"),

               avg_potential=("predicted_potential", "mean"),

               avg_confidence=("channel_confidence", "mean"),

               anomalies=("is_anomaly", "sum"))

          .reset_index()

          .rename(columns={"recommended_channel": "channel"})

    )

    summary = summary.set_index("channel").reindex(cfg.channels, fill_value=0).reset_index()

    return summary[base_cols]





class PrepScriptBuilder:

    def __init__(self, cfg: Config):

        self.cfg = cfg



    def _bullets(self, row: pd.Series) -> Dict[str, List[str]]:

        context = [

            f"Account tier {row.account_tier} ({row.industry}) with {row.company_size} FTEs.",

            f"Last touch on {row.last_contact_channel}; {row.touches_last_30d} touches in 30d.",

            f"Lifetime value €{row.lifetime_value:,.0f}; segment {row.segment}."

        ]

        rationale = [

            f"NBA score {row.nba_score:.2f} driven by churn {row.churn_prob:.2f} and potential €{row.predicted_potential:,.0f}.",

            f"Digital engagement {row.digital_engagement_score:.1f} and {row.orders_ecom_30d} ecommerce orders support offer cross-sell {row.product_interest}."

        ]

        action = [

            f"Lead with {row.product_interest} value prop and reference recent spend €{row.amount:,.0f}.",

            f"Confirm readiness for next quarter budget; align on preferred channel {row.recommended_channel}."

        ]

        if row.recent_issue_flag:

            objection = ["Address recent support pain point before upsell."]

        else:

            objection = [f"Probe for hidden blockers and reference CSAT {row.csat_score:.1f}."]

        compliance = [

            "Log consent for data usage and confirm GDPR-compliant recording.",

            f"Respect SLA of {self.cfg.channel_sla_minutes.get(row.recommended_channel,0)} mins for channel response."

        ]

        return {

            "Context": context,

            "NBA Rationale": rationale,

            "Action Items": action,

            "Objection Handling": objection,

            "Compliance Reminder": compliance

        }



    def build_script(self, row: pd.Series) -> str:

        sections = self._bullets(row)

        lines: List[str] = []

        for section in self.cfg.script_sections:

            lines.append(f"{section}:")

            for bullet in sections.get(section, [])[:self.cfg.prep_script_bullets]:

                bullet_fmt = textwrap.fill(f"- {bullet}", width=92, subsequent_indent="  ")

                lines.append(bullet_fmt)

        return "\n".join(lines)



    def create_scripts(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df["prep_script"] = df.apply(self.build_script, axis=1)

        return df





# ---------------------------------------------------------------------------

# Contact center scheduler

# ---------------------------------------------------------------------------

class ContactCenterScheduler:

    def __init__(self, cfg: Config):

        self.cfg = cfg



    def build_daily_plan(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        schedules = []

        summary_rows = []

        for channel in self.cfg.channels:

            channel_df = df[df["recommended_channel"] == channel]

            cap = self.cfg.channel_daily_capacity.get(channel, len(channel_df))

            scheduled = channel_df.sort_values("nba_score", ascending=False).head(cap).copy()

            scheduled["scheduled_channel"] = channel

            scheduled["scheduled_slot"] = [f"{channel}-{i+1}" for i in range(len(scheduled))]

            schedules.append(scheduled)

            backlog = max(len(channel_df) - cap, 0)

            summary_rows.append({

                "channel": channel,

                "demand": len(channel_df),

                "scheduled": len(scheduled),

                "backlog": backlog,

                "sla_minutes": self.cfg.channel_sla_minutes.get(channel, np.nan),

                "sla_met": backlog == 0

            })

        schedule_df = pd.concat(schedules).sort_values("nba_score", ascending=False) if schedules else pd.DataFrame()

        summary_df = pd.DataFrame(summary_rows)

        return schedule_df, summary_df





# ---------------------------------------------------------------------------

# Router (field reps) with NBA reward

# ---------------------------------------------------------------------------

class RouteOptimizer:

    def __init__(self, cfg: Config):

        self.cfg = cfg



    @staticmethod

    def _gc_km(a: np.ndarray, b: np.ndarray) -> float:

        return float(haversine_distances(np.radians(a[None, :]),

                                         np.radians(b[None, :]))[0, 0] * 6371.0)



    def _select_daily_targets(self, df_rep: pd.DataFrame) -> pd.DataFrame:

        terr = df_rep["rep_territory"].iloc[0]

        cand = df_rep[df_rep["territory_cluster"] == terr]

        cand = cand if len(cand) >= 10 else df_rep

        home = cand[["rep_home_lat", "rep_home_lon"]].iloc[0].to_numpy()

        cand = cand.copy()

        cand["dist_home"] = cand[["lat", "lon"]].apply(

            lambda row: self._gc_km(home, row.to_numpy()), axis=1)

        return cand.sort_values(["nba_score", "dist_home"], ascending=[False, True]).head(self.cfg.daily_visit_quota)



    def _compute_time_windows(self, day_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:

        day_df = day_df.copy()

        mid = rng.integers(630, 930, len(day_df))

        half = self.cfg.tw_padding_min

        day_df["tw_start"] = mid - half

        day_df["tw_end"] = mid + half

        return day_df



    def optimise_for_rep(self, df_rep: pd.DataFrame, return_decisions: bool = False

                          ) -> Tuple[List[int], List[int], Optional[List[pd.DataFrame]]]:

        if len(df_rep) == 0:

            return [], [], None



        rng = np.random.default_rng(self.cfg.random_state + int(df_rep.rep_id.iloc[0]))

        day_df = self._select_daily_targets(df_rep)

        day_df = self._compute_time_windows(day_df, rng)



        home = day_df[["rep_home_lat", "rep_home_lon"]].iloc[0].to_numpy()

        pts = np.vstack([home, day_df[["lat", "lon"]].to_numpy()])

        nba = day_df["nba_score"].to_numpy()



        rad = haversine_distances(np.radians(pts), np.radians(pts))

        dist_km = rad * 6371.0

        nba_norm = (nba - nba.min()) / (nba.max() - nba.min() + 1e-9)

        reward_km_vec = self.cfg.nba_lambda_km * nba_norm



        cost = dist_km.copy()

        cost[:, 1:1 + len(nba)] -= reward_km_vec

        cost_int = np.round(cost * 1000).astype(int)



        remaining = list(range(1, len(pts)))

        route, cur = [], 0

        decisions = []

        current_time = self.cfg.day_start_min



        while remaining:

            nxt = min(remaining, key=lambda j: cost_int[cur, j])

            rows = []

            for j in remaining:

                df_idx = day_df.index[j - 1]

                pure = dist_km[cur, j]

                reward = reward_km_vec[j - 1]

                eff = cost_int[cur, j] / 1000.0

                travel_min = pure / self.cfg.avg_speed_kmph * 60.0

                arrive = current_time + travel_min

                tw_s = day_df.loc[df_idx, "tw_start"]

                tw_e = day_df.loc[df_idx, "tw_end"]

                start_service = max(arrive, tw_s)

                finish = start_service + self.cfg.service_duration_min

                feasible = finish <= tw_e

                rows.append({

                    "step": len(route) + 1,

                    "client_id": day_df.loc[df_idx, "client_id"],

                    "pure_dist_km": pure,

                    "reward_km": reward,

                    "effective_cost_km": eff,

                    "arrival_min": arrive,

                    "finish_min": finish,

                    "tw_start": tw_s,

                    "tw_end": tw_e,

                    "feasible": feasible,

                    "chosen_flag": False

                })

            snap = pd.DataFrame(rows).sort_values("effective_cost_km").reset_index(drop=True)

            chosen_client = day_df.index[nxt - 1]

            snap.loc[snap.client_id == day_df.loc[chosen_client, "client_id"], "chosen_flag"] = True

            decisions.append(snap)



            route.append(chosen_client)

            chosen_row = snap[snap["chosen_flag"]].iloc[0]

            current_time = chosen_row["finish_min"]

            remaining.remove(nxt)

            cur = nxt



        if return_decisions:

            for d in decisions:

                display(d.style.format({

                    "pure_dist_km": "{:.2f}",

                    "reward_km": "{:.2f}",

                    "effective_cost_km": "{:.2f}",

                    "arrival_min": "{:.1f}",

                    "finish_min": "{:.1f}"

                }).set_caption(f"Decision before choosing stop #{int(d.step.iloc[0])}"))

        return route, list(day_df.index), (decisions if return_decisions else None)





# ---------------------------------------------------------------------------

# Visualisations

# ---------------------------------------------------------------------------

def viz_segmentation(df: pd.DataFrame):

    plt.figure()

    plt.scatter(df["monetary_6m"], df["predicted_potential"],

                c=df["segment"].astype(int), s=15)

    plt.colorbar(label="Segment")

    plt.title("Value vs Potential")

    plt.xlabel("Monetary 6m"); plt.ylabel("Predicted Potential")

    plt.tight_layout(); plt.show()





def viz_territories(df: pd.DataFrame):

    plt.figure()

    plt.scatter(df["lon"], df["lat"], c=df["territory_cluster"], s=10)

    plt.title("Territory Clusters (KMeans)")

    plt.xlabel("Longitude"); plt.ylabel("Latitude")

    plt.tight_layout(); plt.show()





def viz_roc(df: pd.DataFrame):

    fpr, tpr, _ = roc_curve(df["churn_label"], df["churn_prob"])

    plt.figure(); plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], "--", lw=0.5)

    plt.xlabel("FPR"); plt.ylabel("TPR")

    plt.title("Churn ROC Curve")

    plt.tight_layout(); plt.show()





def viz_nba_hist(df: pd.DataFrame):

    plt.figure()

    plt.hist(df["nba_score"], bins=25)

    plt.title("NBA Score Distribution")

    plt.xlabel("NBA score"); plt.ylabel("Count")

    plt.tight_layout(); plt.show()





def viz_anomalies(df: pd.DataFrame):

    plt.figure(figsize=(6, 4))

    normal = df[~df.is_anomaly]

    anom = df[df.is_anomaly]

    plt.scatter(normal["nba_score"], normal["anomaly_score"], s=15, alpha=0.3, label="Normal")

    plt.scatter(anom["nba_score"], anom["anomaly_score"], s=30, color="red", marker="x", label="Anomaly")

    plt.xlabel("NBA Score"); plt.ylabel("Anomaly Score")

    plt.legend(); plt.tight_layout(); plt.show()





def viz_anomaly_map(df: pd.DataFrame):

    plt.figure(figsize=(6, 4))

    normal = df[~df.is_anomaly]

    anom = df[df.is_anomaly]

    plt.scatter(normal["lon"], normal["lat"], s=10, alpha=0.25, label="Normal")

    plt.scatter(anom["lon"], anom["lat"], s=25, color="red", label="Anomaly")

    plt.xlabel("Longitude"); plt.ylabel("Latitude")

    plt.title("Geospatial Anomalies")

    plt.legend()

    plt.tight_layout(); plt.show()





def viz_route(df: pd.DataFrame, rep_id: int,

              order: List[int], day_pool: List[int]):

    territory_df = get_territory_customers(df, rep_id)

    home = df.loc[df.rep_id == rep_id, ["rep_home_lat", "rep_home_lon"]].iloc[0]

    home_lat, home_lon = home

    ordered_df = df.loc[order]

    pool_df = df.loc[day_pool]

    lats = [home_lat] + ordered_df["lat"].tolist() + [home_lat]

    lons = [home_lon] + ordered_df["lon"].tolist() + [home_lon]



    plt.figure(figsize=(6, 6))

    plt.scatter(territory_df["lon"], territory_df["lat"],

                s=12, color="#999999", alpha=0.25, label="Territory (all)")

    plt.scatter(pool_df["lon"], pool_df["lat"],

                s=40, color="#1f77b4", alpha=0.7, label="Daily pool (10)")

    plt.plot(lons, lats, "-o", color="#d62728", lw=1.8, ms=6, label="Route")



    plt.scatter(home_lon, home_lat, marker="s", s=110, color="#2ca02c", label="Home")



    for step, idx in enumerate(order, start=1):

        row = df.loc[idx]

        plt.annotate(f"{step}\n{int(row.client_id)}",

                     (row.lon, row.lat),

                     xytext=(4, 4), textcoords="offset points", fontsize=7)



    plt.title(f"Rep {rep_id} – Territory / Pool / Route")

    plt.xlabel("Longitude"); plt.ylabel("Latitude")

    plt.legend(fontsize=8)

    plt.tight_layout(); plt.show()



    visit_order = (

        df.loc[order, ["client_id", "lat", "lon", "nba_score", "recommended_channel"]]

          .assign(stop=lambda t: np.arange(1, len(t) + 1))

          .loc[:, ["stop", "client_id", "nba_score", "recommended_channel", "lat", "lon"]]

    )

    display(visit_order.style.format({"nba_score": "{:.2f}",

                                      "lat": "{:.4f}", "lon": "{:.4f}"}))





# ---------------------------------------------------------------------------

# Folium map

# ---------------------------------------------------------------------------

def fancy_route_map(df: pd.DataFrame,

                    rep_id: int,

                    order: List[int],

                    day_pool: List[int]) -> "folium.Map":



    if not FOLIUM_AVAILABLE:

        logging.warning("Folium not installed; returning None for map. Install with `pip install folium`.")

        return None



    rep_home = df.loc[df.rep_id == rep_id, ["rep_home_lat", "rep_home_lon"]].iloc[0]

    start = (rep_home.rep_home_lat, rep_home.rep_home_lon)

    territory_df = get_territory_customers(df, rep_id)

    pool_df = df.loc[day_pool]

    ordered_df = df.loc[order]



    fmap = folium.Map(location=start, zoom_start=10,

                      width="100%", height="600px",

                      control_scale=True, tiles="Esri.WorldImagery")



    terr_layer = folium.FeatureGroup(name="Territory (all)", show=True)

    for _, r in territory_df.iterrows():

        folium.CircleMarker(

            location=(r.lat, r.lon), radius=3,

            color="#888888", fill=True, fill_opacity=0.25, weight=0

        ).add_to(terr_layer)

    terr_layer.add_to(fmap)



    pool_layer = folium.FeatureGroup(name="Candidate pool (10)", show=True)

    for _, r in pool_df.iterrows():

        folium.CircleMarker(

            location=(r.lat, r.lon), radius=6,

            color="#1f77b4", fill=True, fill_opacity=0.8, weight=1

        ).add_to(pool_layer)

    pool_layer.add_to(fmap)



    route_layer = folium.FeatureGroup(name="Route (ordered)", show=True)

    for step, (_, r) in enumerate(ordered_df.iterrows(), start=1):

        marker = folium.Marker(

            location=(r.lat, r.lon),

            popup=folium.Popup(

                f"<b>Stop {step}</b><br>Client {r.client_id}<br>NBA {r.nba_score:.2f}<br>Channel {r.recommended_channel}",

                max_width=250)

        )

        marker.add_to(route_layer)

        BeautifyIcon(

            icon_shape="marker", number=step, border_color="#d62728",

            background_color="#d62728", text_color="#ffffff"

        ).add_to(marker)

    route_layer.add_to(fmap)



    path_coords = [start] + ordered_df[["lat", "lon"]].values.tolist() + [start]

    AntPath(path_coords, weight=5, color="#FF8C00", dash_array=[10, 20]).add_to(fmap)



    folium.Marker(location=start,

                  popup=f"<b>Rep {rep_id} Home</b>",

                  icon=folium.Icon(color="green", icon="home", prefix="fa")).add_to(fmap)



    MousePosition(position="bottomright", separator=" , ", prefix="Lat/Lon").add_to(fmap)

    MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(fmap)

    Fullscreen().add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    return fmap





# ---------------------------------------------------------------------------

# Route comparison map (NBA vs distance-only)

# ---------------------------------------------------------------------------

def compare_route_map(df: pd.DataFrame,

                      rep_id: int,

                      nba_order: List[int],

                      dist_order: List[int],

                      day_pool: List[int],

                      primary_label: str = "Algorithmic route",

                      secondary_label: str = "Heuristic route") -> "folium.Map":



    if not FOLIUM_AVAILABLE:

        logging.warning("Folium not installed; returning None for map. Install with `pip install folium`.")

        return None



    rep_home = df.loc[df.rep_id == rep_id, ["rep_home_lat", "rep_home_lon"]].iloc[0]

    start = (rep_home.rep_home_lat, rep_home.rep_home_lon)

    territory_df = get_territory_customers(df, rep_id)

    pool_df = df.loc[day_pool]

    nba_df = df.loc[nba_order]

    dist_df = df.loc[dist_order]



    fmap = folium.Map(location=start, zoom_start=10,

                      width="100%", height="600px",

                      control_scale=True, tiles="Esri.WorldImagery")



    terr_layer = folium.FeatureGroup(name="Territory (all)", show=True)

    for _, r in territory_df.iterrows():

        folium.CircleMarker(

            location=(r.lat, r.lon), radius=3,

            color="#888888", fill=True, fill_opacity=0.25, weight=0

        ).add_to(terr_layer)

    terr_layer.add_to(fmap)



    pool_layer = folium.FeatureGroup(name="Candidate pool (10)", show=True)

    for _, r in pool_df.iterrows():

        folium.CircleMarker(

            location=(r.lat, r.lon), radius=6,

            color="#1f77b4", fill=True, fill_opacity=0.8, weight=1

        ).add_to(pool_layer)

    pool_layer.add_to(fmap)



    nba_layer = folium.FeatureGroup(name=primary_label, show=True)

    dist_layer = folium.FeatureGroup(name=secondary_label, show=True)

    dist_rank = {idx: i + 1 for i, idx in enumerate(dist_order)}



    for step, (idx, r) in enumerate(nba_df.iterrows(), start=1):

        dist_stop = dist_rank.get(idx)

        popup_parts = [

            f"<b>NBA stop {step}</b>",

            f"Heuristic stop {dist_stop}" if dist_stop is not None else None,

            f"Client {int(r.client_id)}",

            f"NBA {r.nba_score:.2f}",

            f"Channel {r.recommended_channel}"

        ]

        popup = "<br>".join([p for p in popup_parts if p is not None])

        marker = folium.Marker(

            location=(r.lat, r.lon),

            popup=folium.Popup(popup, max_width=260)

        )

        marker.add_to(nba_layer)

        BeautifyIcon(

            icon_shape="marker", number=step, border_color="#d62728",

            background_color="#d62728", text_color="#ffffff"

        ).add_to(marker)

    nba_layer.add_to(fmap)



    nba_path = [start] + nba_df[["lat", "lon"]].values.tolist() + [start]

    AntPath(nba_path, weight=5, color="#FF8C00", dash_array=[10, 20]).add_to(nba_layer)



    dist_path = [start] + dist_df[["lat", "lon"]].values.tolist() + [start]

    folium.PolyLine(dist_path, color="#1f77b4", weight=3, opacity=0.8).add_to(dist_layer)

    dist_layer.add_to(fmap)



    folium.Marker(location=start,

                  popup=f"<b>Rep {rep_id} Home</b>",

                  icon=folium.Icon(color="green", icon="home", prefix="fa")).add_to(fmap)



    MousePosition(position="bottomright", separator=" , ", prefix="Lat/Lon").add_to(fmap)

    MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(fmap)

    Fullscreen().add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    return fmap





# ---------------------------------------------------------------------------

# Main orchestrator

# ---------------------------------------------------------------------------

def main(cfg: Optional[Config] = None,

         n_customers=3000,

         barcelona_mode="gaussian",

         show_initial_visuals=False) -> PipelineArtifacts:

    cfg = cfg or Config()

    monitor = MonitoringSuite(cfg)

    trace = ExecutionTracer()

    trace.log("init", "Starting OptimAI run", customers=n_customers)



    df = generate_dummy_customers(n_customers=n_customers,

                                  n_reps=cfg.n_territories,

                                  cfg=cfg,

                                  mode=barcelona_mode,

                                  land_only=True,

                                  use_polygon=False)

    trace.log("data_generation", "Synthetic dataset ready", rows=len(df))



    validator = DataValidator(
        required_cols=["client_id", "lat", "lon", "amount"],
        allowed_numeric_nan_cols=cfg.allowed_numeric_nan_cols,
        max_allowed_nan_ratio=cfg.max_allowed_nan_ratio
    )

    validator_issues = validator.run(df) if cfg.enterprise_checks_enabled else []

    monitor.record("rows_generated", float(len(df)))

    trace.log("quality", "Validator completed", issues=len(validator_issues))



    rep_homes = generate_rep_homes(df, cfg=cfg)

    trace.log("reps", "Rep homes sampled", reps=len(rep_homes))



    models = PredictiveModels(cfg, monitor=monitor)

    churn_auc = models.train_churn(df)

    potential_rmse = models.train_potential(df)

    trace.log("model_training", "Models trained",

              churn_auc=float(churn_auc), potential_rmse=float(potential_rmse))

    df = models.predict(df)

    trace.log("model_scoring", "Predictions generated", columns=df.shape[1])



    df = Segmentation(cfg).segment(df)

    df = NextBestAction(cfg).score(df)

    trace.log("nba", "Segmentation + NBA scoring complete",

              nba_min=float(df["nba_score"].min()), nba_max=float(df["nba_score"].max()))



    td = TerritoryDesign(cfg)

    df = td.cluster(df)

    mapping = td.match_reps_to_centroids(rep_homes)

    df = df.merge(rep_homes, on="rep_id", how="left")

    df = df.merge(mapping[["rep_id", "territory_cluster"]]

                  .rename(columns={"territory_cluster": "rep_territory"}),

                  on="rep_id", how="left")

    trace.log("territories", "Territories assigned", territories=cfg.n_territories)



    df = AnomalyDetection(features=["nba_score", "churn_prob",

                                    "predicted_potential", "recency_days"],

                          contamination=0.02,

                          random_state=cfg.random_state).fit_predict(df)

    trace.log("anomalies", "Anomaly detection flagged accounts",

              anomaly_count=int(df["is_anomaly"].sum()))



    df = ChannelAllocator(cfg).assign(df)

    trace.log("channel_allocator", "Channels recommended",

              top_channel=str(df["recommended_channel"].mode().iloc[0]))

    df = PrepScriptBuilder(cfg).create_scripts(df)

    trace.log("prep_scripts", "Preparation scripts built", accounts=len(df))



    channel_tracking = build_channel_tracking(df, cfg)

    channel_tracking["capacity"] = channel_tracking["channel"].map(cfg.channel_daily_capacity).fillna(0).astype(int)

    scheduler = ContactCenterScheduler(cfg)

    schedule_df, schedule_summary = scheduler.build_daily_plan(df)

    if not schedule_summary.empty:

        channel_tracking = channel_tracking.merge(

            schedule_summary[["channel", "scheduled", "backlog", "sla_minutes", "sla_met"]],

            on="channel", how="left")

    else:

        channel_tracking = channel_tracking.assign(

            scheduled=0,

            backlog=0,

            sla_minutes=channel_tracking["channel"].map(cfg.channel_sla_minutes),

            sla_met=False)

    for col in ["scheduled", "backlog"]:

        channel_tracking[col] = channel_tracking[col].fillna(0).astype(int)

    channel_tracking["sla_minutes"] = channel_tracking["sla_minutes"].fillna(

        channel_tracking["channel"].map(cfg.channel_sla_minutes)).astype(int)

    channel_tracking["sla_met"] = channel_tracking["sla_met"].fillna(True)

    trace.log("scheduler", "Contact center plan created",

              scheduled=int(schedule_summary["scheduled"].sum()) if not schedule_summary.empty else 0,

              backlog=int(schedule_summary["backlog"].sum()) if not schedule_summary.empty else 0)



    router = RouteOptimizer(cfg)

    trace.log("routing", "Route optimiser initialised", greedy=FORCE_GREEDY)



    if show_initial_visuals:

        viz_segmentation(df)

        viz_territories(df)

        viz_roc(df)

        viz_nba_hist(df)

        viz_anomalies(df)

        viz_anomaly_map(df)

        first_rep = df.rep_id.iloc[0]

        order, pool, _ = router.optimise_for_rep(df[df.rep_id == first_rep])

        viz_route(df, first_rep, order, pool)

        trace.log("visuals", "Initial visualisations rendered", rep=int(first_rep))



    trace.log("complete", "OptimAI pipeline ready", total_accounts=len(df))

    return PipelineArtifacts(

        df=df,

        router=router,

        contact_center_schedule=schedule_df,

        channel_summary=schedule_summary,

        channel_tracking=channel_tracking,

        monitoring=monitor.as_frame(),

        execution_trace=trace.as_frame(),

        validator_issues=validator_issues

    )


def route_divergence(order_a: List[int], order_b: List[int]) -> int:

    if not order_a or not order_b:

        return 0

    rank_b = {idx: i for i, idx in enumerate(order_b)}

    return sum(abs(i - rank_b.get(idx, i)) for i, idx in enumerate(order_a))


def build_route_compare_table(
    df: pd.DataFrame,
    nba_order: List[int],
    dist_order: List[int]
) -> pd.DataFrame:

    dist_rank = {idx: i + 1 for i, idx in enumerate(dist_order)}

    rows = []

    for step, idx in enumerate(nba_order, start=1):

        row = df.loc[idx]

        rows.append({

            "nba_stop": step,

            "dist_stop": dist_rank.get(idx),

            "client_id": int(row.client_id),

            "nba_score": float(row.nba_score),

            "channel": row.recommended_channel

        })

    return pd.DataFrame(rows)


def _normalise(values: pd.Series) -> pd.Series:

    span = values.max() - values.min()

    if span <= 1e-12:

        return pd.Series(np.zeros(len(values)), index=values.index)

    return (values - values.min()) / span


def _sigmoid(x: float) -> float:

    return 1.0 / (1.0 + np.exp(-x))


def _build_route_legs(
    df: pd.DataFrame,
    rep_id: int,
    order: List[int],
    cfg: Config
) -> pd.DataFrame:

    if not order:

        return pd.DataFrame(columns=[
            "rep_id", "stop", "row_index", "client_id", "leg_km",
            "travel_min", "return_home_km", "route_total_km"
        ])

    home = df.loc[df.rep_id == rep_id, ["rep_home_lat", "rep_home_lon"]].iloc[0].to_numpy(dtype=float)
    prev = home.copy()
    legs = []
    cumulative = 0.0

    for stop, idx in enumerate(order, start=1):

        curr = df.loc[idx, ["lat", "lon"]].to_numpy(dtype=float)
        leg_km = RouteOptimizer._gc_km(prev, curr)
        cumulative += leg_km
        legs.append({
            "rep_id": int(rep_id),
            "stop": int(stop),
            "row_index": int(idx),
            "client_id": int(df.loc[idx, "client_id"]),
            "leg_km": float(leg_km),
            "travel_min": float((leg_km / cfg.avg_speed_kmph) * 60.0),
            "route_km_without_return": float(cumulative)
        })
        prev = curr

    return_home_km = RouteOptimizer._gc_km(prev, home)
    route_total_km = cumulative + return_home_km
    legs_df = pd.DataFrame(legs)
    legs_df["return_home_km"] = float(return_home_km)
    legs_df["route_total_km"] = float(route_total_km)
    return legs_df


def _heuristic_route_for_rep(df: pd.DataFrame, rep_id: int, cfg: Config) -> Tuple[List[int], List[int]]:

    rep_df = df[df.rep_id == rep_id]

    if rep_df.empty:

        return [], []

    terr = rep_df["rep_territory"].iloc[0]
    candidates = rep_df[rep_df["territory_cluster"] == terr]

    if len(candidates) < cfg.daily_visit_quota:

        candidates = rep_df

    candidates = candidates.copy()
    home = candidates[["rep_home_lat", "rep_home_lon"]].iloc[0].to_numpy()
    candidates["dist_home"] = candidates[["lat", "lon"]].apply(
        lambda row: RouteOptimizer._gc_km(home, row.to_numpy(dtype=float)), axis=1
    )
    if "recency_days" not in candidates.columns:
        candidates["recency_days"] = (
            pd.Timestamp.today() - pd.to_datetime(candidates["last_purchase_date"])
        ).dt.days

    candidates["heuristic_priority"] = (
        0.55 * (1.0 - _normalise(candidates["dist_home"])) +
        0.30 * _normalise(candidates["recency_days"]) +
        0.15 * _normalise(candidates["touches_last_30d"])
    )
    pool = candidates.sort_values(
        ["heuristic_priority", "dist_home"], ascending=[False, True]
    ).head(cfg.daily_visit_quota)

    remaining = pool.index.tolist()
    route_order: List[int] = []
    current = home
    while remaining:
        nxt = min(
            remaining,
            key=lambda idx: RouteOptimizer._gc_km(current, pool.loc[idx, ["lat", "lon"]].to_numpy(dtype=float))
        )
        route_order.append(int(nxt))
        current = pool.loc[nxt, ["lat", "lon"]].to_numpy(dtype=float)
        remaining.remove(nxt)

    return route_order, [int(i) for i in pool.index.tolist()]


def _build_strategy_route_plan(
    df: pd.DataFrame,
    cfg: Config,
    router: RouteOptimizer,
    rep_id: int,
    strategy: str
) -> pd.DataFrame:

    if strategy == "algorithmic":
        order, _, _ = router.optimise_for_rep(df[df.rep_id == rep_id], return_decisions=False)
    elif strategy == "heuristic":
        order, _ = _heuristic_route_for_rep(df, rep_id, cfg)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    route_df = _build_route_legs(df, rep_id, order, cfg)

    if route_df.empty:

        return route_df

    feature_cols = [
        "client_id", "nba_score", "predicted_potential", "churn_prob", "csat_score",
        "relationship_strength", "budget_fit_score", "authority_access_score",
        "need_intensity_score", "timeline_urgency_score", "meeting_accept_rate",
        "competitor_pressure_index", "implementation_risk_score",
        "value_realization_score", "account_complexity_index",
        "strategic_fit_score", "recent_issue_flag"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    account_features = (
        df.loc[order, feature_cols]
          .reset_index()
          .rename(columns={"index": "row_index"})
    )
    route_df = route_df.merge(account_features, on=["row_index", "client_id"], how="left")
    route_df["strategy"] = strategy
    return route_df


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 2000,
    alpha: float = 0.05
) -> Tuple[float, float]:

    if values.size == 0:

        return np.nan, np.nan

    if values.size == 1:

        val = float(values[0])
        return val, val

    idx = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[idx].mean(axis=1)
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lower, upper


def _paired_permutation_pvalue(
    delta: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 5000
) -> float:

    if delta.size == 0:

        return np.nan

    obs = float(np.mean(delta))

    if delta.size == 1:

        return 1.0 if np.isclose(obs, 0.0) else 0.5

    signs = rng.choice([-1.0, 1.0], size=(iterations, delta.size), replace=True)
    perm_means = np.mean(delta * signs, axis=1)
    p = (np.sum(np.abs(perm_means) >= abs(obs)) + 1.0) / (iterations + 1.0)
    return float(p)


def _benjamini_hochberg(pvalues: List[float]) -> List[float]:

    arr = np.asarray(pvalues, dtype=float)
    n = arr.size

    if n == 0:

        return []

    order = np.argsort(arr)
    ranked = arr[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        adjusted[i] = val
        prev = val

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out.tolist()


def _resolve_openai_api_key(api_key: Optional[str] = None) -> Optional[str]:

    if api_key and api_key.strip():

        return api_key.strip()

    env_key = os.getenv("OPENAI_API_KEY")

    if env_key and env_key.strip():

        return env_key.strip()

    def _read_dotenv_key(path: Path, key: str) -> Optional[str]:
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() != key:
                continue
            value = v.strip().strip('"').strip("'")
            return value or None
        return None

    dotenv_candidates = [
        Path.cwd() / ".env",
        Path.home() / ".env",
        Path("/Users/macjcp/.env"),
    ]
    for dotenv_path in dotenv_candidates:
        key = _read_dotenv_key(dotenv_path, "OPENAI_API_KEY")
        if key:
            return key

    alt_env_key = os.getenv("OPENAI_API_KEY_MACJCP")

    if alt_env_key and alt_env_key.strip():

        return alt_env_key.strip()

    key_files = [
        Path.home() / ".openai_api_key",
        Path.home() / ".config" / "openai" / "api_key",
    ]
    for path in key_files:
        if path.exists():
            raw = path.read_text(encoding="utf-8").strip()
            if raw:
                return raw
    return None


def has_openai_api_key() -> bool:

    return bool(_resolve_openai_api_key())


def _safe_metric(value: Any, digits: int = 2) -> str:

    if value is None:

        return "n/a"

    if isinstance(value, (float, np.floating)):

        if np.isnan(value):
            return "n/a"
        return f"{float(value):.{digits}f}"

    if isinstance(value, (int, np.integer)):

        return str(int(value))

    return str(value)


def _to_float(value: Any) -> float:

    try:
        if value is None:
            return np.nan
        out = float(value)
        if np.isnan(out):
            return np.nan
        return out
    except Exception:
        return np.nan


def _industry_language_guidance(industry: str) -> str:

    guide = {
        "Retail": "Use language around basket size, footfall conversion, seasonality, and margin protection.",
        "Healthcare": "Use language around patient outcomes, care continuity, compliance, and service reliability.",
        "SaaS": "Use language around adoption, activation, ARR expansion, and time-to-value.",
        "Education": "Use language around enrollment retention, learner outcomes, and academic cycle timing.",
        "Manufacturing": "Use language around throughput, downtime reduction, quality yield, and OEE impact.",
        "Logistics": "Use language around SLA performance, on-time delivery, route reliability, and cost-to-serve.",
        "Energy": "Use language around uptime, operational risk, efficiency, and transition-readiness.",
    }
    return guide.get(industry, "Use commercial language aligned with the customer's operating KPIs and priorities.")


def _derive_engagement_mode(row: pd.Series) -> str:

    csat = _to_float(row.get("csat_score"))
    churn = _to_float(row.get("churn_prob"))
    potential = _to_float(row.get("predicted_potential"))
    potential_z = _to_float(row.get("predicted_potential_z"))
    recent_issue = int(_to_float(row.get("recent_issue_flag") or 0) or 0) == 1
    support_tickets = _to_float(row.get("support_tickets_90d"))

    dissatisfied = (
        (not np.isnan(csat) and csat <= 6.5) or
        recent_issue or
        (not np.isnan(support_tickets) and support_tickets >= 2.0)
    )
    high_churn = (not np.isnan(churn) and churn >= 0.45)
    low_churn = (not np.isnan(churn) and churn <= 0.28)
    high_potential = (
        (not np.isnan(potential_z) and potential_z >= 0.35) or
        (np.isnan(potential_z) and not np.isnan(potential) and potential >= 3500.0)
    )
    csat_strong = (not np.isnan(csat) and csat >= 7.4)

    if dissatisfied or high_churn:
        return "recovery_then_sales"
    if low_churn and high_potential and csat_strong:
        return "pure_sales_industry"
    return "consultative_growth"


def _build_aida_context_snapshot(
    row: pd.Series,
    rep_id: int,
    stop: int,
    strategy: str,
    engagement_mode: str,
    industry_guidance: str,
) -> str:

    context_lines = [
        f"Rep: {rep_id} | Route stop: {stop} | Strategy: {strategy}",
        f"Client ID: {_safe_metric(row.get('client_id'))}",
        f"Engagement mode: {engagement_mode}",
        f"Industry: {row.get('industry', 'n/a')} | Tier: {row.get('account_tier', 'n/a')}",
        f"Industry language guidance: {industry_guidance}",
        f"Company size: {_safe_metric(row.get('company_size'))} FTE",
        f"NBA score: {_safe_metric(row.get('nba_score'))} | Predicted potential: EUR {_safe_metric(row.get('predicted_potential'), 0)}",
        f"Churn probability: {_safe_metric(row.get('churn_prob'))} | CSAT: {_safe_metric(row.get('csat_score'))}",
        f"Need score: {_safe_metric(row.get('need_intensity_score'))} | Budget fit: {_safe_metric(row.get('budget_fit_score'))}",
        f"Authority access: {_safe_metric(row.get('authority_access_score'))} | Timeline urgency: {_safe_metric(row.get('timeline_urgency_score'))}",
        f"Relationship strength: {_safe_metric(row.get('relationship_strength'))} | Value realization: {_safe_metric(row.get('value_realization_score'))}",
        f"Competitor pressure: {_safe_metric(row.get('competitor_pressure_index'))} | Implementation risk: {_safe_metric(row.get('implementation_risk_score'))}",
        f"Product interest: {row.get('product_interest', 'n/a')} | Recommended channel: {row.get('recommended_channel', 'field')}",
        f"Recent issue flag: {_safe_metric(row.get('recent_issue_flag'))} | Touches 30d: {_safe_metric(row.get('touches_last_30d'))}",
    ]
    return "\n".join(context_lines)


def _build_aida_prompt(
    context_snapshot: str,
    engagement_mode: str,
    industry_guidance: str,
) -> str:

    mode_instruction = {
        "recovery_then_sales": (
            "- First add section heading exactly: Resolution Bridge (Before AIDA).\n"
            "- In this section, acknowledge issues, isolate root cause, and give a concrete near-term fix commitment.\n"
            "- Add one bullet that transitions naturally from issue resolution to business value discussion.\n"
            "- Then continue with AIDA sections."
        ),
        "pure_sales_industry": (
            "- No recovery section unless context explicitly indicates active unresolved complaint.\n"
            "- Focus on growth, value expansion, and concrete commercial next step.\n"
            "- Use strong industry-native phrasing."
        ),
        "consultative_growth": (
            "- Use consultative discovery style before proposing solution.\n"
            "- Balance risk management and growth narrative."
        ),
    }.get(engagement_mode, "- Adapt tone to context with consultative discipline.")

    return (
        "You are an enterprise field-sales enablement assistant.\n"
        "Write a concise, practical visit script grounded in context and intended to improve or leverage it.\n"
        "Do NOT reference internal metrics directly in the script.\n"
        "Constraints:\n"
        "- Face-to-face visit context only.\n"
        "- Translate signals into customer-facing language and actions.\n"
        "- Do NOT mention these terms in the final script: CSAT, churn, NBA, predicted potential, probabilities, z-score.\n"
        "- Do not state raw numeric model scores in the final script.\n"
        "- Use industry language guidance naturally: "
        f"{industry_guidance}\n"
        "- Include exactly these AIDA headings: Attention, Interest, Desire, Action.\n"
        "- Each AIDA section must contain 2-4 short bullet points.\n"
        "- Keep tone professional, consultative, and concrete.\n"
        "- Include one final line starting with 'Compliance:' mentioning consent/GDPR-safe practice.\n"
        "- Do not fabricate unavailable facts.\n"
        f"{mode_instruction}\n\n"
        "Account context:\n"
        f"{context_snapshot}\n"
    )


def _call_openai_chat_completion(
    prompt: str,
    model: str,
    api_key: str,
    timeout_sec: int = 60
) -> str:

    payload = {
        "model": model,
        "temperature": 0.35,
        "messages": [
            {
                "role": "system",
                "content": "You produce sales scripts with precise AIDA structure.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    req = urlrequest.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed: HTTP {exc.code} | {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    try:
        text = body["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenAI response format: {body}") from exc

    return str(text).strip()


def generate_pilot_aida_scripts(
    artifacts: PipelineArtifacts,
    pilot: PilotExperimentArtifacts,
    strategy: str = "algorithmic",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    max_visits_per_rep: Optional[int] = None,
    llm_callable: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:

    if strategy not in {"algorithmic", "heuristic"}:

        raise ValueError("strategy must be either 'algorithmic' or 'heuristic'")

    route_plan = pilot.route_plan.copy()
    route_plan = route_plan[
        route_plan["rep_id"].isin(pilot.selected_reps) &
        (route_plan["strategy"] == strategy)
    ].sort_values(["rep_id", "stop"])

    if max_visits_per_rep is not None:
        route_plan = (
            route_plan.groupby("rep_id", as_index=False, group_keys=False)
            .head(max(1, int(max_visits_per_rep)))
        )

    if route_plan.empty:
        return pd.DataFrame(columns=[
            "rep_id",
            "strategy",
            "stop",
            "row_index",
            "client_id",
            "industry",
            "engagement_mode",
            "nba_score",
            "predicted_potential",
            "churn_prob",
            "recommended_channel",
            "context_snapshot",
            "aida_script",
            "error",
        ])

    if llm_callable is None:
        resolved_key = _resolve_openai_api_key(api_key)
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in environment or .env "
                "(project .env or /Users/macjcp/.env), or pass api_key explicitly."
            )

        def llm_callable(prompt: str) -> str:
            return _call_openai_chat_completion(prompt=prompt, model=model, api_key=resolved_key)

    rows: List[Dict[str, Any]] = []
    df = artifacts.df
    for visit in route_plan.itertuples(index=False):
        row_index = int(visit.row_index)
        if row_index not in df.index:
            rows.append({
                "rep_id": int(visit.rep_id),
                "strategy": strategy,
                "stop": int(visit.stop),
                "row_index": row_index,
                "client_id": int(visit.client_id),
                "industry": "n/a",
                "engagement_mode": "n/a",
                "nba_score": np.nan,
                "predicted_potential": np.nan,
                "churn_prob": np.nan,
                "recommended_channel": "field",
                "context_snapshot": "",
                "aida_script": "",
                "error": f"row_index {row_index} not found in dataset",
            })
            continue

        row = df.loc[row_index]
        industry = str(row.get("industry", "n/a"))
        engagement_mode = _derive_engagement_mode(row)
        industry_guidance = _industry_language_guidance(industry)
        context_snapshot = _build_aida_context_snapshot(
            row=row,
            rep_id=int(visit.rep_id),
            stop=int(visit.stop),
            strategy=strategy,
            engagement_mode=engagement_mode,
            industry_guidance=industry_guidance,
        )
        prompt = _build_aida_prompt(
            context_snapshot=context_snapshot,
            engagement_mode=engagement_mode,
            industry_guidance=industry_guidance,
        )
        try:
            script = llm_callable(prompt)
            error_text = ""
        except Exception as exc:
            script = ""
            error_text = str(exc)

        rows.append({
            "rep_id": int(visit.rep_id),
            "strategy": strategy,
            "stop": int(visit.stop),
            "row_index": row_index,
            "client_id": int(row.get("client_id", visit.client_id)),
            "industry": industry,
            "engagement_mode": engagement_mode,
            "nba_score": float(row.get("nba_score", np.nan)),
            "predicted_potential": float(row.get("predicted_potential", np.nan)),
            "churn_prob": float(row.get("churn_prob", np.nan)),
            "recommended_channel": str(row.get("recommended_channel", "field")),
            "context_snapshot": context_snapshot,
            "aida_script": script,
            "error": error_text,
        })

    return pd.DataFrame(rows)


def run_field_pilot_experiment(
    artifacts: PipelineArtifacts,
    cfg: Optional[Config] = None,
    n_salespersons: Optional[int] = None,
    n_runs: Optional[int] = None,
    n_days: Optional[int] = None,
    random_state: Optional[int] = None
) -> PilotExperimentArtifacts:

    cfg = cfg or Config()
    df = artifacts.df.copy()
    n_salespersons = n_salespersons or cfg.pilot_salespersons
    n_runs = n_runs or cfg.monte_carlo_runs
    n_days = n_days or cfg.pilot_days
    seed = cfg.random_state if random_state is None else random_state
    rng = np.random.default_rng(seed)

    rep_rank = df.groupby("rep_id").size().sort_values(ascending=False)
    selected_reps = [int(r) for r in rep_rank.head(max(1, n_salespersons)).index.tolist()]
    if len(selected_reps) < 2:
        logging.warning(
            "Requested %d salespersons but only %d available in dataset.",
            n_salespersons, len(selected_reps)
        )

    plans = []
    for rep_id in selected_reps:
        for strategy in ("heuristic", "algorithmic"):
            p = _build_strategy_route_plan(df, cfg, artifacts.router, rep_id, strategy)
            if not p.empty:
                plans.append(p)
    route_plan = pd.concat(plans, ignore_index=True) if plans else pd.DataFrame()

    if route_plan.empty:
        return PilotExperimentArtifacts(
            selected_reps=selected_reps,
            route_plan=route_plan,
            unit_metrics=pd.DataFrame(),
            kpi_summary=pd.DataFrame(),
            stats_summary=pd.DataFrame()
        )

    grouped_plans = {
        key: grp.sort_values("stop").reset_index(drop=True)
        for key, grp in route_plan.groupby(["rep_id", "strategy"], sort=False)
    }

    unit_rows = []
    for run_id in range(1, n_runs + 1):
        for day in range(1, n_days + 1):
            market_factor = float(np.clip(rng.normal(1.0, 0.08), 0.75, 1.30))
            csat_context = float(rng.normal(0.0, 0.15))
            for rep_id in selected_reps:
                for strategy in ("heuristic", "algorithmic"):
                    plan = grouped_plans.get((rep_id, strategy))
                    if plan is None or plan.empty:
                        continue

                    revenue = 0.0
                    sales_wins = 0
                    service_hours = 0.0
                    csat_samples = []
                    visit_count = len(plan)
                    route_km = float(plan["route_total_km"].iloc[0])
                    for row in plan.itertuples(index=False):
                        stop_load = (row.stop - 1) / max(visit_count - 1, 1)
                        readiness = (
                            0.22 * row.budget_fit_score +
                            0.17 * row.authority_access_score +
                            0.20 * row.need_intensity_score +
                            0.15 * row.timeline_urgency_score +
                            0.14 * (row.relationship_strength / 100.0) +
                            0.12 * row.meeting_accept_rate
                        )
                        friction = (
                            0.30 * (row.competitor_pressure_index / 100.0) +
                            0.22 * (row.implementation_risk_score / 100.0) +
                            0.20 * (row.account_complexity_index / 100.0) +
                            0.16 * row.churn_prob +
                            0.12 * row.recent_issue_flag
                        )
                        strategy_bonus = 0.24 if strategy == "algorithmic" else -0.04
                        p_win = _sigmoid(
                            -1.15
                            + 2.35 * readiness
                            - 1.55 * friction
                            - 0.52 * stop_load
                            + 0.28 * (row.value_realization_score / 100.0)
                            + strategy_bonus
                            + rng.normal(0, 0.08)
                        )
                        converted = rng.random() < p_win
                        sales_wins += int(converted)
                        if converted:
                            deal_value = (
                                max(120.0, row.predicted_potential)
                                * market_factor
                                * (0.45 + 0.35 * row.strategic_fit_score + 0.20 * rng.random())
                            )
                            revenue += float(deal_value)

                        service_minutes = cfg.service_duration_min * (
                            0.85 + 0.22 * (row.account_complexity_index / 100.0) + 0.10 * rng.random()
                        )
                        travel_time_factor = 0.94 if strategy == "algorithmic" else 1.06
                        service_time_factor = 0.90 if strategy == "algorithmic" else 1.02
                        total_minutes = row.travel_min * travel_time_factor + service_minutes * service_time_factor
                        service_hours += total_minutes / 60.0

                        visit_csat = np.clip(
                            row.csat_score
                            + 0.85 * int(converted)
                            + (0.35 if strategy == "algorithmic" else -0.05)
                            - 0.60 * stop_load
                            - 0.32 * row.recent_issue_flag
                            + csat_context
                            + rng.normal(0, 0.50),
                            0.0,
                            10.0
                        )
                        csat_samples.append(float(visit_csat))

                    sales_per_hour = revenue / max(service_hours, 1e-9)
                    unit_rows.append({
                        "run_id": int(run_id),
                        "day": int(day),
                        "rep_id": int(rep_id),
                        "strategy": strategy,
                        "visits": int(visit_count),
                        "route_km": route_km,
                        "sales_eur": float(revenue),
                        "sales_wins": int(sales_wins),
                        "win_rate": float(sales_wins / max(visit_count, 1)),
                        "service_hours": float(service_hours),
                        "sales_per_hour_eur": float(sales_per_hour),
                        "avg_post_visit_csat": float(np.mean(csat_samples) if csat_samples else np.nan),
                    })

    unit_metrics = pd.DataFrame(unit_rows)
    metric_map = {
        "sales": "sales_eur",
        "efficiency": "sales_per_hour_eur",
        "customer_satisfaction": "avg_post_visit_csat"
    }

    summary_rows = []
    stats_rows = []
    stats_rng = np.random.default_rng(seed + 101)
    p_for_fdr = []
    for kpi_name, kpi_col in metric_map.items():
        paired = (
            unit_metrics.pivot_table(
                index=["run_id", "day", "rep_id"],
                columns="strategy",
                values=kpi_col,
                aggfunc="mean"
            )
            .dropna()
        )
        if paired.empty:
            continue
        before = paired["heuristic"].to_numpy(dtype=float)
        after = paired["algorithmic"].to_numpy(dtype=float)
        delta = after - before
        delta_mean = float(delta.mean())
        before_mean = float(before.mean())
        after_mean = float(after.mean())
        delta_pct = float(100.0 * delta_mean / (abs(before_mean) + 1e-9))
        ci_low, ci_high = _bootstrap_mean_ci(
            delta, stats_rng, iterations=cfg.bootstrap_iterations, alpha=0.05
        )
        ttest = stats.ttest_rel(after, before, alternative="two-sided")
        permutation_pvalue = _paired_permutation_pvalue(
            delta, stats_rng, iterations=cfg.permutation_iterations
        )
        non_zero = int(np.sum(delta != 0))
        if non_zero > 0:
            sign = stats.binomtest(
                int(np.sum(delta > 0)),
                n=non_zero,
                p=0.5,
                alternative="greater"
            )
            sign_pvalue = float(sign.pvalue)
        else:
            sign_pvalue = 1.0
        try:
            wilcoxon = stats.wilcoxon(after, before, alternative="two-sided", zero_method="wilcox")
            wilcoxon_pvalue = float(wilcoxon.pvalue)
        except Exception:
            wilcoxon_pvalue = np.nan
        effect_size = float(delta_mean / (delta.std(ddof=1) + 1e-9))
        win_probability = float(np.mean(delta > 0))
        summary_rows.append({
            "kpi": kpi_name,
            "heuristic_mean": before_mean,
            "algorithmic_mean": after_mean,
            "delta_abs": delta_mean,
            "delta_pct": delta_pct
        })
        stats_rows.append({
            "kpi": kpi_name,
            "paired_units": int(len(delta)),
            "delta_ci_low": float(ci_low),
            "delta_ci_high": float(ci_high),
            "ttest_pvalue": float(ttest.pvalue),
            "permutation_pvalue": permutation_pvalue,
            "wilcoxon_pvalue": wilcoxon_pvalue,
            "sign_test_pvalue": sign_pvalue,
            "effect_size_cohens_d": effect_size,
            "win_probability": win_probability,
            "significant_95": bool((ci_low > 0) and (permutation_pvalue < 0.05))
        })
        p_for_fdr.append(float(permutation_pvalue) if np.isfinite(permutation_pvalue) else 1.0)

    kpi_summary = pd.DataFrame(summary_rows)
    stats_summary = pd.DataFrame(stats_rows)
    if not stats_summary.empty:
        stats_summary["fdr_qvalue"] = _benjamini_hochberg(p_for_fdr)
        stats_summary["significant_fdr_95"] = (
            (stats_summary["delta_ci_low"] > 0) &
            (stats_summary["fdr_qvalue"] < 0.05)
        )
    return PilotExperimentArtifacts(
        selected_reps=selected_reps,
        route_plan=route_plan,
        unit_metrics=unit_metrics,
        kpi_summary=kpi_summary,
        stats_summary=stats_summary
    )

# metrics.py
# Takes true and forecasted values and computes metrics

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from math import sqrt
from typing import Optional, Dict, Tuple, List


# =========================================================
# CONFIG
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()

# Cities & models you want included in the summary tables:
CITY_LIST   = ["boulder", "palo_alto", "dundee", "perth"]
MODEL_LIST  = ["arima", "xgboost", "gru", "lstm", "transformer"]

# min=10min native; hour=1h native; day=1d native
BLOCKS = {
    "min":  [1, 2, 3],          # 10, 20, 30 minutes
    "hour": [2, 4, 6, 8],       # 2, 4, 6, 8 hours
    "day":  [1, 2, 3, 4, 5],    # 1–5 days
}

# Precision for metrics
PRINT_DECIMALS = 2


# =========================================================
# IO HELPERS
# =========================================================
def _try_read_df(path: Path) -> Optional[pd.DataFrame]:
    """
    Read a DataFrame and ensure a DatetimeIndex (or derive from 'Timestamp').
    ".parquet" or ".pkl".
    """
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            df = pickle.load(f)
    else:
        return None
    if not isinstance(df, pd.DataFrame):
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["Timestamp"])).drop(columns=["Timestamp"], errors="ignore")
        else:
            return None
    return df.sort_index()


def load_preds(project_dir: Path, city: str, model: str, gran: str) -> Optional[pd.DataFrame]:
    """
    Load predictions for (city, model, granularity)
    """
    base = project_dir / "results" / f"results_{city}" / f"{city}_{model}"
    p = base / f"{model}_{gran}.pkl"
    df = _try_read_df(p)
    if df is not None:
        print(f"[OK] {city}/{model}/{gran} <- {p.name}")
        return df
    print(f"[MISS] {city}/{model}/{gran} (no file {p.name})")
    return None


# =========================================================
# ID INFERENCE
# =========================================================
def ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'station_id' and 'region_id' columns exist.
    If missing, infer each from the argmax over their one-hot groups.
    """
    out = df.copy()
    if "station_id" not in out.columns:
        st_cols = [c for c in out.columns if c.startswith("station_")]
        out["station_id"] = out[st_cols].idxmax(axis=1) if st_cols else "station_UNKNOWN"
    if "region_id" not in out.columns:
        rg_cols = [c for c in out.columns if c.startswith("region_")]
        out["region_id"] = out[rg_cols].idxmax(axis=1) if rg_cols else "region_UNKNOWN"
    return out


# =========================================================
# METRICS & AGGREGATION
# =========================================================
def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (MAE, RMSE) for numpy arrays."""
    err = y_pred - y_true
    return float(np.mean(np.abs(err))), float(sqrt(np.mean(err**2)))


def _block_reduce_mean(ts_df: pd.DataFrame, block: int) -> pd.DataFrame:
    """
    Given a 2-col DataFrame with y_true_scaled, y_pred_scaled indexed by time,
    truncate to full non-overlapping blocks of length `block` and return block means.
    """
    n = len(ts_df)
    if n < block:
        return pd.DataFrame(columns=["y_true_scaled", "y_pred_scaled"])
    full = (n // block) * block
    ts_df = ts_df.iloc[:full]
    blk = np.arange(len(ts_df)) // block
    return ts_df.assign(_blk=blk).groupby("_blk")[["y_true_scaled", "y_pred_scaled"]].mean()


def group_block_scores(df: pd.DataFrame, id_col: str, block: int):
    """
    For each group id (station or region):
      1) collapse duplicate timestamps within the id via SUM,
      2) block means,
      3) MAE/RMSE on block means.
    Returns (mean_MAE across ids, mean_RMSE across ids).
    """
    rows = []
    for gid, g in df.groupby(id_col, sort=False):
        ts_sum = g.groupby(g.index)[["y_true_scaled", "y_pred_scaled"]].sum()
        agg = _block_reduce_mean(ts_sum, block)
        if agg.empty:
            continue
        mae, rmse = mae_rmse(agg["y_true_scaled"].values, agg["y_pred_scaled"].values)
        rows.append({id_col: gid, "MAE": mae, "RMSE": rmse})
    res = pd.DataFrame(rows)
    mean_mae = float(res["MAE"].mean()) if not res.empty else np.nan
    mean_rmse = float(res["RMSE"].mean()) if not res.empty else np.nan
    return mean_mae, mean_rmse


def city_block_scores(df: pd.DataFrame, block: int):
    """
    City-wide totals:
      - SUM across stations per timestamp,
      - block means,
      - MAE/RMSE on block means.
    """
    ts_sum = df.groupby(df.index)[["y_true_scaled", "y_pred_scaled"]].sum()
    agg = _block_reduce_mean(ts_sum, block)
    if agg.empty:
        return np.nan, np.nan
    return mae_rmse(agg["y_true_scaled"].values, agg["y_pred_scaled"].values)


def evaluate_one(df: pd.DataFrame, block: int):
    """
    Compute evaluation metrics for a single DataFrame and block size.
    Returns a dict with Station/Region mean MAE/RMSE and City MAE/RMSE.
    """
    need = {"y_true_scaled", "y_pred_scaled"}
    if not need.issubset(df.columns):
        raise ValueError("Preds DF must have y_true_scaled and y_pred_scaled.")
    df2 = ensure_ids(df)

    sta_mae = sta_rmse = reg_mae = reg_rmse = np.nan
    if (df2["station_id"] != "station_UNKNOWN").any():
        sta_mae, sta_rmse = group_block_scores(df2, "station_id", block)
    if (df2["region_id"] != "region_UNKNOWN").any():
        reg_mae, reg_rmse = group_block_scores(df2, "region_id", block)
    city_mae, city_rmse = city_block_scores(df2, block)

    return dict(
        StationMAE=sta_mae, StationRMSE=sta_rmse,
        RegionMAE=reg_mae, RegionRMSE=reg_rmse,
        CityMAE=city_mae, CityRMSE=city_rmse
    )


# =========================================================
# SMALL HELPERS (labels, table shaping)
# =========================================================
def pretty_model(m: str) -> str:
    """Map internal model keys to pretty names for tables."""
    return {"arima":"ARIMA","xgboost":"XGBoost","gru":"GRU","lstm":"LSTM","transformer":"Transformer"}.get(m.lower(), m)


def build_table_city(
    store: Dict[Tuple[str, str, int, str], dict],
    city: str,
    gran_key: str,
    step: int,
    models: List[str]
) -> pd.DataFrame:
    """
    Assemble a metrics table for a given (city, granularity, block length) across models.
    Each cell is that city's metrics.
    """
    rows = ["StationMAE","StationRMSE","RegionMAE","RegionRMSE","CityMAE","CityRMSE"]
    data = {}
    for m in models:
        d = store.get((city, gran_key, step, m), None)
        if d is None:
            data[pretty_model(m)] = [np.nan] * len(rows)
        else:
            data[pretty_model(m)] = [float(d.get(r, np.nan)) for r in rows]
    return pd.DataFrame(data, index=rows)


# =========================================================
# MAIN
# =========================================================
def _window_label(gran: str, step: int) -> str:
    """Human-friendly window label for console headers."""
    if gran == "min":  return f"{step*10} minutes"
    if gran == "hour": return f"{step} hours"
    if gran == "day":  return f"{step} days"
    return f"{gran}:{step}"


def main():
    """
    1) Load per-(city, model, gran) predictions
    2) Compute metrics on block MEANS for each requested window.
    3) Print pretty tables.
    4) Save combined CSVs (min/hour/day) PER CITY (so 12 files total).
    """
    # store[(city, gran, step, model)] -> metrics dict for that city/model/gran/window
    store: Dict[Tuple[str, str, int, str], dict] = {}

    # ---- Evaluate per city ----
    for city in CITY_LIST:
        for model in MODEL_LIST:
            for gran, steps in BLOCKS.items():
                df = load_preds(PROJECT_DIR, city, model, gran)
                if df is None:
                    continue
                for block in steps:
                    try:
                        metrics = evaluate_one(df, block)
                        store[(city, gran, block, model)] = metrics
                    except Exception as e:
                        print(f"[SKIP] {city}/{model}/{gran} step={block}: {e}")

    fmt = lambda x: "NaN" if pd.isna(x) else f"{x:.{PRINT_DECIMALS}f}"

    # Pretty print per-window tables (PER CITY)
    for city in CITY_LIST:
        for gran, steps in BLOCKS.items():
            for step in steps:
                table = build_table_city(store, city, gran, step, MODEL_LIST)
                print(f"\n=== {city.upper()} — {gran.upper()} metrics — block MEAN over {_window_label(gran, step)} ===")
                print(table.to_string(float_format=fmt))

    # Save combined CSVs PER CITY
    outdir = PROJECT_DIR / "results" / "summaries"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = ["StationMAE","StationRMSE","RegionMAE","RegionRMSE","CityMAE","CityRMSE"]

    for city in CITY_LIST:
        for gran in ["min", "hour", "day"]:
            pieces = []
            col_tuples = []
            for m in MODEL_LIST:
                for step in BLOCKS[gran]:
                    t = build_table_city(store, city, gran, step, [m])  # single-column table
                    pieces.append(t)
                    label = f"{step*10}m" if gran == "min" else (f"{step}h" if gran == "hour" else f"{step}d")
                    col_tuples.append((pretty_model(m), label))
            if pieces:
                wide = pd.concat(pieces, axis=1)
                wide.columns = pd.MultiIndex.from_tuples(col_tuples, names=["Model", "Window"])
                wide = wide.loc[rows]
                wide.to_csv(outdir / f"metrics_{gran}_combined_{city}.csv")


if __name__ == "__main__":
    main()
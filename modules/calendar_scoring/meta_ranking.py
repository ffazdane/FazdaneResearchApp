import os
import pickle
import sqlite3
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from utils.persistence import get_db_path

logger = logging.getLogger("CalendarMetaRanking")

# Try to import XGBoost and standard Scikit-Learn tools
try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn is not available. Meta-ranking model will use heuristic proxy.")

MODEL_FILE = Path(os.path.dirname(get_db_path("calendar_scoring"))) / "meta_ranking_model.pkl"

# Default feature columns for the ML model
FEATURE_COLS = [
    "final_score", "trend_score", "option_structure_score", "volatility_score",
    "pca_score", "cluster_score", "leading_lagging_score", "liquidity_score",
    "event_risk_score", "institutional_flow_score", "price_at_decision",
    "atr_14", "rsi_14", "adx_14", "ema_20", "ema_50", "ema_200", "iv_rank",
    "iv_percentile", "front_iv", "back_iv", "iv_term_structure", "avg_option_volume",
    "avg_open_interest", "bid_ask_spread_pct", "setup_delta", "setup_gamma",
    "setup_theta", "setup_vega", "net_debit"
]

CATEGORICAL_COLS = ["cluster_label", "leading_lagging_state", "market_regime", "fdts_signal"]

def train_meta_ranking_model() -> dict:
    """
    Query the SQLite database for past decisions and outcomes,
    train a regressor to predict forward return (pnl_pct), and serialize it.
    """
    if not _SKLEARN_AVAILABLE:
        return {"status": "error", "message": "scikit-learn is not installed. Run: pip install scikit-learn"}
        
    db_path = get_db_path("calendar_scoring")
    if not db_path.exists():
        return {"status": "cold_start", "message": "Database file does not exist yet."}
        
    try:
        conn = sqlite3.connect(db_path)
        
        # Join decision logs, trade setups, and outcome logs
        query = """
            SELECT d.*, s.setup_delta, s.setup_gamma, s.setup_theta, s.setup_vega, s.net_debit, s.max_risk,
                   o.pnl_pct
            FROM ticker_decision_log d
            JOIN option_trade_setup_log s ON d.decision_id = s.decision_id
            JOIN decision_outcome_log o ON d.decision_id = o.decision_id
            WHERE o.pnl_pct IS NOT NULL AND o.review_day = 20
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty or len(df) < 10:
            return {
                "status": "cold_start",
                "message": f"Insufficient historical records with outcomes (Found {len(df)} records, minimum of 10 required for training)."
            }
            
        # Prepare Target & Features
        y = df["pnl_pct"].values
        
        # Categorical processing (Label encoding mapping)
        cat_mappings = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)
                unique_vals = sorted(df[col].unique())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df[col + "_encoded"] = df[col].map(mapping)
                cat_mappings[col] = mapping
            else:
                df[col + "_encoded"] = 0
                cat_mappings[col] = {}
                
        # Numeric processing (Fill NaNs with column median)
        X_df = pd.DataFrame(index=df.index)
        for col in FEATURE_COLS:
            if col in df.columns:
                X_df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if not pd.isna(df[col].median()) else 0.0)
            else:
                X_df[col] = 0.0
                
        # Append encoded categorical features
        for col in CATEGORICAL_COLS:
            X_df[col + "_encoded"] = df[col + "_encoded"]
            
        X = X_df.values
        feature_names = list(X_df.columns)
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Regressor
        if _XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.08, random_state=42)
            model.fit(X_scaled, y)
            model_type = "XGBoost"
            importances = model.feature_importances_
        else:
            model = GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.07, random_state=42)
            model.fit(X_scaled, y)
            model_type = "Gradient Boosting"
            importances = model.feature_importances_
            
        # Evaluate model on training data (R2 & RMSE)
        y_pred = model.predict(X_scaled)
        r2 = float(r2_score(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        
        # Build feature importance output
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
        # Sort importances
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        # Save Model artifact
        model_payload = {
            "model": model,
            "scaler": scaler,
            "cat_mappings": cat_mappings,
            "feature_names": feature_names,
            "model_type": model_type,
            "r2": r2,
            "rmse": rmse,
            "sample_count": len(df),
            "feature_importances": sorted_importance
        }
        
        # Ensure directories exist
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model_payload, f)
            
        logger.info(f"Meta-ranking model ({model_type}) trained and saved to {MODEL_FILE}")
        return {
            "status": "success",
            "model_type": model_type,
            "r2": r2,
            "rmse": rmse,
            "sample_count": len(df),
            "feature_importances": sorted_importance
        }
        
    except Exception as e:
        logger.error(f"Error training meta-ranking model: {e}")
        return {"status": "error", "message": f"Training failed: {str(e)}"}

def predict_meta_rankings(candidates: list) -> list:
    """
    Take a list of scored candidate dictionaries, predict their expected returns
    using the trained ML model, and inject it as a key 'ml_predicted_return'.
    """
    # Load model if it exists
    model_payload = None
    if MODEL_FILE.exists():
        try:
            with open(MODEL_FILE, "rb") as f:
                model_payload = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load serialized meta-ranking model: {e}")
            
    # If no model found or sklearn missing, fall back to heuristic score ranking
    if model_payload is None or not _SKLEARN_AVAILABLE:
        for c in candidates:
            # Cold-start formula: expected 20D return is roughly scaled to final score
            # Score of 50 = 0% return, Score of 85 = +14% return, Score of 95 = +18% return
            score = c.get("final_score", 50.0)
            c["ml_predicted_return"] = round(float((score - 50.0) * 0.4), 2)
            c["ml_model_used"] = "Cold-Start Heuristic"
        return candidates

    try:
        model = model_payload["model"]
        scaler = model_payload["scaler"]
        cat_mappings = model_payload["cat_mappings"]
        feature_names = model_payload["feature_names"]
        model_type = model_payload["model_type"]
        
        # Prepare inference vectors
        rows = []
        for c in candidates:
            setup = c.get("option_setup", {})
            row_dict = {}
            
            # Map numeric features
            for col in FEATURE_COLS:
                if col in c:
                    row_dict[col] = c[col]
                elif col in setup:
                    row_dict[col] = setup[col]
                elif col == "price_at_decision":
                    row_dict[col] = c.get("spot_price", 0.0)
                else:
                    row_dict[col] = 0.0
                    
            # Map categorical features
            for col in CATEGORICAL_COLS:
                val = str(c.get(col, ""))
                mapping = cat_mappings.get(col, {})
                # Encode with mapping or 0 if unknown category
                row_dict[col + "_encoded"] = mapping.get(val, 0)
                
            # Align columns with model feature names
            ordered_row = [row_dict.get(fname, 0.0) for fname in feature_names]
            rows.append(ordered_row)
            
        # Inference
        X = np.array(rows)
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        for c, pred in zip(candidates, predictions):
            c["ml_predicted_return"] = round(float(pred), 2)
            c["ml_model_used"] = model_type
            
    except Exception as e:
        logger.error(f"Error executing meta-ranking inference: {e}. Using cold-start fallback.")
        for c in candidates:
            score = c.get("final_score", 50.0)
            c["ml_predicted_return"] = round(float((score - 50.0) * 0.4), 2)
            c["ml_model_used"] = "Fallback Heuristic"
            
    return candidates

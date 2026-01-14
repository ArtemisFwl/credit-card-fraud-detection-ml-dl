"""
NOTE:
- main.py = orchestration only
- OLD (non-pipeline) approaches are kept as COMMENTS for learning
- NEW (pipeline) approaches are ACTIVE
"""

from src.utils.logger import logger
from src.utils.exceptions import FraudException

# =============================================================================
# STEP 1: Data Loading
# =============================================================================
from src.data.data_loader import DataLoader

if __name__ == "__main__":
    try:
        logger.info("Pipeline Started")

        loader = DataLoader("data/creditcard.csv")
        df = loader.load_data()

        logger.info("Dataloader setup completed successfully")

    except Exception as e:
        logger.error("Pipeline Failed")
        raise FraudException("Main Pipeline Failed", e)

# =============================================================================
# STEP 2: Preprocessing
# =============================================================================
from src.preprocessing.preprocessor import Preprocessor

preprocessor = Preprocessor()

X, y = preprocessor.split_features_target(df)
X_train, X_test, y_train, y_test = preprocessor.train_test_split_data(X, y)

logger.info("Preprocessing step completed successfully")

# =============================================================================
# STEP 7.1: Feature Sanity Checks
# =============================================================================
logger.info("STEP–7.1: Feature sanity check started")

numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

logger.info(f"Numerical features count: {len(numerical_features)}")
logger.info(f"Categorical features count: {len(categorical_features)}")

if X_train.isnull().sum().sum() == 0:
    logger.info("No null values found")

logger.info("STEP–7.1 completed")

# =============================================================================
# STEP 3: Logistic Regression
# =============================================================================
from src.models.logistic_regression import LogisticRegressionTrainer

trainer = LogisticRegressionTrainer()

# -----------------------------------------------------------------
# OLD APPROACH (NO PIPELINE) — LEARNING ONLY
# -----------------------------------------------------------------
# model = trainer.model.fit(X_train, y_train)
# report, auc = trainer.evaluate(X_test, y_test)

# -----------------------------------------------------------------
# NEW APPROACH (PIPELINE) — ACTIVE
# -----------------------------------------------------------------
trainer.train(X_train, y_train)
report, auc = trainer.evaluate(X_test, y_test)

logger.info("Baseline Logistic Regression Completed")

# =============================================================================
# STEP 4: Random Forest
# =============================================================================
from src.models.random_forest import RandomForestTrainer

rf_model = RandomForestTrainer()
rf_model.train(X_train, y_train)
rf_report, rf_auc = rf_model.evaluate(X_test, y_test)

logger.info("Baseline Random Forest Completed")

# =============================================================================
# LEARNING ONLY: Random Forest Threshold Tuning
# =============================================================================
import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report

y_proba_rf = rf_model.model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_rf)

target_recall = 0.90
best_threshold = 0.5

for t, r in zip(thresholds, recall[1:]):
    if r >= target_recall:
        best_threshold = t
        break

logger.info(f"[RF] Chosen Threshold (recall>=0.90): {best_threshold}")
logger.info(classification_report(y_test, (y_proba_rf >= best_threshold).astype(int)))

# =============================================================================
# LEARNING ONLY: Logistic Regression Threshold Tuning
# =============================================================================
y_proba_lr = trainer.pipeline.predict_proba(X_test)[:, 1]
precision_lr, recall_lr, thresholds_lr = precision_recall_curve(y_test, y_proba_lr)

best_lr_threshold = 0.5
for t, r in zip(thresholds_lr, recall_lr[1:]):
    if r >= 0.95:
        best_lr_threshold = t
        break

logger.info(f"[LR] Chosen Threshold (recall>=0.95): {best_lr_threshold}")
logger.info(classification_report(y_test, (y_proba_lr >= best_lr_threshold).astype(int)))

# =============================================================================
# STEP 5: XGBoost
# =============================================================================
from src.models.xgboost_model import XGBoostModel

try:
    xgb_model = XGBoostModel()

    # OLD APPROACH (NO PIPELINE) — LEARNING ONLY
    # xgb_model.model.fit(X_train, y_train)

    # NEW APPROACH (PIPELINE) — ACTIVE
    xgb_model.train(X_train, y_train)
    xgb_report, xgb_auc = xgb_model.evaluate(X_test, y_test)

except Exception as e:
    raise FraudException("XGBoost Pipeline Failed", e)

# =============================================================================
# MODEL COMPARISON
# =============================================================================
logger.info("===== MODEL COMPARISON (ROC-AUC) =====")
logger.info(f"Logistic Regression ROC-AUC: {auc}")
logger.info(f"Random Forest ROC-AUC: {rf_auc}")
logger.info(f"XGBoost ROC-AUC: {xgb_auc}")

# =============================================================================
# LEARNING ONLY: XGBoost Threshold Tuning
# =============================================================================
from sklearn.metrics import recall_score

y_proba_xgb = xgb_model.pipeline.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 50)

best_xgb_threshold = 0.5
for t in thresholds:
    if recall_score(y_test, (y_proba_xgb >= t).astype(int)) >= 0.90:
        best_xgb_threshold = t
        break

logger.info(f"[XGB] Chosen Threshold (recall>=0.90): {best_xgb_threshold}")
logger.info(classification_report(y_test, (y_proba_xgb >= best_xgb_threshold).astype(int)))


#Save the model
from src.utils.model_saver import save_model

# Assume XGBoost best hai
save_model(xgb_model.pipeline, "artifacts/best_model.pkl")

import joblib

model = joblib.load("artifacts/best_model.pkl")
_ = model.predict(X_test[:5])
logger.info("Inference test passed")

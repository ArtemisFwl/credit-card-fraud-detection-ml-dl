from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import get_preprocessing_pipeline
from src.utils.logger import logger
from src.utils.exceptions import FraudException


class XGBoostModel:
    def __init__(self, random_state: int = 42):
        """
        REMARK:
        - GENERAL MODEL INIT
        - Gradient boosting model for imbalanced fraud data
        """
        try:
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=random_state,
                n_jobs=-1
            )
            self.pipeline = None

        except Exception as e:
            raise FraudException("XGBoost init failed", e)

    def train(self, X_train, y_train):
        """
        REMARK:
        - OLD APPROACH (commented): direct model.fit
        - NEW APPROACH (active): Pipeline to avoid data leakage
        """
        try:
            logger.info("Training XGBoost Model")

            # ===============================
            # OLD APPROACH (NO PIPELINE)
            # Learning reference only
            # ===============================
            # self.model.fit(X_train, y_train)

            # ===============================
            # NEW APPROACH (PIPELINE)
            # ===============================
            self.pipeline = Pipeline(
                steps=[
                    ("preprocess", get_preprocessing_pipeline()),
                    ("model", self.model)
                ]
            )

            self.pipeline.fit(X_train, y_train)

            logger.info("XGBoost training completed")

        except Exception as e:
            raise FraudException("XGBoost training failed", e)

    def evaluate(self, X_test, y_test):
        """
        REMARK:
        - Evaluation using trained pipeline
        """
        try:
            logger.info("Evaluating XGBoost model")

            # ===============================
            # OLD APPROACH (NO PIPELINE)
            # ===============================
            # y_pred = self.model.predict(X_test)
            # y_proba = self.model.predict_proba(X_test)[:, 1]

            # ===============================
            # NEW APPROACH (PIPELINE)
            # ===============================
            y_pred = self.pipeline.predict(X_test)
            y_proba = self.pipeline.predict_proba(X_test)[:, 1]

            report = classification_report(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            logger.info(f"XGBoost Classification Report:\n{report}")
            logger.info(f"XGBoost ROC-AUC Score: {auc}")

            return report, auc

        except Exception as e:
            raise FraudException("XGBoost evaluation failed", e)

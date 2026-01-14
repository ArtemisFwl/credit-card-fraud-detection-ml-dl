import joblib
from src.utils.logger import logger

def save_model(model, path):
    joblib.dump(model, path)
    logger.info(f"Model saved at {path}")

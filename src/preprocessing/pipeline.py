from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import RobustScaler

def get_preprocessing_pipeline():
  return Pipeline (
    steps=[("scaler", RobustScaler())]
  )
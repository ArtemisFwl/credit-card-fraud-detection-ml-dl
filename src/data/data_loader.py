import pandas as pd

from src.utils.logger import logger
from src.utils.exceptions import FraudException

class DataLoader: 

  def __init__ (self, file_path:str):
    self.file_path=file_path


  def load_data(self):
    try:
      logger.info(f"Loading dataset from {self.file_path}")

      df=pd.read_csv(self.file_path)

      logger.info(
        f"Dataset loaded successfully with shape {df.shape}"
      )
      return df
    
    except Exception as e:
      logger.error("Failed to load dataset")

      raise FraudException ("Data Loading Failed, e")
    

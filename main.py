from src.data.data_loader import DataLoader

from src.utils.logger import logger
from src.utils.exceptions import FraudException

if __name__== "__main__":
  try:
    logger.info("Pipeline Started")

    loader=DataLoader("data/creditcard.csv")

    df=loader.load_data()
    logger.info("Dataloader Setup completed successfully")

  except Exception as e:
    logger.error("Pipeline Failed")
    raise FraudException("Main Pipeline Failed")
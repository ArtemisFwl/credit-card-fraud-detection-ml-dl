print("Project setup successful")

from src.utils.logger import logger
from src.utils.exceptions import FraudException

if __name__== "__main__":
  try:
    logger.info("Project Execution Started")

    x=1/0

  except Exception as e:
    logger.error("Error occured in main execution")

    raise FraudException ("Main Pipeline Failed",e)

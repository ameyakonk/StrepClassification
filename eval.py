from src.pipeline.data_ingestion import DataIngestion
from src.pipeline.create_dataloader import CreateDataloader
from src.pipeline.evaluate_model import ModelEvaluation

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

STAGE_NAME = "Data Ingestion"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestion()
    data_ingestion.data_ingestion()
    train_idx, test_dx, val_idx = data_ingestion.train_test_split()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Prepare Data Loader"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    create_dataloader = CreateDataloader(train_idx, test_dx, val_idx)
    test_loader = create_dataloader.create_test_dataloader(16)
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Evaluate Model"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    modelevaluation = ModelEvaluation(test_loader)
    modelevaluation.model_evaluate()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logging.exception(e)
    raise e
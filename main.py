from src.pipeline.data_ingestion import DataIngestion
from src.pipeline.create_dataloader import CreateDataloader
from src.pipeline.train_model import ModelTraining
from src.pipeline.evaluate_model import ModelEvaluation

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

import argparse
parser = argparse.ArgumentParser(description="Script parameters")

# 2. Add arguments
parser.add_argument("--dataset", type=str, default="cnh", help="Choose the dataset")

# 3. Parse the arguments
args = parser.parse_args()

STAGE_NAME = "Data Ingestion"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestion(args.dataset)
    data_ingestion.data_ingestion()
    train_idx, test_dx, val_idx = data_ingestion.train_test_split()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Prepare Data Loader"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    create_dataloader = CreateDataloader(train_idx, test_dx, val_idx, args.dataset)
    train_loader, test_loader, val_loader = create_dataloader.create_dataloader()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Train Model"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    modeltraining = ModelTraining(train_loader, val_loader)
    modeltraining.model_train()
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
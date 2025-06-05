from Mushroom_edibility_classifier import logger
from Mushroom_edibility_classifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from Mushroom_edibility_classifier.pipeline.stage_02_preapre_base_model import PrepareBaseModelPipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"
try:
    print(f">>>>> stage {STAGE_NAME} started <<<<<")
    prepare_base_model_pipeline = PrepareBaseModelPipeline()
    prepare_base_model_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"An error occurred: {e}")
    raise e

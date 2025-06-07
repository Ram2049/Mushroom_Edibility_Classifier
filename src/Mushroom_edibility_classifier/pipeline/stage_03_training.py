from Mushroom_edibility_classifier.config.configuration import ConfigurationManager
from Mushroom_edibility_classifier.components.Prepare_callback import PrepareCallback 
from Mushroom_edibility_classifier.components.Training import Training
from Mushroom_edibility_classifier import logger

STAGE_NAME = "Training"

class Model_training_pipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.PrepareCallbackConfig()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.create_data_generators()
        training.train(
            callback_list=callback_list
        )
    

if __name__ == "__main__":
    try:
        print(f">>>>> stage {STAGE_NAME} started <<<<<")
        Trainig = Model_training_pipeline()
        Trainig.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise e
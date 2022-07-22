from re import M
from venv import create
from src.utils.all_utils import read_yaml, create_directory
from src.utils.callbacks_utils import get_callbacks
from src.utils.model_utils import load_full_model, get_unique_path_to_save_model
from src.utils.data_management_utils import train_and_valid_generator
import argparse
import os
import logging as lg






def train_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    ## getting the directories from the config file

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    callbacks_dir = artifacts["CALLBACKS_DIR"]
    
    untrained_full_model_path = os.path.join(artifacts_dir,artifacts["BASE_MODEL_DIR"],
                                artifacts["UPDATED_BASE_MODEL_NAME"])
    
    callback_dir_path = os.path.join(artifacts_dir, callbacks_dir)

    model = load_full_model(untrained_full_model_path)
    
    callbacks = get_callbacks(callback_dir_path)

    train_generator , valid_generator = train_and_valid_generator(data_dir= artifacts["DATA_DIR"],
                                                                    image_size = params["IMAGE_SIZE"][:-1],
                                                                    do_data_augmentation= params["AUGMENTATION"],
                                                                    batchSize = params["BATCH_SIZE"])

    step_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    model.fit(train_generator,
                validation_data = valid_generator,
                epochs = params["EPOCHS"],
                steps_per_epoch = step_per_epoch,
                validation_steps = validation_steps,
                callbacks = callbacks )

    trained_model_dir = os.path.join(artifacts_dir,artifacts["TRAINED_MODEL_DIR"])
    create_directory([trained_model_dir])


    trained_model_path = get_unique_path_to_save_model(trained_model_dir)
    model.save(trained_model_path)

    lg.info(f"trained model saved at path {trained_model_path}")

if __name__ == "__main__":

    
    logging_format = "[%(asctime)s -- %(module)s -- %(levelname)s] ===> %(message)s"
    log_dir = "logs"
    create_directory([log_dir])
    lg.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level= lg.INFO, format= logging_format, filemode="a")
    
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parse_args = args.parse_args()
    try:
        lg.info("stage 4 --->  started")
        train_model(parse_args.config,parse_args.params)
        lg.info("stage 4     --->  completed(training completed and model is saved)")
    except Exception as e:
        lg.exception(e)
        raise e
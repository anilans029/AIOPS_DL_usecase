from venv import create
from src.utils.all_utils import read_yaml, create_directory
from src.utils.model_utils import get_VGG16_model,prepare_full_model
import pandas as pd
import argparse
import os
import shutil
from tqdm import tqdm
import logging as lg
import io






def prepare_base_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    ## getting the directories from the config file

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    trained_mdoel_dir = artifacts["TRAINED_MODEL_DIR"]
    base_model_dir = artifacts["BASE_MODEL_DIR"]
    base_model_name = artifacts["BASE_MODEL_NAME"]

    base_model_dir_path = os.path.join(artifacts_dir, base_model_dir)

    create_directory([base_model_dir_path])

    base_model_path = os.path.join(base_model_dir_path, base_model_name) 

    base_model = get_VGG16_model(inputShape =params["IMAGE_SIZE"], model_path=base_model_path)

    full_model = prepare_full_model(
        base_model,
        CLASSES = params["CLASSES"],
        freeze_all = True,
        freeze_till = None,
        learning_rate = params["LEARNING_RATE"]
    )

    updated_base_model_name = artifacts["UPDATED_BASE_MODEL_NAME"]
    updated_base_model_path = os.path.join(base_model_dir_path, updated_base_model_name)
    full_model.save(updated_base_model_path)

    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn = lambda x : stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
            return summary_str
        
    lg.info(f"full model summary: \n {_log_model_summary(full_model)}")
  

  

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
        lg.info("stage 1 --->  (loading data from source to local) started")
        prepare_base_model(parse_args.config,parse_args.params)
        lg.info("stage 1 --->  (loading data from source to local) completed")
    except Exception as e:
        lg.exception(e)
        raise e
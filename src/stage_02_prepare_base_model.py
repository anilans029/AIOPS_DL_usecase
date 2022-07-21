from venv import create
from src.utils.all_utils import read_yaml, create_directory
import pandas as pd
import argparse
import os
import shutil
from tqdm import tqdm
import logging as lg






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

    model = get_VGG16(input_shape =params["INPUT_SHAPE"], batch_size = params["BATCH_SIZE"], epochs = params["EPOCHS"])

   

  

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
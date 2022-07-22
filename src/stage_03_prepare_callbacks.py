from venv import create
from src.utils.all_utils import read_yaml, create_directory
import pandas as pd
import argparse
import os
import shutil
from tqdm import tqdm
import logging as lg
import io






def prepare_callbacks(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    ## getting the directories from the config file

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
 
  

  

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
        lg.info("stage 3 --->  (preparing callbacks) started")
        prepare_callbacks(parse_args.config,parse_args.params)
        lg.info("stage 3 --->  (preparing callbacks) completed")
    except Exception as e:
        lg.exception(e)
        raise e
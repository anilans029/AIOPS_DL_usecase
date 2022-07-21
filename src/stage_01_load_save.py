from venv import create
from src.utils.all_utils import read_yaml, create_directory
import pandas as pd
import argparse
import os
import shutil
from tqdm import tqdm
import logging as lg




def copy_files(source_data_dir, local_data_dir):
    list_files = os.listdir(source_data_dir)
    no_of_files= len(list_files)
    for file in tqdm(list_files, total=no_of_files, desc="copying files from source to local_data_folder", colour='red'):
        src_file = os.path.join(source_data_dir, file)
        dest_file = os.path.join(local_data_dir,file)
        shutil.copy(src=src_file,dst=dest_file)


def get_data(config_path):
    config = read_yaml(config_path)

    ## getting the directories from the config file

    source_data_dir = config["source_data_dir"]
    local_data_dir = config["local_data_dir"]


    for source_data_dir, local_data_dir in tqdm(zip(source_data_dir,local_data_dir),total=2, desc="list of folders", colour="green"):
        create_directory([local_data_dir])
        copy_files(source_data_dir, local_data_dir)


if __name__ == "__main__":

    
    logging_format = "[%(asctime)s -- %(module)s -- %(levelname)s] ===> %(message)s"
    log_dir = "logs"
    create_directory([log_dir])
    lg.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level= lg.INFO, format= logging_format, filemode="a")
    
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parse_args = args.parse_args()
    try:
        lg.info("stage 1 --->  (loading data from source to local) started")
        get_data(parse_args.config)
        lg.info("stage 1 --->  (loading data from source to local) completed")
    except Exception as e:
        lg.exception(e)
        raise e
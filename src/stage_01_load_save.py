from src.utils.all_utils import read_yaml, create_directory
import pandas as pd
import argparse
import os

def get_config(config_path):
    config = read_yaml(config_path)
    remote_data_path =config["data_source"]
    df = pd.read_csv(remote_data_path, sep=";")

### we need to save this data as csv file in the path artifacts/raw_local_dir/data.csv

    artifact_dir = config["artifacts"]["artifacts_dir"]
    raw_local_dir = config["artifacts"]["raw_local_dir"]
    raw_local_fie = config["artifacts"]["raw_local_fie"]

    raw_local_dir_path = os.path.join(artifact_dir,raw_local_dir)
    raw_local_fie_path = os.path.join(raw_local_dir_path,raw_local_fie)
    print(raw_local_fie_path)
    create_directory(dirs = [raw_local_dir_path])
    df.to_csv(raw_local_fie_path, sep = ";", index = False)



if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    parse_args = args.parse_args()
    get_config(parse_args.config)
import yaml
import os
import json
import logging as lg

def read_yaml(path_to_yaml: str)-> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    
    lg.info(f"loaded the yaml file from {path_to_yaml} and returning")
    return content

def create_directory(dirs: list):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        lg.info(f"directory created at {dir}")

def save_data_local(data, data_path, index_status = False):
    data.to_csv(data_path, index= index_status)
    lg.info(f"saved the dataframe {data} to csv in local at {data_path}")

def save_report(report: dict, report_path: str, indentation= 4):
    with open(report_path, "w") as f:
        json.dump(report,f,indent = indentation)
    lg.info(f"report created at {report_path}")
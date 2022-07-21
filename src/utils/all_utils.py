import yaml
import os
import json

def read_yaml(path_to_yaml: str)-> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)

    return content

def create_directory(dirs: list):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"directory created at {dir}")

def save_data_local(data, data_path, index_status = False):
    data.to_csv(data_path, index= index_status)

def save_report(report: dict, report_path: str, indentation= 4):
    with open(report_path, "w") as f:
        json.dump(report,f,indent = indentation)
    
    print(f"report created at {report_path}")
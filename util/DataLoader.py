import csv
import json
import re

from TextProcess.textProcess import TextPreprocessor
from util import Dao
import sys

sys.path.append("../../")


def load_config_from_json(filepath: str):
    config = {}
    try:
        config = json.load(open(filepath, 'r'))
    except IOError:
        print('ConfigFile ' + filepath + ' Not Found')
    return config


def promise_dataloader(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        data = [row for row in reader]
    f.close()
    return data


def nfr_so_dataloader(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()
        content = f.readlines()
        data = [row.split('!#!') for row in content]
    f.close()
    return data


def get_project_code(project: str):
    project_code_dict = {"ARIES": "ARIES", "ZOOKEEPER": "ZOOKEEPER", "HADOOP": "HADOOP", "MAVEN": "MNG"}
    code = ""
    if project in project_code_dict.keys():
        code = project_code_dict[project]
    if len(code) == 0:
        print("Warning: Unrecognized project")
    return code


def summary_component_dataloader(project: str, component_range=1000, do_label_augmentation=False, augment_type=""):
    sql_config = load_config_from_json("config/MYSQLConfig.json")
    text_processor = TextPreprocessor()
    project = project.upper()
    project_code = get_project_code(project)
    raw_data, data = Dao.get_issue_component(sql_config, sql_config["database"], project_code), []
    for _ in raw_data:
        summary, component = _["summary"], _["component"]
        if summary is None or component is None:
            continue
        text = summary.replace('\n', ' ')
        data.append({"issue_key": _["issue_key"], "text": text_processor.text_process(text).capitalize(),
                     "create_time": _["create_time"], "issue_type": _["issue_type"].capitalize(),
                     "creator": _["creator"], "component": re.sub(r"[^\w]+", "-", _["component"]).upper(), "priority": _["priority"],
                     "status": _["status"], "resolution": _["resolution"], "assignee": _["assignee"],
                     "reporter": _["reporter"]})
    component_cnt = Dao.get_component_count(sql_config, sql_config["database"], project_code, component_range)
    for _ in component_cnt:
        _["component"] = re.sub(r"[^\w]+", "-", _["component"]).upper()
    if do_label_augmentation:
        for _ in data:
            _["component"] = label_augmentation(project, augment_type, _["component"])
        for _ in component_cnt:
            _["component"] = label_augmentation(project, augment_type, _["component"])
    data, component_id_dict = component_cnt_filter(data, component_cnt)
    data.sort(key=lambda x: x["issue_key"])
    return data, component_id_dict


def component_cnt_filter(data: list, component_cnt: list):
    component_id_dict = {}
    for i, key in enumerate(component_cnt):
        component_id_dict[key["component"]] = i
    data = [_ for _ in data if _["component"] in component_id_dict.keys()]
    return data, component_id_dict


def component_first_occurrence(project: str, component_range=1000):
    sql_config = load_config_from_json("config/MYSQLConfig.json")
    project = project.upper()
    project_code = get_project_code(project)
    raw_data = Dao.get_component_by_time(sql_config, sql_config["database"], project_code)
    return raw_data[: component_range]


def label_augmentation(project: str, augment_type, label_str: str):
    augmented_label = label_str
    if augment_type == "prefix":
        augmented_label = project + "-" + augmented_label
    elif augment_type == "detail":
        label_detail = load_config_from_json(f"config/label_mapping/{project}.json")
        if label_str in label_detail.keys():
            augmented_label = label_detail[label_str]
        else:
            print(label_str)
            print("Warning: Unrecognized label")
    return augmented_label

import json
from os.path import exists
from typing import Dict, Any

import pandas as pd


def read_json(path: str):
    assert isinstance(path, str) and exists(path)
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def read_google_spreadsheet(google_id: str, sheet_name: str):
    assert isinstance(google_id, str)
    assert isinstance(sheet_name, str)
    url = f"https://docs.google.com/spreadsheets/d/{google_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url, on_bad_lines='skip')
    return df



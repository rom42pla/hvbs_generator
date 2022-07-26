import threading
import time
from os.path import exists
from typing import Dict

from utils import read_google_spreadsheet, read_json


class Database:
    def __init__(self, endpoints_path: str,
                 refresh_interval: float = 2):
        assert refresh_interval > 0
        self.refresh_interval = refresh_interval

        assert exists(endpoints_path)
        self.endpoints_path = endpoints_path
        self.data = {}
        self.refresh()

        def refresh_thread_fn():
            while True:
                time.sleep(self.refresh_interval)
                self.refresh()

        self.refresh_thread = threading.Thread(target=refresh_thread_fn)
        self.refresh_thread.start()

    def refresh(self):
        def parse_data(data: Dict):
            assert isinstance(data, dict)
            for k, v in data.items():
                if isinstance(v, dict):
                    # identifies a google spreadsheet
                    if set(v.keys()) == {"google_id", "sheets"}:
                        google_id, sheets = v["google_id"], v["sheets"]
                        data[k] = {
                            sheet: read_google_spreadsheet(google_id=google_id, sheet_name=sheet).to_dict("records")
                            for sheet in sheets
                        }
                    else:
                        data[k] = parse_data(data[v])
            return data

        self.data = parse_data(read_json(self.endpoints_path))

    def get_data(self):
        return self.data

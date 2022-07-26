from os import makedirs
from os.path import isdir, join
from typing import Optional, Dict, Any

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only

import pandas as pd


class HvbLogger(LightningLoggerBase):
    def __init__(
            self,
            path: Optional[str] = None
    ):
        super().__init__()
        assert path is None or isinstance(path, str)
        self.path = path
        if self.path is not None:
            if not isdir(self.path):
                makedirs(self.path)
        self.logs_df: pd.DataFrame = pd.DataFrame()
        self.hparams: Dict[str, Any] = {}

    @property
    def name(self):
        return "HvbGeneratorLogger"

    @property
    def logs(self):
        return self.logs_df

    @property
    def version(self):
        return "0.1"

    @property
    def experiment(self) -> Any:
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        self.hparams = vars(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.logs_df = pd.concat([self.logs_df, pd.DataFrame([metrics])],
                                 ignore_index=True)

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        assert not self.logs_df.empty, f"dsds"
        if self.path is not None:
            # saves the logs
            self.logs_df.to_csv(join(self.path, "logs.csv"))



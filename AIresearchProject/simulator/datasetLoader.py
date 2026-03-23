from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ast
from pandas.core import col


class datasetLoader:
    def __init__(self, filePath: str, workloadData: Optional[str] = None):
        self.filePath = Path(filePath)

        if not self.filePath.exists():
            raise FileNotFoundError(f"Dataset not found: {self.filePath}")

        self.data = pd.read_csv(self.filePath)
        self.index = 0

        if self.data.empty:
            raise ValueError("No Data")

        self.numericData = self.data.select_dtypes(include=[np.number])

        if self.numericData.empty:
            raise ValueError("No numeric data found")

        #print("In full dataframe columns?", "average_usage" in self.data.columns)
        #print("In numeric dataframe columns?", "average_usage" in self.numericData.columns)
        #print(self.data.dtypes)

        self.workloadCol = self.resWorkload(workloadData)

        self.data[self.workloadCol] = self.data[self.workloadCol].apply(lambda x: ast.literal_eval(x)['cpus'] if isinstance(x, str) else 0.0)

        self.workloadSig = self.workloadSignal()

    def resWorkload(self, workloadData: Optional[str] = None):
        if workloadData is not None:
            if workloadData not in self.data.columns:
                raise ValueError(f"'{workloadData}'not found. Available columns: {list(self.data.columns)}")
            return workloadData

        colNames = {"cpu_usage", "cpu", "cpu_utilization", "mean_cpu_usage_rate", "resource_request", "resource_usage", "load", "usage", "average_usage"}

        lowerMap = {col.lower(): col for col in self.data.columns}

        for name in colNames:
            if name in lowerMap:
                return lowerMap[name]

        return self.data.columns[0]

    def workloadSignal(self) -> np.ndarray:

        series = self.data[self.workloadCol].astype(float).fillna(0.0).to_numpy()

        minValue = np.min(series)
        maxValue = np.max(series)

        if maxValue == minValue:
            return np.zeros_like(series, dtype=np.float32)

        normalize = (series - minValue) / (maxValue - minValue)

        return normalize.astype(np.float32)

    def nextLoad(self) -> float:

        value = float(self.workloadSig[self.index])
        self.index = (self.index + 1) % len(self.workloadSig)
        return value

    def reset(self) -> None:
        self.index = 0

    def preview(self, rows: int = 5) -> pd.DataFrame:
        return self.data.head(rows)

    def numericCols(self) -> list[str]:
        return list(self.numericData.columns)

    def workloadCols(self) -> str:
        return self.workloadCol


    #-----------testing below, remove after----------------

    #from simulator.datasetLoader import datasetLoader

#loader = datasetLoader("../data/borg_traces_data.csv")

##print("Chosen workload column:", loader.workloadCols())
#print("Numeric columns:", loader.numericCols()[:10])
#print(loader.preview())

#for _ in range(10):
#    print(loader.nextLoad())

#print(loader.data["average_usage"].min())
#print(loader.data["average_usage"].max())





















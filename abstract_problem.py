from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class AbstractProblem(ABC):
    
    data = None

    def __init__(self, dataset_path):
        self.prepare_dataset(dataset_path)

    def create_x0(self, data):
        return pd.DataFrame({'x0': np.ones(len(data.index)) * - 1})

    @abstractmethod
    def prepare_dataset(self, path):
        pass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import MinMaxScaler
from tsl.data.datamodule import TemporalSplitter, SpatioTemporalDataModule

class ElectricDataset:
    def __init__(self, batch_size=64, periods=24):
        
        self.periods = periods
        self.batch_size = batch_size

        self.dataset = pd.read_csv('./data/electric/normal.csv')

        self.nodes = self.dataset.shape[1]

        data_range = pd.date_range(start=self.start_date, end=self.end_date, freq='H')

        connectivity = self.get_connectivity(
            threshold=0.1,
            include_self=False,
            normalize_axis=1,
            )

        covariates = {'u': data_range.dayofyear.values}

        torch_dataset = SpatioTemporalDataset(
            target=self.dataset,
            connectivity = connectivity,
            mask=np.ones_like(self.dataset).astype(bool),
            covariates=covariates,
            horizon=24,
            window=self.periods,
            stride=24
            )

        scalers = {'target': MinMaxScaler(axis=(0,1))}
       
        splitter = TemporalSplitter(val_len=0.1, test_len=0.2)

        self.dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=splitter,
            batch_size=self.batch_size,
            )
        
        self.dm.setup()

    def get_dm(self):
        return self.dm
    
    def get_connectivity(self, threshold, include_self=False, normalize_axis=1):
        n_nodes = self.dataset.shape[1]
        correlation = self.dataset.corr()


        edge_weights = []
        edge_index = np.array([
            np.repeat(np.arange(n_nodes), n_nodes),
            np.tile(np.arange(n_nodes), n_nodes)
        ])

        for i in range(edge_index.shape[1]):
            e1 = edge_index[0, i]
            e2 = edge_index[1, i]
            value = correlation.iloc[e1, e2]
            value = 0 if threshold > value else value
            edge_weights.append(value)
        edge_weights = np.array(edge_weights).astype(np.float32)

        if not include_self:
            edge_index = edge_index[:, edge_weights != 1]
            edge_weights = edge_weights[edge_weights != 1]

        scaler = MinMaxScaler(axis=normalize_axis)
        edge_weights = scaler.fit_transform(edge_weights.reshape(-1, 1)).reshape(-1)

        return edge_index, edge_weights
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_nodes(self):
        return self.nodes
    
    def get_periods(self):
        return self.periods

class NormalPeriod(ElectricDataset):
    def __init__(self, **kwargs):
        self.seed = 9101112
        self.dataset= 'electric_normal'
        self.start_date = '2019-09-15 00:00:00'
        self.end_date = '2019-12-15 23:00:00'
        super().__init__(**kwargs)
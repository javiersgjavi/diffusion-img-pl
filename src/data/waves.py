import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import MinMaxScaler
from tsl.data.datamodule import TemporalSplitter, SpatioTemporalDataModule

class WavesDataset:
    def __init__(self, batch_size=256, periods=24, nodes=6, lenght=10e4):
        
        self.periods = periods
        self.batch_size = batch_size

        self.nodes = nodes
        self.length = int(lenght)

        self.dataset = self.populate_dataset()

        connectivity = self.get_connectivity()

        torch_dataset = SpatioTemporalDataset(
            target=self.dataset,
            connectivity = connectivity,
            mask=np.ones_like(self.dataset).astype(bool),
            horizon=24,
            window=self.periods,
            stride=1
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
    
    def populate_dataset(self):

        periods = np.linspace(0,10,self.nodes//2)
        periods = np.sort(np.concatenate([periods, periods]))

        values_x = np.linspace(1, 100, self.length)

        data = np.zeros((self.nodes, self.length))

        for i in range(0, self.nodes, 2):
            w = 2*np.pi*periods[i]

            data[i,:] = np.sin(w*values_x)
            data[i,:] = np.cos(w*values_x)

        return data.transpose()

        
    
    def get_connectivity(self):

        edge_weights = []
        edge_index = np.array([np.arange(self.nodes),np.arange(self.nodes)])
        edge_weights = np.array([1 for _ in range(self.nodes)])

        return edge_index, edge_weights
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_nodes(self):
        return self.nodes
    
    def get_periods(self):
        return self.periods

class Waves(WavesDataset):
    def __init__(self, **kwargs):
        self.seed = 9101112
        self.dataset= 'waves'
        super().__init__(**kwargs)
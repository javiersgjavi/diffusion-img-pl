import sys
sys.path.append('./')

from torch.optim import Adam, AdamW
from src.data.electric import NormalPeriod
from src.data.waves import Waves
from src.models.engine import DiffusionEngine
from src.models.unet import UNet
from src.models.diffusion_model import DiffusionModel

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer


def main():

    #dataset = NormalPeriod()
    #dm = dataset.get_dm()

    dataset = Waves()
    dm = dataset.get_dm()


    imputer = DiffusionEngine(
        model_class= UNet,
        model_kwargs=None,
        optim_class=AdamW,
        optim_kwargs=dict({'lr': 1e-3}),
        scheduler_class=None,
        scheduler_kwargs=None,
        scaler=dm.scalers['target'],
        batch_size=dataset.get_batch_size(),
    )

    logger = TensorBoardLogger(
        save_dir='./logs',
    )

    callbaks = [
        EarlyStopping(
            monitor='val_loss',
            patience=200,
            verbose=True,
            mode='min'
        ),
        ModelCheckpoint(
            monitor='val_loss',
            filename='base_experiment',
            save_top_k=1,
            mode='min'
        )
    ]

    trainer = Trainer(
        max_epochs=100,
        default_root_dir='./logs',
        logger=logger,
        accelerator='gpu',
        devices=1,
        callbacks=callbaks,
        )

    trainer.fit(imputer, dm)

    trainer.test(imputer, datamodule=dm)

if __name__=='__main__':
    main()
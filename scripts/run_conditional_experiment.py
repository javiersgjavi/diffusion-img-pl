import sys
sys.path.append('./')

from src.data.landscape import LandscapeDataModule
from src.models.diffusion import DiffusionModel

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer


def main():

    #dataset = NormalPeriod()
    #dm = dataset.get_dm()

    dm = LandscapeDataModule()


    model = DiffusionModel()

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
        max_epochs=300,
        default_root_dir='./logs',
        logger=logger,
        accelerator='gpu',
        devices=[2],
        callbacks=callbaks,
        )

    trainer.fit(model, dm)

    trainer.test(model, datamodule=dm)

if __name__=='__main__':
    main()
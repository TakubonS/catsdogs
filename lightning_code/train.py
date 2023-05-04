import torch
import pytorch_lightning as pl
from dataset import MyDataModule
from model import Net
import config
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


if __name__ == '__main__':
    model = Net(
        config.LEARNING_RATE,
        config.LOSS_FN
    )
    data_module = MyDataModule(
        config.TRAIN_DATA_DIR,
        config.PRED_DATA_DIR,
        config.BATCH_SIZE,
        config.NUM_WORKERS, 
        config.TRANSFORM
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.DEVICES,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            LearningRateMonitor(logging_interval='step'),
        ],
        # overfit_batches=1,
        # fast_dev_run=True,
        # auto_lr_find=True[],
    )
    # trainer.tune(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module,
                # ckpt_path="/home/son/_study_group/lightning15_catsdogs_structured/lightning_logs/version_7/checkpoints/epoch=999-step=1000.ckpt"
            )
    # trainer.validate(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
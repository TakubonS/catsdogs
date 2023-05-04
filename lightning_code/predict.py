import pytorch_lightning as pl
from dataset import MyDataModule
from model import Net
import config
import itertools
import pandas as pd
import numpy as np  

if __name__ == '__main__':
    data_module = MyDataModule(
        config.TRAIN_DATA_DIR,
        config.PRED_DATA_DIR,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
        config.TRANSFORM
    )
    trainer = pl.Trainer(
        precision=config.PRECISION,
        accelerator="gpu",
        devices=1, #single GPU for prediction
    )

    # locate the xxx.ckpt file
    ckpt_path = "/home/son/_study_group/kaggle/dogs_n_cats/lightning_logs/version_24/checkpoints/epoch=4-step=3125.ckpt"

    model = Net.load_from_checkpoint(ckpt_path)
    predictions = trainer.predict(model, datamodule=data_module)
    for i in range(len(predictions)):
        predictions[i] = predictions[i].tolist()
    predictions_flatten = list(itertools.chain.from_iterable(predictions))
    # predictions_clipped = np.clip(predictions_flatten,  0.01, 0.99)

    csv_path = "/home/son/_study_group/kaggle/dogs_n_cats/catsdogs_dataset_kaggle/data/sample_submission.csv"
    df = pd.read_csv(csv_path)
    df["label"] = predictions_flatten
    df.to_csv("/home/son/_study_group/kaggle/dogs_n_cats/catsdogs_dataset_kaggle/data/submission.csv", index=False)


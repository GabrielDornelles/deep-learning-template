from models.resnet import ResNet18
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
import pytorch_lightning as pl


if __name__ == "__main__":

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',  # Metric to monitor for saving the best model
        filename='model_weights_{epoch:02d}-{val_acc:.2f}',  # Filename pattern
        save_top_k=1,  # Save only the best model
        mode='max',  # Mode of the monitored metric, minimize or maximize (min,max)
    )

    model = ResNet18()
    
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=10, 
        callbacks=[RichProgressBar(), checkpoint_callback],
        precision=16
    )
    trainer.fit(model)

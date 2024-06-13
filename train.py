import os
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from LitResnet import LitResnet
from torchvision.transforms import v2
import wandb
import hydra
import omegaconf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    seed_everything(cfg.seed)
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_logger = WandbLogger(project=cfg.project_name)
    exp_name = wandb_logger.experiment.name
    wandb.define_metric("val_acc", summary="max")

    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

    if cfg.use_da:
        train_transforms = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(10),
                v2.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                v2.ToTensor(),
                cifar10_normalization(),
            ]
        )
    else:
        train_transforms = v2.Compose(
            [
                v2.ToTensor(),
                cifar10_normalization(),
            ]
        )

    test_transforms = v2.Compose(
        [
            v2.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=PATH_DATASETS,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    model = LitResnet(
        batch_size=cfg.batch_size,
        label_smoothing=cfg.model.label_smoothing,
        optimizer=cfg.optimizer,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"results/{exp_name}",
        save_top_k=1,
        save_last=True,
        filename="best",
        every_n_epochs=1,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=cfg.trainer.num_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=10),
        ],
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
    )

    trainer.fit(model, cifar10_dm)


if __name__ == "__main__":
    main()

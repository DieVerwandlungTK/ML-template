import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)

    datamodule: L.LightningDataModule = instantiate(cfg.data)
    model: L.LightningModule = instantiate(cfg.model)
    callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]
    logger = instantiate(cfg.logger)

    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

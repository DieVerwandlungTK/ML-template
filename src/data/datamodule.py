import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR10DataModule(L.LightningDataModule):
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 128,
        num_workers: int = 4,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        resize = [transforms.Resize(image_size)] if image_size != 32 else []
        self.train_transform = transforms.Compose(
            resize
            + [
                transforms.RandomCrop(image_size, padding=image_size // 8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )
        self.val_transform = transforms.Compose(
            resize
            + [
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = datasets.CIFAR10(
            self.hparams.data_dir, train=True, download=True, transform=self.train_transform
        )
        self.val_dataset = datasets.CIFAR10(
            self.hparams.data_dir, train=False, download=True, transform=self.val_transform
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

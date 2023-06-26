"""
Adapted from Joshua Dimasaka, Andrew McDonald, Meghan Plumridge, Jay Torry, and
Andrés Camilo Zúñiga González's https://github.com/ai4er-cdt/sea-ice-classification/blob/main/train.py
and from Tom Andersson's https://github.com/tom-andersson/icenet-paper/blob/main/icenet/train_icenet.py
Modified the former from segmentation to forecasting.
Modified the latter from Tensorflow to PyTorch and PyTorch Lightning
"""
import os
import sys
print(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))  # if using cmd
import pandas as pd
import lightning.pytorch as pl
import wandb
from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.ops.focal_loss import sigmoid_focal_loss
from src.utils import IceNetDataset, Visualise
from src.models import UNet, LitUNet, Generator, Discriminator, LitGAN
from src import config

# trade off speed and performance depending on gpu
torch.set_float32_matmul_precision("medium")
# torch.set_float32_matmul_precision("high")


def train_icenet(args):
    """
    Train IceNet using the arguments specified in the `args` namespace.
    :param args: Namespace of configuration parameters
    """
    # init
    pl.seed_everything(args.seed)
    
    # configure datasets and dataloaders
    dataloader_config_fpath = os.path.join(config.dataloader_config_folder, args.dataloader_config)
    train_dataset = IceNetDataset(dataloader_config_fpath, mode="train")
    val_dataset = IceNetDataset(dataloader_config_fpath, mode="val")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                                 persistent_workers=True, shuffle=False)  # IceNetDataset class handles shuffling
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                                persistent_workers=True, shuffle=False)  # IceNetDataset class handles shuffling

    # configure model
    if args.model == "unet":

        # construct unet
        model = UNet(input_channels=train_dataset.tot_num_channels,
                     filter_size=args.filter_size,
                     n_filters_factor=args.n_filters_factor,
                     n_forecast_months=train_dataset.config["n_forecast_months"])
        
        # configure unet loss with reduction="none" for sample weighting
        if args.criterion == "ce":
            criterion = nn.CrossEntropyLoss(reduction="none")
        elif args.criterion == "focal":
            criterion = sigmoid_focal_loss  # reduction="none" by default
        else:
            raise NotImplementedError(f"Invalid UNet loss function: {args.criterion}.")
        
        # configure PyTorch Lightning module
        lit_module = LitUNet(
            model=model,
            criterion=criterion,
            learning_rate=args.learning_rate
        )

    elif args.model == "gan":

        # construct generator
        generator = Generator(input_channels=train_dataset.tot_num_channels,
                              filter_size=args.filter_size,
                              n_filters_factor=args.n_filters_factor,
                              n_forecast_months=train_dataset.config["n_forecast_months"],
                              sigma=args.sigma)

        # construct discriminator
        discriminator = Discriminator(input_channels=train_dataset.tot_num_channels,
                                      filter_size=args.filter_size,
                                      n_filters_factor=args.n_filters_factor,
                                      n_forecast_months=train_dataset.config["n_forecast_months"])
        
        # configure losses with reduction="none" for sample weighting
        if args.generator_fake_criterion == "ce":
            generator_fake_criterion = nn.CrossEntropyLoss(reduction="none")
        else:
            raise ValueError(f"Invalid generator fake loss function: {args.criterion}.")

        if args.generator_structural_criterion == "l1":
            generator_structural_criterion = nn.L1Loss(reduction="none")
        else:
            raise ValueError(f"Invalid generator structural loss function: {args.criterion}.")

        if args.discriminator_criterion == "ce":
            discriminator_criterion = nn.CrossEntropyLoss(reduction="none")
        else:
            raise ValueError(f"Invalid discriminator loss function: {args.criterion}.")
        
        # configure PyTorch Lightning module
        lit_module = LitGAN(
            generator=generator,
            discriminator=discriminator,
            generator_fake_criterion=generator_fake_criterion,
            generator_structural_criterion=generator_structural_criterion,
            generator_lambda=args.generator_lambda,
            discriminator_criterion=discriminator_criterion,
            learning_rate=args.learning_rate,
            d_lr_factor=args.d_lr_factor
        )

    else:
        raise NotImplementedError(f"Invalid model architecture specified: {args.model}")

    # set up wandb logging
    wandb.init(project="icenet-gan")
    if args.name != "default":
        wandb.run.name = args.name
    wandb_logger = pl.loggers.WandbLogger(project="icenet-gan")
    wandb_logger.experiment.config.update(args)

    # comment/uncomment the following line to track gradients
    # note that wandb cannot parallelise across multiple gpus when tracking gradients
    # wandb_logger.watch(model, log="all", log_freq=10)

    # set up trainer configuration
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run
    )
    trainer.logger = wandb_logger
    trainer.callbacks.append(ModelCheckpoint(monitor="val_accuracy"))
    trainer.callbacks.append(Visualise(val_dataloader, 
                                       n_to_visualise=args.n_to_visualise, 
                                       n_forecast_months=val_dataset.config["n_forecast_months"]))

    # train model
    print(f"Training {len(train_dataset)} examples / {len(train_dataloader)} batches (batch size {args.batch_size}).")
    print(f"Validating {len(val_dataset)} examples / {len(val_dataloader)} batches (batch size {args.batch_size}).")
    print(f"All arguments: {args}")
    trainer.fit(lit_module, train_dataloader, val_dataloader)


if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser(description="Train IceNet")
    parser.add_argument("--name", default="default", type=str, help="Name of wandb run")
    parser.add_argument("--model", default="unet", type=str, choices=["unet", "gan"],
                        help="Choice of model architecture", required=False)
    
    # model configurations applicable to UNet
    parser.add_argument("--criterion", default="focal", type=str, choices=["ce", "focal"],
                        help="Loss to train UNet", required=False)
    
    # model configurations applicable to GAN
    parser.add_argument("--sigma", default=1, type=float,
                        help="Noise factor to set sampling temperature in G", required=False)    
    parser.add_argument("--generator_fake_criterion", default="ce", type=str, choices=["ce"],
                        help="Loss to train GAN generator instance-level fake-out", required=False)
    parser.add_argument("--generator_structural_criterion", default="l1", type=str, choices=["l1"],
                        help="Loss to train GAN generator structural similarity", required=False)    
    parser.add_argument("--generator_lambda", default=100, type=float,
                        help="Trade off between instance-level fake-out and structural loss", required=False)    
    parser.add_argument("--discriminator_criterion", default="ce", type=str, choices=["ce"],
                        help="Loss to train GAN discriminator", required=False)
    parser.add_argument("--d_lr_factor", default=5, type=float,
                        help="Factor by which to multiply G learning rate for use on D", required=False)    
    
    # model configurations applicable to both UNet and GAN
    parser.add_argument("--dataloader_config", default="2023_06_24_1235_icenet_gan.json", type=str,
                        help="Filename of dataloader_config.json file")
    parser.add_argument("--filter_size", default=3, type=int, help="Filter (kernel) size for convolutional layers")
    parser.add_argument("--n_filters_factor", default=1.0, type=float,
                        help="Scale factor with which to modify number of convolutional filters")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for UNet/Generator")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    # hardware configurations applicable to both UNet and GAN
    parser.add_argument("--accelerator", default="auto", type=str, help="PytorchLightning training accelerator")
    parser.add_argument("--devices", default=1, type=int, help="PytorchLightning number of devices to run on")
    parser.add_argument("--n_workers", default=8, type=int, help="Number of workers in dataloader")
    parser.add_argument("--precision", default=16, type=int, choices=[32, 16], help="Precision for training")
    
    # logging configurations applicable to both UNet and GAN
    parser.add_argument("--log_every_n_steps", default=10, type=int, help="How often to log during training")
    parser.add_argument("--max_epochs", default=100, type=int, help="Number of epochs to train")
    parser.add_argument("--num_sanity_val_steps", default=1, type=int, 
                        help="Number of batches to sanity check before training")
    parser.add_argument("--limit_train_batches", default=1.0, type=float, help="Proportion of training dataset to use")
    parser.add_argument("--limit_val_batches", default=1.0, type=float, help="Proportion of validation dataset to use")
    parser.add_argument("--n_to_visualise", default=1, type=int, help="How many forecasts to visualise")
    parser.add_argument("--fast_dev_run", default=False, type=eval, help="Whether to conduct a fast one-batch dev run")

    # let's go
    args = parser.parse_args()
    train_icenet(args)

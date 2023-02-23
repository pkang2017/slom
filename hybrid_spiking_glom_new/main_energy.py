import os
import numpy as np
import warnings
import torch
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule
from datamodules_slom import SmallNORBDataModule, CIFAR100DataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from spikingjelly.clock_driven import surrogate

from model_energy import SpikingGlom
from utils_slom import TwoCropTransform, count_parameters
from custom_transforms_slom import CustomTransforms

import flags_Agglomerator_slom
from absl import app
from absl import flags
FLAGS = flags.FLAGS

torch.cuda.empty_cache()


def init_all():
    warnings.filterwarnings("ignore")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # torch.backends.cudnn.deterministic = True

    pl.seed_everything(FLAGS.seed)
    torch.cuda.empty_cache()


def main(argv):
    init_all()
    wandb_logger = WandbLogger(project="SpikingGlom", name=FLAGS.exp_name)
    wandb_logger.experiment.config.update(FLAGS)

    DataModuleWrapper = {
        "MNIST": MNISTDataModule,
        "FashionMNIST": FashionMNISTDataModule,
        "smallNORB": SmallNORBDataModule,
        "CIFAR10": CIFAR10DataModule,
        "CIFAR100": CIFAR100DataModule,
        "IMAGENET": ImagenetDataModule
    }

    if FLAGS.dataset not in DataModuleWrapper.keys():
        print("❌ Dataset not compatible")
        quit(0)

    dm = DataModuleWrapper[FLAGS.dataset](
        "./datasets", 
        batch_size=FLAGS.batch_size, 
        shuffle=True, 
        pin_memory=True, 
        drop_last=True
    )
    
    ct = CustomTransforms(FLAGS)

    # Apply trainsforms
    if FLAGS.supervise:
        dm.train_transforms = ct.train_transforms[FLAGS.dataset]
        dm.val_transforms = ct.test_transforms[FLAGS.dataset]
        dm.test_transforms = ct.test_transforms[FLAGS.dataset]
    else:
        dm.train_transforms = TwoCropTransform(ct.train_transforms[FLAGS.dataset])
        dm.val_transforms = TwoCropTransform(ct.test_transforms[FLAGS.dataset])
        dm.test_transforms = TwoCropTransform(ct.test_transforms[FLAGS.dataset])

    model = SpikingGlom(FLAGS, surrogate_function=surrogate.ATan())

    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.load_checkpoint_dir)

    print("Total trainable parameters: ", count_parameters(model))

    if FLAGS.mode == "energy":
        
        datasplits = [dm.val_dataloader]
        modes = ["Validation"]
        features_names = ['/features_val']
        labels_names = ['/labels_val']

        for i, (d, m, f, l) in enumerate(zip(datasplits, modes, features_names, labels_names)):
            model = model.load_from_checkpoint(checkpoint_dir, FLAGS=FLAGS, strict=False)
            model.configure_optimizers()

            dm.prepare_data()
            dm.setup()

            trainer = pl.Trainer(
                gpus=-1, 
                strategy='dp', 
                resume_from_checkpoint=checkpoint_dir, 
                max_epochs=FLAGS.max_epochs,
                limit_train_batches=FLAGS.limit_train, 
                limit_val_batches=FLAGS.limit_val, 
                limit_test_batches=FLAGS.limit_test
            )

            trainer.test(model, dataloaders=d())
            print("fr1", sum(model.fr1) / len(model.fr1))
            print("fr2", sum(model.fr2) / len(model.fr2))
            print("fr_bu1", sum(model.fr_bu1) / len(model.fr_bu1))
            print("fr_bu2", sum(model.fr_bu2) / len(model.fr_bu2))
            print("fr_td1", sum(model.fr_td1) / len(model.fr_td1))
            print("fr_td2", sum(model.fr_td2) / len(model.fr_td2))


if __name__ == '__main__':
    app.run(main)

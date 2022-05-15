"""Train script.

Usage:
    train.py <hparams> <dataset> <dataset_root>
"""
import os
import vision
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.trainer import Trainer
from glow.config import JsonConfig


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset = args["<dataset>"]
    dataset_root = args["<dataset_root>"]
    if False:
        assert dataset in vision.Datasets, (
            "`{}` is not supported, use `{}`".format(dataset, vision.Datasets.keys()))
        assert os.path.exists(dataset_root), (
            "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    hparams = JsonConfig(hparams)

    #hparams.Dir.log_root = hparams.Dir.log_root.replace('result/','resultv2/')
    built = build(hparams, True)
    if hparams.Data.dataset== 'celeba':
        dataset = vision.Datasets[dataset]
        # set transform of dataset
        transform = transforms.Compose([
            transforms.CenterCrop(hparams.Data.center_crop),
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor()])
        # build graph and dataset
        dataset = dataset(dataset_root, transform=transform)
    elif hparams.Data.dataset == 'clever':
        dataset_root = "/home/thu/dataset/CLEVR_v1.0/images/train"
        from vision.datasets.clever import CleverDataset
        # set transform of dataset
        transform = transforms.Compose([
            transforms.CenterCrop(hparams.Data.center_crop),
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor()])
        # build graph and dataset
        dataset = CleverDataset(dataset_root, transform=transform)
    elif hparams.Data.dataset == 'celebmask256':
        from vision.datasets.celebamask30k_1024 import CelebAMaskHQ256
        dataset = CelebAMaskHQ256(n_bits=hparams.Data.n_bits, mode=True)
    else:raise

    # begin to train
    trainer = Trainer(**built, dataset=dataset, hparams=hparams)
    trainer.train()

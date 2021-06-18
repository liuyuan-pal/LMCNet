import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/lmcnet/lmcnet_sift_outdoor_test.yaml')
flags = parser.parse_args()

trainer = Trainer(load_cfg(flags.cfg))
trainer.run()
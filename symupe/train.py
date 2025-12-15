""" A minimal training script. """

import argparse
import os
from shutil import copyfile

from symupe.experiments import Trainer, ExperimentModules

if __name__ == "__main__":
    parser = argparse.ArgumentParser("training the model")
    parser.add_argument("--config-root", "-r", type=str, default="../recipes")
    parser.add_argument("--config-name", "-n", type=str, default="symulu/cfm/config.yaml")

    args = parser.parse_args()

    exp_comps = ExperimentModules(
        config=args.config_name,
        config_root=args.config_root
    )
    modules = exp_comps.init_modules()

    trainer = Trainer(
        **modules,
        config=exp_comps.config
    )

    copyfile(os.path.join(args.config_root, args.config_name), os.path.join(trainer.config.output_dir, "config.yaml"))

    trainer.train()

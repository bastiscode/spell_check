import argparse
import os
import pickle
import pprint
import shutil

import omegaconf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiments", nargs="+", type=str, required=True)
    parser.add_argument("-d", "--dry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OLD_PREFIX = "GNN_LIB_"
    NEW_PREFIX = "NSC_"

    OLD_DIR_NAME = "spelling_correction"
    NEW_DIR_NAME = "spell_checking"

    for experiment in args.experiments:
        with open(os.path.join(experiment, "cfg.pkl"), "rb") as inf:
            cfg, env_vars = pickle.load(inf)

        shutil.copy2(os.path.join(experiment, "cfg.pkl"), os.path.join(experiment, "cfg_backup.pkl"))

        new_env_vars = {
            f"{NEW_PREFIX}{k[len(OLD_PREFIX):]}": v.replace(OLD_DIR_NAME, NEW_DIR_NAME)
            for k, v in env_vars.items()
        }

        yaml_string = omegaconf.OmegaConf.to_yaml(cfg, resolve=False)
        new_yaml_config = yaml_string.replace(OLD_PREFIX, NEW_PREFIX)
        with open("cfg_temp.yaml", "w") as of:
            of.write(new_yaml_config)

        new_cfg = omegaconf.OmegaConf.load("cfg_temp.yaml")

        print()
        print(f"Experiment {experiment}")
        print("New env variables:", pprint.pformat(new_env_vars), sep="\n")
        print("New config:", omegaconf.OmegaConf.to_yaml(new_cfg, resolve=False), sep="\n")

        if not args.dry:
            with open(os.path.join(experiment, "cfg.pkl"), "wb") as of:
                pickle.dump((new_cfg, new_env_vars), of)

        os.remove("cfg_temp.yaml")

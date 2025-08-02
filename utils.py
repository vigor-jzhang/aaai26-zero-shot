import os
import logging
from omegaconf import OmegaConf
from datetime import datetime


def get_config(config_path):
    config = OmegaConf.load(config_path)
    return config


def folder_create(config):
    config.experiment.work_dir = config.experiment.work_dir + '/' + config.experiment.task_name + '/' + config.experiment.net_name
    os.makedirs(config.experiment.work_dir, exist_ok=True)
    config.experiment.ckpt_dir = config.experiment.work_dir + '/ckpt'
    os.makedirs(config.experiment.ckpt_dir, exist_ok=True)
    config.experiment.log_dir = config.experiment.work_dir + '/logs'
    os.makedirs(config.experiment.log_dir, exist_ok=True)
    return config


def setup_logger(config):
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H_%M_%S")
    logging.basicConfig(
        filename=f"{config.experiment.log_dir}/log-{time_string}.txt",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")
    return logger


def adopt_weight(disc_factor, i, threshold, value=0.):
    if i < threshold:
        disc_factor = value
    return disc_factor


if __name__ == '__main__':
    config = get_config('./configs/ses1t1-unc-config.yaml')
    print(config)
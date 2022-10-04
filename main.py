import datetime
import random
import torch
import numpy as np
from config import get_config
from runner import MultiWOZRunner
import sys
from utils.io_utils import get_or_create_logger

logger = get_or_create_logger(__name__)

def main():
    """ main function """
    cfg = get_config()
    cuda_available= torch.cuda.is_available()

    if cuda_available:
        if cfg.num_gpus>1:
            logger.info('Using Multi-GPU training, number of GPU is {}'.format(cfg.num_gpus))
            torch.cuda.set_device(cfg.local_rank)
            device = torch.device('cuda', cfg.local_rank)
            t=datetime.timedelta(days=1, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
            torch.distributed.init_process_group(backend='nccl',timeout=t)
        else:
            logger.info('Using single GPU training.')
            device = torch.device("cuda")
    else:
        device=torch.device("cpu")

    setattr(cfg, "device", device)

    logger.info("Device: %s (the number of GPUs: %d)", str(device), cfg.num_gpus)

    if cfg.seed > 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to %d", cfg.seed)

    runner = MultiWOZRunner(cfg)

    if cfg.run_type == "train":
        runner.train()
    else:
        runner.predict()


if __name__ == "__main__":
    main()

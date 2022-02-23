import argparse
import datetime
import os
import pickle
import time
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch import distributed as dist
from torch.backends import cudnn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch.utils import tensorboard
from tqdm import tqdm

import gnn_lib
from gnn_lib.data import utils
from gnn_lib.modules.lr_scheduler import get_lr_scheduler_from_config
from gnn_lib.modules.optimizer import get_optimizer_from_config
from gnn_lib.tasks import utils as task_utils
from gnn_lib.utils import common, data_containers, config, io
from gnn_lib.utils.distributed import DistributedDevice, unwrap_ddp

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(args: argparse.Namespace, device: DistributedDevice) -> None:
    logger = common.get_logger("TRAIN")
    base_dir = os.getcwd()

    # create config
    if args.config is not None:
        cfg = OmegaConf.load(args.config)
        OmegaConf.resolve(cfg)
        schema = OmegaConf.structured(config.TrainConfig)
        cfg = OmegaConf.merge(schema, cfg)
        resuming_training = False
    else:
        cfg = gnn_lib.load_experiment_config(args.resume)
        resuming_training = True

    logger.info(f"Using distributed device: {device}")

    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    if device.is_main_process and not resuming_training:
        # retry up to ten times with two seconds delay
        # because if two similar experiments are started in the same second they might have the same directory name
        num_retries = 10
        experiment_dir = None
        for retry in range(num_retries):
            experiment_dir = os.path.relpath(
                os.path.join(
                    cfg.experiment_dir,
                    cfg.variant.type,
                    f"{cfg.experiment_name}_"
                    f"{'_'.join(os.path.basename(dataset) for dataset in cfg.datasets)}_"
                    f"{datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
                ),
                base_dir
            )
            try:
                os.makedirs(experiment_dir)
                os.makedirs(os.path.join(experiment_dir, "checkpoints"))
                break
            except OSError:
                # wait for two seconds before retry
                if retry + 1 == num_retries:
                    logger.fatal(f"Unable to create experiment")
                    exit(1)
                else:
                    time.sleep(2)
        assert experiment_dir is not None

        # save config and environment variables as pickle
        with open(os.path.join(experiment_dir, "cfg.pkl"), "wb") as f:  # type: ignore
            env = {
                env_var_name: env_var_val
                for env_var_name, env_var_val in os.environ.items() if env_var_name.startswith("GNN_LIB_")
            }
            unresolved_cfg = OmegaConf.load(args.config)
            pickle.dump((unresolved_cfg, env), f)  # type: ignore
    elif resuming_training:
        experiment_dir = args.resume
    else:
        experiment_dir = ""

    task = gnn_lib.get_task(
        variant_cfg=cfg.variant,
        checkpoint_dir=os.path.join(experiment_dir, "checkpoints"),
        seed=cfg.seed
    )
    # IMPORTANT: the sample graph needs to contain
    # all possible node and edge types
    sample_g, _ = task.generate_sample_inputs(2)
    model = task.get_model(
        sample_g=sample_g,
        cfg=cfg.model,
        device=device.device,
    )

    optimizer = get_optimizer_from_config(
        cfg=cfg.optimizer,
        model=model
    )
    grad_scaler = amp.GradScaler(enabled=cfg.mixed_precision)

    if resuming_training:
        # load last checkpoint
        last_checkpoint_path = os.path.join(experiment_dir, "checkpoints", "checkpoint_last.pt")
        last_checkpoint = io.load_checkpoint(last_checkpoint_path, device.device)
        optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
        model.load_state_dict(last_checkpoint["model_state_dict"])

        task.step = last_checkpoint["step"]
        task.best_val_loss = last_checkpoint["val_loss"]

        if device.is_main_process:
            logger.info(f"Successfully loaded last checkpoint from {last_checkpoint_path}")
            logger.info(f"Resuming from training step {task.step} with a best validation loss of "
                        f"{task.best_val_loss:.4f}")

    elif not resuming_training and cfg.start_from_checkpoint is not None:
        # init model weights from checkpoint
        checkpoint = io.load_checkpoint(cfg.start_from_checkpoint, device.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if device.is_main_process:
            logger.info(f"Successfully loaded weights from {cfg.start_from_checkpoint}")

    model = DistributedDataParallel(model, device_ids=[device.local_rank], output_device=device.local_rank)

    # this makes sure we can use DDP with parameter sharing and gradient checkpointing,
    # but we have to make sure that our models to not have any control flow in them (e.g. if statements),
    # which is not the case for our models
    # see: https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/distributed.py#L1627
    model._set_static_graph()

    train_dataset, val_dataset = gnn_lib.data.get_train_val_datasets(
        datasets=cfg.datasets,
        variant_cfg=cfg.variant,
        seed=cfg.seed,
        val_splits=cfg.val_splits,
        dataset_limits=cfg.dataset_limits
    )

    available_cpu_cores = len(os.sched_getaffinity(0))
    num_workers = cfg.num_workers if cfg.num_workers is not None \
        else max((available_cpu_cores - device.local_world_size) // device.local_world_size, 2)
    if device.is_main_process:
        logger.info(f"Using {num_workers} dataloader workers per GPU process")

    if cfg.batch_max_length is None:
        train_generator = torch.Generator()
        train_generator = train_generator.manual_seed(cfg.seed)
        train_batch_sampler = utils.DistributedDynamicSampler(
            sampler=data.BatchSampler(
                sampler=data.RandomSampler(train_dataset, generator=train_generator),
                batch_size=max(cfg.batch_size // device.world_size, 1),
                drop_last=True
            ),
            device=device,
            seed=cfg.seed,
            shuffle=True,
            drop_last=True
        )
        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=utils.graph_collate,
            num_workers=num_workers
        )

        val_batch_sampler = data.BatchSampler(
            sampler=data.SequentialSampler(val_dataset),
            batch_size=max(cfg.batch_size // device.world_size, 1),
            drop_last=False
        )
        val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_sampler=val_batch_sampler,
            collate_fn=utils.graph_collate,
            num_workers=num_workers
        )
    else:
        train_batch_sampler = utils.DistributedDynamicSampler(
            sampler=utils.BucketSampler(
                dataset=train_dataset,
                values=train_dataset.get_lengths(),
                batch_max_value=cfg.batch_max_length // device.world_size,
                bucket_span=cfg.bucket_span,
                seed=cfg.seed,
                shuffle=True
            ),
            device=device,
            seed=cfg.seed,
            drop_last=True,
            shuffle=True
        )
        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=utils.graph_collate,
            num_workers=num_workers
        )

        val_batch_sampler = utils.BucketSampler(
            dataset=val_dataset,
            values=val_dataset.get_lengths(),
            batch_max_value=cfg.batch_max_length // device.world_size,
            bucket_span=cfg.bucket_span,
            seed=cfg.seed,
            shuffle=False
        )
        val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_sampler=val_batch_sampler,
            collate_fn=utils.graph_collate,
            num_workers=num_workers
        )

    ema: Optional[task_utils.EMA] = None
    ema_update_every = 1
    if device.is_main_process:
        logger.info(f"Using config:\n{OmegaConf.to_yaml(cfg)}")
        logger.info(f"Model:\n{unwrap_ddp(model)}")
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPUs in environment, torch threads set to {torch.get_num_threads()}")
        for gpu in range(num_gpus):
            device_props = torch.cuda.get_device_properties(gpu)
            logger.info(f"[GPU:{gpu}] {device_props.name}, {device_props.total_memory // 1024 // 1024:.0f}MiB "
                        f"({device_props.major}.{device_props.minor}, {device_props.multi_processor_count})")
        logger.info(f"Train dataset contains {len(train_dataset)} samples")
        logger.info(f"Validation dataset contains {len(val_dataset)} samples")
        logger.info(f"Model has "
                    f"{common.get_num_parameters(model)['total']:,} "
                    f"parameters")
        # only use tensorboard writer on main process such that we do not log multiple times
        writer = tensorboard.SummaryWriter(
            log_dir=os.path.join(experiment_dir, "tensorboard"),
            flush_secs=15
        )
        # only use ema on main process because we also only save models on main process
        if cfg.exponential_moving_average is not None:
            assert len(cfg.exponential_moving_average) == 2
            ema_factor = cfg.exponential_moving_average[0]
            ema_update_every = int(cfg.exponential_moving_average[1])

            ema = task_utils.EMA(
                model=unwrap_ddp(model),
                ema_factor=ema_factor
            )
    else:
        writer = None

    # create lr scheduler after prepare because
    # train loader length could be different now
    # (when running on multiple GPUs)
    lr_scheduler = None
    if cfg.lr_scheduler:
        lr_scheduler = get_lr_scheduler_from_config(
            cfg=cfg.lr_scheduler,
            optimizer=optimizer,
            num_training_steps=len(train_loader) * cfg.epochs
        )
        if resuming_training and "lr_scheduler_state_dict" in last_checkpoint:
            lr_scheduler.load_state_dict(last_checkpoint["lr_scheduler_state_dict"])

    unused_parameters = task.disable_unused_parameters(
        model, device, grad_scaler
    )

    start_epoch = 0
    steps_to_fast_forward = 0
    if resuming_training:
        # this assumes that train loader length does not change across epochs which is usually the case
        start_epoch = task.step // len(train_loader)
        steps_to_fast_forward = task.step % max(len(train_loader), 1)
        logger.info(f"Resuming training at epoch {start_epoch + 1}, fast forwarding "
                    f"{steps_to_fast_forward} training steps to step {task.step}")

    for e in tqdm(range(start_epoch, cfg.epochs),
                  total=cfg.epochs - start_epoch,
                  desc="Epoch",
                  disable=not device.is_main_process or common.disable_tqdm()):
        train_batch_sampler.set_epoch(e)
        train_batch_sampler.set_steps_to_fast_forward(steps_to_fast_forward)

        task.train(
            train_loader,
            model,
            optimizer,
            grad_scaler,
            writer,
            device,
            lr_scheduler,
            val_loader,
            log_per_epoch=cfg.log_per_epoch,
            eval_per_epoch=cfg.eval_per_epoch,
            keep_last_n_checkpoints=cfg.keep_last_n_checkpoints,
            ema=ema,
            ema_update_every=ema_update_every,
            steps_to_fast_forward=steps_to_fast_forward
        )

        # to not fast forward next epoch, set to zero
        steps_to_fast_forward = 0

    if device.is_main_process:
        hyperparams = {
            "experiment": cfg.experiment_name,
            "variant": cfg.variant.type,
            "data": ", ".join(os.path.basename(dataset) for dataset in cfg.datasets),
            "gnn": cfg.model.gnn.type,
            "optimizer": cfg.optimizer.type,
            "lr": cfg.optimizer.lr,
            "epochs": cfg.epochs,
            "num_parameters": ", ".join(
                f"{k}={v:,}" for k, v in sorted(common.get_num_parameters(model, unused_parameters).items())
            ),
            "seed": cfg.seed
        }
        if cfg.batch_max_length:
            hyperparams["batch_max_length"] = cfg.batch_max_length
        else:
            hyperparams["batch_size"] = cfg.batch_size
        if cfg.lr_scheduler:
            hyperparams["lr_scheduler"] = cfg.lr_scheduler.type

        hyperparams_container = data_containers.HyperparameterContainer(name="Hyperparams")
        hyperparams_container.add((hyperparams, task.metrics()))
        hyperparams_container.log_to_tensorboard(writer, task.step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    train_group = parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument("--config", type=str, default=None)
    train_group.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def initialize() -> DistributedDevice:
    assert torch.cuda.device_count() > 0, "need at least one GPU for training, but found none"
    assert dist.is_available(), "distributed package must be available for training"
    assert dist.is_nccl_available(), "nccl backend for distributed training must be available"
    logger = common.get_logger("TRAIN_INITIALIZE")
    logger.info(f"Found {torch.cuda.device_count()} GPUs "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    assert (
            "MASTER_ADDR" in os.environ
            and "MASTER_PORT" in os.environ
            and "WORLD_SIZE" in os.environ
    ), f"could not find at least one of MASTER_ADDR, MASTER_PORT and WORLD_SIZE env variables"
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    world_size = int(os.environ["WORLD_SIZE"])

    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(rank % torch.cuda.device_count())
        local_world_size = torch.cuda.device_count()
        logger.info(
            f"Running on Slurm Cluster: master_addr={master_addr}, master_port={master_port}, "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}, local_world_size={local_world_size}"
        )
    else:
        assert (
                "RANK" in os.environ
                and "LOCAL_RANK" in os.environ
        ), "could not find RANK and LOCAL_RANK env variables, you probably did not use torchrun to run this script"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        logger.info(
            f"Running using torchrun: master_addr={master_addr}, master_port={master_port}, "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}, local_world_size={local_world_size}"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=dist.Backend.NCCL,
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    assert dist.is_initialized(), "failed to initialize process group"

    return DistributedDevice(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size
    )


def de_initialize() -> None:
    dist.destroy_process_group()


if __name__ == "__main__":
    # mp.set_start_method("spawn")
    train(parse_args(), initialize())
    de_initialize()

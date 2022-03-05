import enum
import os
import time
from typing import Any, Type, Dict, Optional, Tuple, List, Set

import dgl
import omegaconf
import psutil
import torch
from torch import optim, nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn_lib import models
from gnn_lib.data import variants, utils as data_utils
from gnn_lib.data.variants import get_variant_from_config, DatasetVariants
from gnn_lib.models import Model
from gnn_lib.tasks import utils
from gnn_lib.utils import io, common, data_containers
from gnn_lib.utils.distributed import DistributedDevice, unwrap_ddp


class Tasks(enum.IntEnum):
    GRAPH_CLASSIFICATION = 1
    MULTI_NODE_CLASSIFICATION = 2
    TOKENIZATION_REPAIR = 3
    TOKENIZATION_REPAIR_NMT = 4
    SEC_WORDS_NMT = 5
    SEC_NMT = 6
    SED_WORDS = 7


class Task:
    expected_model: Type = Model

    def __init__(self,
                 variant_cfg: variants.DatasetVariantConfig,
                 checkpoint_dir: str,
                 seed: int):
        self.step = 0
        self.best_val_loss = float("inf")
        self.checkpoint_dir = checkpoint_dir

        self.variant_cfg = variant_cfg
        self.seed = seed
        self.variant = get_variant_from_config(variant_cfg, seed)

        self.logger = common.get_logger("TASK")

    def generate_sample_inputs(self, num_samples: int) -> Tuple[dgl.DGLHeteroGraph, List[Dict[str, Any]]]:
        return data_utils.graph_collate([
            self.variant.prepare_sequence(utils.SAMPLE_SEQUENCE) for _ in range(num_samples)
        ])

    def disable_unused_parameters(self,
                                  model: DDP,
                                  device: DistributedDevice,
                                  grad_scaler: amp.GradScaler) -> Set[str]:
        batch = self.generate_sample_inputs(num_samples=2)
        inputs, _ = self._prepare_inputs_and_labels(batch, device)

        with amp.autocast(enabled=grad_scaler.is_enabled()):
            outputs, _ = model(**inputs)

            if isinstance(outputs, dict):
                loss = sum(v.sum() for v in outputs.values())
            elif isinstance(outputs, torch.Tensor):
                loss = outputs.sum()
            else:
                raise ValueError(f"Expected model output to be either dictionary of tensors or tensor")

        grad_scaler.scale(loss).backward()

        unused_parameters = set()
        for name, p in model.named_parameters():
            if p.grad is None and p.requires_grad:
                unused_parameters.add(name)
                p.requires_grad = False

        if device.is_main_process:
            sample_g: dgl.DGLHeteroGraph = dgl.unbatch(inputs["g"])[0]
            self.logger.info(
                f"Sample graph: {sample_g} "
                f"({list((n_type, sample_g.node_attr_schemes(n_type)) for n_type in sample_g.ntypes)}, "
                f"{list((e_type, sample_g.edge_attr_schemes(e_type)) for e_type in sample_g.canonical_etypes)}"
            )

            for param in sorted(unused_parameters):
                self.logger.info(f"Got unused parameter {param}, setting requires_grad=False")

        return unused_parameters

    def _get_additional_stats(self, model: models.Model) -> Dict[str, data_containers.DataContainer]:
        return {}

    def _prepare_inputs_and_labels(self,
                                   batch: Tuple[dgl.DGLHeteroGraph, List[Dict[str, Any]]],
                                   device: DistributedDevice) \
            -> Tuple[Dict[str, Any], Any]:
        raise NotImplementedError

    def _calc_loss(self,
                   labels: Any,
                   model_output: Any,
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def _update_stats(self,
                      model: models.Model,
                      inputs: Dict[str, Any],
                      labels: Any,
                      model_output: Any,
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      total_steps: int) -> None:
        return

    def train(
            self,
            train_loader: DataLoader,
            model: DDP,
            optimizer: optim.Optimizer,
            grad_scaler: amp.GradScaler,
            writer: tensorboard.SummaryWriter,
            device: DistributedDevice,
            lr_scheduler: Optional[optim.lr_scheduler.LambdaLR] = None,
            val_loader: Optional[DataLoader] = None,
            log_per_epoch: int = 0,
            eval_per_epoch: int = 1,
            keep_last_n_checkpoints: int = 3,
            ema: Optional[utils.EMA] = None,
            ema_update_every: int = 100,
            steps_to_fast_forward: int = 0
    ) -> None:
        unwrapped_model = unwrap_ddp(model)
        self._check_model(unwrapped_model)

        org_train_loader_length = len(train_loader) + steps_to_fast_forward
        num_updates = max(org_train_loader_length, 1)
        log_every = max(num_updates // log_per_epoch, 1)
        eval_every = max(num_updates // eval_per_epoch, 1)

        model = model.to(device.device)
        model = model.train()

        # setup some training statistics to log to tensorboard
        loss_stat = data_containers.AverageScalarContainer(name="train_loss")
        batch_size_stat = data_containers.AverageScalarContainer(name="train_batch_size")
        iteration_perf_stat = data_containers.AverageScalarContainer(name="train_iteration_performance")
        forward_pass_perf_stat = data_containers.AverageScalarContainer(name="train_forward_pass_performance")
        backward_and_update_perf_stat = data_containers.AverageScalarContainer(
            name="train_backward_and_update_performance"
        )
        if device.is_main_process:
            stats = self._get_additional_stats(unwrapped_model)
            for _, stat in stats.items():
                if not stat.name.startswith("train_"):
                    stat.name = "train_" + stat.name
        else:
            stats = {}

        start = time.perf_counter()
        for i, batch in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc="Training",
                leave=False,
                disable=not device.is_main_process or common.disable_tqdm()
        ):
            start_iteration = time.perf_counter()
            iteration = i + 1 + steps_to_fast_forward

            self.step += 1
            inputs, labels = self._prepare_inputs_and_labels(batch, device)

            input_g: dgl.DGLHeteroGraph = inputs["g"]
            if (any(input_g.num_edges(e_type) == 0 for e_type in input_g.canonical_etypes) or
                    any(input_g.num_nodes(n_type) == 0 for n_type in input_g.ntypes)):
                self.logger.warning(f"Input graph does not contain all node or edge types: \n{input_g}")
                continue

            with amp.autocast(enabled=grad_scaler.is_enabled()):
                start_forward_pass = time.perf_counter()

                output, losses = model(**inputs)

                loss = self._calc_loss(
                    labels=labels,
                    model_output=output,
                    additional_losses=losses
                )

                end_forward_pass = time.perf_counter()

            start_backward_and_update = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            end_backward_and_update = time.perf_counter()

            if not device.is_main_process:
                # ema, logging and evaluation is only done on main process
                continue

            if ema is not None and self.step % ema_update_every == 0:
                ema.update()

            # update stats
            self._update_stats(unwrapped_model, inputs, labels, output, stats, self.step, log_every)
            loss_stat.add(loss.item())
            # this is an approximation, since we only record batch statistics on the main process, but
            # want to log the overall batch size in tensorboard, we multiply by the world size here
            batch_size_stat.add(input_g.batch_size * device.world_size)
            forward_pass_perf_stat.add((end_forward_pass - start_forward_pass) * 1000)
            backward_and_update_perf_stat.add((end_backward_and_update - start_backward_and_update) * 1000)
            # we time the iteration here because we do not want to count logging and evaluation, which is
            # anyway done infrequently, to the duration of an average iteration
            end_iteration = time.perf_counter()
            iteration_perf_stat.add((end_iteration - start_iteration) * 1000)

            if self.step % log_every == 0:
                max_mem_usage = 0
                for device_id in range(device.local_world_size):
                    mem_total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                    mem_alloc = torch.cuda.max_memory_reserved(device_id) / 1024**3
                    mem_usage = mem_alloc / mem_total
                    if mem_usage > max_mem_usage:
                        max_mem_usage = mem_usage

                writer.add_scalar(
                    "max_gpu_memory_usage",
                    max_mem_usage,
                    self.step
                )

                writer.add_scalar(
                    "node_cpu_usage",
                    psutil.cpu_percent(),
                    self.step
                )

                writer.add_scalar(
                    "node_ram_usage",
                    psutil.virtual_memory().percent,
                    self.step
                )

                if common.disable_tqdm():
                    self.logger.info(f"[{self.step}|{iteration}/{org_train_loader_length}] "
                                     f"train_loss={loss_stat.value:.5f}")
                    minutes = (time.perf_counter() - start) / 60
                    self.logger.info(
                        f"[{self.step}|{iteration}/{org_train_loader_length}] "
                        f"{common.eta_minutes(minutes, i + 1, len(train_loader))}"
                    )
                    self.logger.info(f"GPU 0:\n{torch.cuda.memory_summary(0, True)}")
                    torch.cuda.reset_peak_memory_stats()

                loss_stat.log_to_tensorboard(writer, self.step)
                loss_stat.reset()

                batch_size_stat.log_to_tensorboard(writer, self.step)
                batch_size_stat.reset()

                forward_pass_perf_stat.log_to_tensorboard(writer, self.step)
                forward_pass_perf_stat.reset()

                iteration_perf_stat.log_to_tensorboard(writer, self.step)
                iteration_perf_stat.reset()

                backward_and_update_perf_stat.log_to_tensorboard(writer, self.step)
                backward_and_update_perf_stat.reset()

                for _, stat in stats.items():
                    stat.log_to_tensorboard(writer, self.step)
                    stat.reset()

                if lr_scheduler:
                    writer.add_scalar(
                        "train_lr",
                        lr_scheduler.get_last_lr()[0],
                        self.step
                    )

            if self.step % eval_every == 0:
                assert val_loader is not None, "validation loader must be given if you want" \
                                               "to validate during the training epochs"

                val_loss, best = self.evaluate(val_loader,
                                               model if ema is None else ema.ema_model,
                                               writer,
                                               device,
                                               grad_scaler)

                if common.disable_tqdm():
                    self.logger.info(f"[{self.step}|{iteration}/{org_train_loader_length}] "
                                     f"val_loss={val_loss:.5f}, best={best}")

                self.save_checkpoint(
                    unwrapped_model if ema is None else ema.ema_model,
                    device,
                    val_loss,
                    best,
                    optimizer,
                    lr_scheduler,
                    keep_last_n_checkpoints
                )

                model = model.train()

    @torch.no_grad()
    def evaluate(
            self,
            val_loader: DataLoader,
            model: DDP,
            writer: tensorboard.SummaryWriter,
            device: DistributedDevice,
            grad_scaler: amp.GradScaler
    ) -> Tuple[float, bool]:
        unwrapped_model = unwrap_ddp(model)
        self._check_model(unwrapped_model)

        assert device.is_main_process, "evaluation should only be done on the main process"

        model = model.to(device.device)
        model = model.eval()

        # setup some evaluation statistics to log to tensorboard
        loss_stat = data_containers.AverageScalarContainer(name="val_loss")
        stats = self._get_additional_stats(unwrapped_model)
        for _, stat in stats.items():
            if not stat.name.startswith("val_"):
                stat.name = "val_" + stat.name

        for i, batch in tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc="Evaluating",
                leave=False,
                disable=common.disable_tqdm()
        ):
            inputs, labels = self._prepare_inputs_and_labels(batch, device)

            with amp.autocast(enabled=grad_scaler.is_enabled()):
                output, losses = model(**inputs)

                loss = self._calc_loss(
                    labels=labels,
                    model_output=output,
                    additional_losses=losses
                )
                loss_stat.add(loss.item())

            # update stats
            self._update_stats(unwrapped_model, inputs, labels, output, stats, i + 1, len(val_loader))

        loss_stat.log_to_tensorboard(writer, self.step)

        for _, stat in stats.items():
            stat.log_to_tensorboard(writer, self.step)

        best = loss_stat.value < self.best_val_loss
        if best:
            self.best_val_loss = loss_stat.value

        return loss_stat.value, best

    @torch.inference_mode()
    def inference(
            self,
            model: Model,
            inputs: Any,
            **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    def get_model(self,
                  sample_g: dgl.DGLHeteroGraph,
                  cfg: omegaconf.DictConfig,
                  device: torch.device) -> Model:
        model = models.get_model_from_config(cfg, sample_g, device).to(device)
        self._check_model(model)
        return model

    def save_checkpoint(self,
                        model: Model,
                        device: DistributedDevice,
                        val_loss: float,
                        best: bool = False,
                        optimizer: Optional[optim.Optimizer] = None,
                        lr_scheduler: Optional[optim.lr_scheduler.LambdaLR] = None,
                        keep_last_n_checkpoints: Optional[int] = None) -> None:
        if not device.is_main_process:
            return

        if best:
            io.save_checkpoint(
                os.path.join(self.checkpoint_dir, "checkpoint_best.pt"),
                model,
                self.step,
                val_loss,
                optimizer,
                lr_scheduler
            )

        io.save_checkpoint(
            os.path.join(self.checkpoint_dir, "checkpoint_last.pt"),
            model,
            self.step,
            val_loss,
            optimizer,
            lr_scheduler
        )

        if keep_last_n_checkpoints is not None and keep_last_n_checkpoints > 0:
            io.save_checkpoint(
                os.path.join(self.checkpoint_dir, f"checkpoint_{self.step}.pt"),
                model,
                self.step,
                val_loss,
                optimizer,
                lr_scheduler
            )
            # potentially delete too old checkpoints
            checkpoints = io.glob_safe(os.path.join(self.checkpoint_dir, "checkpoint_*.pt"), error_on_empty=False)
            checkpoints = [
                cp
                for cp in checkpoints
                if os.path.basename(cp) != "checkpoint_best.pt" and os.path.basename(cp) != "checkpoint_last.pt"
            ]
            for cp in common.natural_sort(checkpoints)[:-keep_last_n_checkpoints]:
                os.remove(cp)

    def load_best(self, model: Model) -> None:
        best_checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "checkpoint_best.pt"
        )
        assert os.path.exists(best_checkpoint_path), \
            f"Expected path {best_checkpoint_path} to exist, but it does not"
        checkpoint = io.load_checkpoint(
            best_checkpoint_path,
            model.device
        )
        model.load_state_dict(checkpoint["model_state_dict"])

    def metrics(self) -> Dict[str, float]:
        return {"best_val_loss": self.best_val_loss}

    def _check_model(self, model: nn.Module) -> None:
        if not isinstance(model, self.expected_model):
            raise ValueError(
                f"expected a model of type "
                f"{self.expected_model.__name__}, but got {type(model)}"
            )


def get_task(
        checkpoint_dir: str,
        variant_cfg: variants.DatasetVariantConfig,
        seed: int
) -> Task:
    from gnn_lib.tasks.tokenization_repair import TokenizationRepair
    from gnn_lib.tasks.tokenization_repair_nmt import TokenizationRepairNMT
    from gnn_lib.tasks.sed_sequence import SEDSequence
    from gnn_lib.tasks.sec_nmt import SECNMT
    from gnn_lib.tasks.sec_words_nmt import SECWordsNMT
    from gnn_lib.tasks.sed_words import SEDWords

    variant_type = DatasetVariants[variant_cfg.type]
    if variant_type == DatasetVariants.SED_SEQUENCE:
        return SEDSequence(variant_cfg, checkpoint_dir, seed)

    elif variant_type == DatasetVariants.SED_WORDS:
        return SEDWords(variant_cfg, checkpoint_dir, seed)

    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR:
        return TokenizationRepair(variant_cfg, checkpoint_dir, seed)

    elif variant_type == DatasetVariants.SEC_NMT:
        return SECNMT(variant_cfg, checkpoint_dir, seed)

    elif variant_type == DatasetVariants.SEC_WORDS_NMT:
        return SECWordsNMT(variant_cfg, checkpoint_dir, seed)

    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR_NMT:
        return TokenizationRepairNMT(variant_cfg, checkpoint_dir, seed)

    else:
        raise ValueError(f"Could not determine task from variant type {variant_type}")

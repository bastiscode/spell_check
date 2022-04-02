import enum
import os
import time
from typing import Any, Type, Dict, Optional, Tuple, Set, Union, List

import dgl
import omegaconf
import psutil
import torch
from torch import optim, nn
from torch.distributed import optim as dist_optim
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
from gnn_lib.utils import io, common, data_containers, Batch
from gnn_lib.utils.distributed import DistributedDevice, unwrap_ddp


class Tasks(enum.IntEnum):
    GRAPH_CLASSIFICATION = 1
    MULTI_NODE_CLASSIFICATION = 2
    TOKENIZATION_REPAIR = 3
    SEC_WORDS_NMT = 4
    SEC_NMT = 5
    SED_WORDS = 6


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

    def generate_sample_inputs(self, num_samples: int) -> Batch:
        return data_utils.collate([
            self.variant.get_inputs(
                utils.SAMPLE_SEQUENCE
            ) for _ in range(num_samples)
        ])

    def disable_unused_parameters(
            self,
            model: DDP,
            device: DistributedDevice
    ) -> Set[str]:
        batch = self.generate_sample_inputs(num_samples=1)
        inputs, _ = self._prepare_inputs_and_labels(batch, device.device)

        unused_parameters = utils.get_unused_parameters(model, **inputs)
        # disable unused parameters
        for name, p in model.named_parameters():
            if name in unused_parameters:
                p.requires_grad = False

        if device.is_main_process:
            sample_data = batch.data
            if isinstance(sample_data, dgl.DGLHeteroGraph):
                sample_g = dgl.unbatch(sample_data)[0]
                self.logger.info(
                    f"Sample graph: {sample_g}\n"
                    f"node attributes: "
                    f"{list((n_type, sample_g.node_attr_schemes(n_type)) for n_type in sample_g.ntypes)}\n"
                    f"edge attributes: "
                    f"{list((e_type, sample_g.edge_attr_schemes(e_type)) for e_type in sample_g.canonical_etypes)}"
                )
            else:
                self.logger.info(
                    f"Sample tensor: {sample_data[0]}"
                )
            sample_info = {k: v[0] for k, v in batch.info.items()}
            self.logger.info(
                f"Sample infos: {sample_info}"
            )

            for param in sorted(unused_parameters):
                self.logger.info(f"Got unused parameter {param}, setting requires_grad=False")

        return unused_parameters

    def _get_additional_stats(self, model: models.Model) -> Dict[str, data_containers.DataContainer]:
        return {}

    def _prepare_inputs_and_labels(
            self,
            batch: Batch,
            device: torch.device
    ) -> Tuple[Dict[str, Any], Any]:
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
            optimizer: Union[optim.Optimizer, dist_optim.ZeroRedundancyOptimizer],
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
            ema_start_at: int = 1,
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
            assert model.training

            start_iteration = time.perf_counter()
            iteration = i + 1 + steps_to_fast_forward

            self.step += 1
            inputs, labels = self._prepare_inputs_and_labels(batch, device.device)

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

            if device.is_main_process:
                # ema and logging is only done on main process
                if ema is not None and ema_start_at >= self.step and self.step % ema_update_every == 0:
                    ema.update(overwrite=ema_start_at == self.step)

                # update stats
                self._update_stats(unwrapped_model, inputs, labels, output, stats, self.step, log_every)
                loss_stat.add(loss.item())
                # this is an approximation, since we only record batch statistics on the main process, but
                # want to log the overall batch size in tensorboard, we multiply by the world size here
                batch_size = utils.get_batch_size_from_data(batch.data)
                batch_size_stat.add(batch_size * device.world_size)
                forward_pass_perf_stat.add((end_forward_pass - start_forward_pass) * 1000)
                backward_and_update_perf_stat.add((end_backward_and_update - start_backward_and_update) * 1000)
                # we time the iteration here because we do not want to count logging and evaluation, which is
                # anyway done infrequently, to the duration of an average iteration
                end_iteration = time.perf_counter()
                iteration_perf_stat.add((end_iteration - start_iteration) * 1000)

                if self.step % log_every == 0:
                    max_mem_usage = 0
                    for device_id in range(device.local_world_size):
                        mem_total = torch.cuda.get_device_properties(device_id).total_memory / 1024 ** 3
                        mem_alloc = torch.cuda.max_memory_reserved(device_id) / 1024 ** 3
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

                if isinstance(optimizer, dist_optim.ZeroRedundancyOptimizer):
                    optimizer.consolidate_state_dict()

                if device.is_main_process:
                    val_loss, best = self.evaluate(val_loader,
                                                   (
                                                       ema.ema_model if ema is not None and ema_start_at >= self.step
                                                       else model
                                                   ),
                                                   writer,
                                                   device,
                                                   grad_scaler)

                    if common.disable_tqdm():
                        self.logger.info(f"[{self.step}|{iteration}/{org_train_loader_length}] "
                                         f"val_loss={val_loss:.5f}, best={best}")

                    self.save_checkpoint(
                        (
                            ema.ema_model if ema is not None and ema_start_at >= self.step
                            else unwrapped_model
                        ),
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
            assert not model.training

            inputs, labels = self._prepare_inputs_and_labels(batch, device.device)

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

    def get_model(self,
                  sample_inputs: Batch,
                  cfg: omegaconf.DictConfig,
                  device: torch.device) -> Model:
        model = models.get_model_from_config(cfg, sample_inputs, device).to(device)
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
            raise ValueError(f"expected a model of type {self.expected_model.__name__}, but got {type(model)}")

    def _split_sample_for_inference(
            self,
            sample: data_utils.Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        raise NotImplementedError

    def prepare_sequences_for_inference(
            self,
            sequences: List[str],
            max_length: int,
            context_length: Optional[int] = None,
            **kwargs: Any
    ) -> Tuple[List[data_utils.Sample], List[data_utils.InferenceInfo]]:
        samples, _ = zip(*self.variant.preprocessing_fn(sequences, [None] * len(sequences), True))
        all_samples = []
        all_infos = []
        for sample in samples:
            sample = data_utils.sanitize_sample(sample, self.variant.unk_token_id)
            sequence = str(sample)
            token_lengths = [len(tokens) for tokens in sample.tokens]
            length = sum(token_lengths)
            if length <= max_length:
                all_samples.append(sample)
                all_infos.append(data_utils.InferenceInfo(
                    ctx_start=0,
                    ctx_end=len(sequence),
                    window_start=0,
                    window_end=len(sequence),
                    window_idx=0,
                    length=length
                ))
            else:
                windows = self._split_sample_for_inference(sample, max_length, context_length, **kwargs)
                for i, (ctx_start, ctx_end, window_start, window_end) in enumerate(windows):
                    sample, _ = self.variant.get_sample(sequence[ctx_start:ctx_end], is_inference=True)
                    all_samples.append(sample)
                    all_infos.append(data_utils.InferenceInfo(
                        ctx_start=ctx_start,
                        ctx_end=ctx_end,
                        window_start=window_start,
                        window_end=window_end,
                        window_idx=i,
                        length=sum(len(t) for t in sample.tokens)
                    ))
        return all_samples, all_infos

    def _batch_sequences_for_inference(
            self,
            sequences: List[Union[str, data_utils.Sample]]
    ) -> Batch:
        items = [self.variant.get_inputs(s, is_inference=True) for s in sequences]
        return data_utils.collate(items)
    
    @torch.inference_mode()
    def inference(
            self,
            model: Model,
            inputs: List[Union[str, data_utils.Sample]],
            **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[data_utils.InferenceInfo],
            predictions: List[Any],
            **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    def postprocess_inference_outputs(
            self,
            sequences: List[str],
            infos: List[data_utils.InferenceInfo],
            predictions: List[Any],
            **kwargs: Any
    ) -> List[Any]:
        grouped_predictions: List[List[int]] = []
        grouped_infos: List[List[data_utils.InferenceInfo]] = []
        prev_info = None
        for info, prediction in zip(infos, predictions):
            if info.window_idx == 0:
                grouped_predictions.append([prediction])
                grouped_infos.append([info])
            elif info.window_idx == prev_info.window_idx + 1:
                grouped_predictions[-1].append(prediction)
                grouped_infos[-1].append(info)
            else:
                raise RuntimeError("should not happen")

            prev_info = info

        assert len(sequences) == len(grouped_predictions) == len(grouped_infos)

        merged_predictions = []
        for sequence, predictions, infos in zip(sequences, grouped_predictions, grouped_infos):
            if len(predictions) == 1:
                merged_predictions.append(predictions[0])
            else:
                merged_predictions.append(self._merge_inference_outputs(sequence, infos, predictions, **kwargs))
        return merged_predictions


def get_task(
        checkpoint_dir: str,
        variant_cfg: variants.DatasetVariantConfig,
        seed: int
) -> Task:
    from gnn_lib.tasks.tokenization_repair import TokenizationRepair
    from gnn_lib.tasks.tokenization_repair_plus import TokenizationRepairPlus
    from gnn_lib.tasks.graph_sed_sequence import GraphSEDSequence
    from gnn_lib.tasks.sed_sequence import SEDSequence
    from gnn_lib.tasks.graph_sec_nmt import GraphSECNMT
    from gnn_lib.tasks.sec_nmt import SECNMT
    from gnn_lib.tasks.graph_sec_words_nmt import GraphSECWordsNMT
    from gnn_lib.tasks.sec_words_nmt import SECWordsNMT
    from gnn_lib.tasks.graph_sed_words import GraphSEDWords
    from gnn_lib.tasks.sed_words import SEDWords

    variant_type = DatasetVariants[variant_cfg.type]
    if variant_type == DatasetVariants.SED_SEQUENCE:
        if variant_cfg.data_scheme == "tensor":
            return SEDSequence(variant_cfg, checkpoint_dir, seed)
        else:
            return GraphSEDSequence(variant_cfg, checkpoint_dir, seed)

    elif variant_type == DatasetVariants.SED_WORDS:
        if variant_cfg.data_scheme == "tensor":
            return SEDWords(variant_cfg, checkpoint_dir, seed)
        else:
            return GraphSEDWords(variant_cfg, checkpoint_dir, seed)

    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR:
        return TokenizationRepair(variant_cfg, checkpoint_dir, seed)

    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR_PLUS:
        return TokenizationRepairPlus(variant_cfg, checkpoint_dir, seed)

    elif variant_type == DatasetVariants.SEC_NMT:
        if variant_cfg.data_scheme == "tensor":
            return SECNMT(variant_cfg, checkpoint_dir, seed)
        else:
            return GraphSECNMT(variant_cfg, checkpoint_dir, seed)

    elif variant_type == DatasetVariants.SEC_WORDS_NMT:
        if variant_cfg.data_scheme == "tensor":
            return SECWordsNMT(variant_cfg, checkpoint_dir, seed)
        else:
            return GraphSECWordsNMT(variant_cfg, checkpoint_dir, seed)

    else:
        raise ValueError(f"Could not determine task from variant type {variant_type}")

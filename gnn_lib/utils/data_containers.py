import collections
from typing import Any, Tuple, Dict, Union, List

import dgl
import torch
from matplotlib.figure import Figure
from torch.utils import tensorboard

from gnn_lib.utils import visualisation


class DataContainer:
    def __init__(self,
                 name: str) -> None:
        self.name = name

    def add(self, data: Any) -> None:
        raise NotImplementedError

    @property
    def value(self) -> Any:
        raise NotImplementedError

    def log_to_tensorboard(self,
                           writer: tensorboard.SummaryWriter,
                           step: int) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class ScalarContainer(DataContainer):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.scalar = None

    def add(self, data: Union[float, torch.Tensor]) -> None:
        if isinstance(data, torch.Tensor):
            data = data.item()
        self.scalar = data

    @property
    def value(self) -> float:
        return self.scalar

    def log_to_tensorboard(self,
                           writer: tensorboard.SummaryWriter,
                           step: int) -> None:
        writer.add_scalar(
            tag=self.name,
            scalar_value=self.value,
            global_step=step
        )

    def reset(self) -> None:
        self.scalar = None


class AverageScalarContainer(DataContainer):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.sum = 0
        self.length = 0

    def add(self, data: Union[float, torch.Tensor, List[float]]) -> None:
        if isinstance(data, torch.Tensor):
            data = data.tolist()
        if isinstance(data, list):
            self.sum += sum(data)
            self.length += len(data)
        else:
            self.sum += data
            self.length += 1

    @property
    def value(self) -> float:
        return self.sum / self.length if self.length > 0 else 0

    def log_to_tensorboard(self, writer: tensorboard.SummaryWriter, step: int) -> None:
        writer.add_scalar(
            tag=self.name,
            scalar_value=self.value,
            global_step=step
        )

    def reset(self) -> None:
        self.sum = 0
        self.length = 0


class F1PrecRecContainer(DataContainer):
    def __init__(self, name: str, class_names: Dict[int, str]) -> None:
        super().__init__(name)
        self.class_names = class_names
        self.tp = collections.defaultdict(int)
        self.fp = collections.defaultdict(int)
        self.fn = collections.defaultdict(int)

    def add(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        labels, predictions = data

        for cls, name in self.class_names.items():
            cls_labels = torch.where(labels == cls, 1, 0)
            cls_predictions = torch.where(predictions == cls, 1, 0)

            predicted_pos_indices = cls_predictions == 1
            num_pos = predicted_pos_indices.sum()
            tp = (cls_labels[predicted_pos_indices] == cls_predictions[predicted_pos_indices]).sum()
            fp = num_pos - tp
            self.tp[(cls, name)] += tp
            self.fp[(cls, name)] += fp

            predicted_neg_indices = torch.logical_not(cls_predictions)
            num_neg = predicted_neg_indices.sum()
            tn = (cls_labels[predicted_neg_indices] == cls_predictions[predicted_neg_indices]).sum()
            fn = num_neg - tn
            self.fn[(cls, name)] += fn

    @property
    def value(self) -> Dict[str, Tuple[float, float, float]]:
        f1_prec_rec = {}
        for cls, name in self.class_names.items():
            tp = self.tp[(cls, name)]
            fp = self.fp[(cls, name)]
            fn = self.fn[(cls, name)]
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = (2 * prec * rec) / (prec + rec)
            f1_prec_rec[name] = (f1, prec, rec)
        return f1_prec_rec

    def log_to_tensorboard(self, writer: tensorboard.SummaryWriter, step: int) -> None:
        for name, (f1, prec, rec) in self.value.items():
            writer.add_scalar(
                tag=f"{self.name}_{name}_f1",
                scalar_value=f1,
                global_step=step
            )
            writer.add_scalar(
                tag=f"{self.name}_{name}_precision",
                scalar_value=prec,
                global_step=step
            )
            writer.add_scalar(
                tag=f"{self.name}_{name}_recall",
                scalar_value=rec,
                global_step=step
            )

    def reset(self) -> None:
        self.tp.clear()
        self.fp.clear()
        self.fn.clear()


class HistogramContainer(DataContainer):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.values = []

    def add(self, data: Union[float, int, torch.Tensor, list]) -> None:
        if isinstance(data, torch.Tensor):
            data = data.tolist()
        if isinstance(data, list):
            self.values.extend(data)
        else:
            self.values.append(data)

    @property
    def value(self) -> List:
        return self.values

    def log_to_tensorboard(self, writer: tensorboard.SummaryWriter, step: int) -> None:
        writer.add_histogram(
            tag=self.name,
            values=torch.tensor(self.value),
            global_step=step
        )

    def reset(self) -> None:
        self.values = []


class MultiTextContainer(DataContainer):
    def __init__(self, name: str, max_samples: int) -> None:
        super().__init__(name)
        self.max_samples = max_samples
        self.samples = []

    def add(self, data: str) -> None:
        if len(self.samples) >= self.max_samples:
            return
        self.samples.append(data)

    @property
    def value(self) -> List[str]:
        return self.samples

    def log_to_tensorboard(self, writer: tensorboard.SummaryWriter, step: int) -> None:
        writer.add_text(
            tag=self.name,
            text_string="\n\n".join(self.samples),
            global_step=step
        )

    def reset(self) -> None:
        self.samples.clear()


class GraphContainer(DataContainer):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.graph = None

    def add(self, data: dgl.DGLHeteroGraph) -> None:
        self.graph = data

    @property
    def value(self) -> Figure:
        return visualisation.visualise_graph_matplotlib(self.graph)

    def log_to_tensorboard(self, writer: tensorboard.SummaryWriter, step: int) -> None:
        writer.add_figure(
            tag=self.name,
            figure=self.value,
            global_step=step
        )

    def reset(self) -> None:
        self.graph = None


class HyperparameterContainer(DataContainer):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.hparams = None
        self.metrics = None

    def add(self, data: Tuple[Dict[str, Any], Dict[str, float]]) -> None:
        self.hparams = data[0]
        self.metrics = data[1]

    @property
    def value(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        return self.hparams, self.metrics

    def log_to_tensorboard(self, writer: tensorboard.SummaryWriter, step: int) -> None:
        hparam, metric = self.value
        writer.add_hparams(
            run_name=self.name,
            hparam_dict=hparam,
            metric_dict=metric
        )

    def reset(self) -> None:
        self.hparams = None
        self.metrics = None

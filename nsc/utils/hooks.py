import copy
from collections import defaultdict
from typing import Any, Dict, List, Type, Union, Optional

from torch import nn


class SaveHook:
    def __init__(self) -> None:
        self.saved: Optional[Any] = None

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def clear(self) -> None:
        self.saved = None


class SaveInputHook(SaveHook):
    def __call__(self, module: nn.Module, inputs: Any, outputs: Any) -> None:
        self.saved = inputs


class SaveOutputHook(SaveHook):
    def __call__(self, module: nn.Module, inputs: Any, outputs: Any) -> None:
        self.saved = outputs


class ModelHook:
    def __init__(self) -> None:
        self.attached_hooks: Dict[str, Dict[str, SaveOutputHook]] = defaultdict(dict)

    def clear(self, name: str = None) -> None:
        for k, v in self.attached_hooks.items():
            if name is not None and k != name:
                continue
            for hook in v.values():
                hook.clear()

    def attach(
            self,
            name: str,
            module: nn.Module,
            hook: Union[SaveOutputHook, SaveInputHook],
            attach_to_cls: Optional[Type[nn.Module]] = None,
            layer_name: str = None,
            pre_hook: bool = False
    ) -> None:
        if attach_to_cls is None:
            self.attached_hooks[name][name] = hook
            if pre_hook:
                module.register_forward_pre_hook(hook)
            else:
                module.register_forward_hook(hook)
            return
        # if attach_to_cls is given, attach the hook to all submodules of model that have type attach_to_cls
        for module_name, module in module.named_modules():
            if isinstance(module, attach_to_cls):
                if layer_name is not None:
                    if layer_name != module_name:
                        continue
                hook_cp = copy.deepcopy(hook)
                self.attached_hooks[name][module_name] = hook_cp
                if pre_hook:
                    module.register_forward_pre_hook(hook_cp)
                else:
                    module.register_forward_hook(hook_cp)

    def __getitem__(self, name: str) -> Dict[str, Any]:
        return {n: v.saved for n, v in self.attached_hooks[name].items() if v.saved is not None}

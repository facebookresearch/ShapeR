# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import functools
import inspect
from typing import List

import numpy as np
import torch

from torch.utils.data._utils.collate import (
    collate,
    collate_tensor_fn,
    default_collate_fn_map,
)


def smart_cat(inp_arr, dim=-1):
    devices = set()
    for i, inp in enumerate(inp_arr):
        if isinstance(inp, TensorWrapper):
            inp_arr[i] = inp._data
        else:
            inp_arr[i] = inp
        devices.add(inp_arr[i].device)
    if len(devices) > 1:
        raise RuntimeError(f"More than one device found! {devices}")
    return torch.cat(inp_arr, dim=dim)


def smart_stack(inp_arr, dim: int = 0):
    devices = set()
    for i, inp in enumerate(inp_arr):
        if isinstance(inp, TensorWrapper):
            inp_arr[i] = inp._data
        else:
            inp_arr[i] = inp
        devices.add(inp_arr[i].device)
    if len(devices) > 1:
        raise RuntimeError(f"More than one device found! {devices}")
    return torch.stack(inp_arr, dim=dim)


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_nonempty_arg_names(func):
    spec = inspect.getfullargspec(func)
    signature = inspect.signature(func)
    return [
        k
        for k in spec.args
        if signature.parameters[k].default is not inspect.Parameter.empty
    ]


def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
    if they are numpy arrays. Use the device and dtype of the wrapper.
    """

    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device("cpu")
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)

        return func(self, *cast_args)

    return wrap


def autoinit(func):
    """
    Helps with initialization. Will auto-reshape and auto-expand input arguments
    to match the first argument, as well as check shapes based on default tensor sizes.
    """

    @functools.wraps(func)
    def wrap(self, *args, **kwargs):
        # Combine args and kwargs.
        arg_names = get_nonempty_arg_names(func)
        all_args = {}
        for i, arg in enumerate(args):
            all_args[arg_names[i]] = arg
        for arg_name in kwargs:
            all_args[arg_name] = kwargs[arg_name]

        # Add default values to all_args if unspecified inputs.
        default_args = get_default_args(func)
        unperturbed_args = {}
        for arg_name in default_args:
            default_arg = default_args[arg_name]
            if default_arg is None:
                pass
            elif not isinstance(default_arg, (TensorWrapper, torch.Tensor)):
                # If not TW or torch tensor, pass it through unperturbed.
                unperturbed_args[arg_name] = all_args.pop(arg_name)
            else:
                if arg_name not in all_args or all_args[arg_name] is None:
                    all_args[arg_name] = default_arg

        # Auto convert numpy,lists,floats to torch, check that shapes are good.
        for arg_name in all_args:
            arg = all_args[arg_name]
            if isinstance(arg, (torch.Tensor, TensorWrapper)):
                pass
            elif isinstance(arg, (int, float)):
                arg = torch.tensor(arg).reshape(1)
            elif isinstance(arg, List):
                arg = torch.tensor(arg)
            elif isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
            else:
                raise ValueError("Unsupported initialization type")
            assert isinstance(arg, (torch.Tensor, TensorWrapper))

            default_arg = default_args[arg_name]
            if default_arg is None:
                # Do not check dims if default arg is None.
                all_args[arg_name] = arg
                default_dims = self._valid_dims
            elif isinstance(default_arg, TensorWrapper):
                # Convert list of torch.Size to tuple of ints.
                default_dims = tuple([da[0] for da in default_arg.shape])
            else:
                default_dims = (default_arg.shape[-1],)
            if arg.shape[-1] not in default_dims:
                # probably need a more general solution here to handle single dim inputs.
                if default_dims[0] == 1:
                    arg = arg.unsqueeze(-1)
                if arg.shape[-1] not in default_dims:
                    raise ValueError(
                        f"Bad shape of {arg.shape[-1]} for {arg_name}, should be in {default_dims}"
                    )

            all_args[arg_name] = arg

        if len(all_args) > 0:
            # Shape of all inputs is determined by first arg.
            first_arg_name = arg_names[0]
            batch_shape = all_args[first_arg_name].shape[:-1]

            has_cuda_tensor = False

            for arg_name in all_args:
                arg = all_args[arg_name]
                # Try to trim any extra dimensions at the beginning of arg shape.
                while True:
                    if (
                        arg.ndim > len(batch_shape)
                        and arg.shape[0] == 1
                        and arg.ndim > 1
                    ):
                        arg = arg.squeeze(0)
                    else:
                        break
                arg = arg.expand(*batch_shape, arg.shape[-1])
                all_args[arg_name] = arg

                if (
                    isinstance(all_args[arg_name], (torch.Tensor, TensorWrapper))
                    and all_args[arg_name].is_cuda
                ):
                    has_cuda_tensor = True

            if has_cuda_tensor:
                for arg_name in all_args:
                    if (
                        isinstance(all_args[arg_name], (torch.Tensor, TensorWrapper))
                        and not all_args[arg_name].is_cuda
                    ):
                        all_args[arg_name] = all_args[arg_name].cuda()

        # Add the unperturbed args back to all args.
        all_args.update(unperturbed_args)

        return func(self, **all_args)

    return wrap


def tensor_wrapper_collate(batch, *, collate_fn_map=None):
    """Simply call stack for TensorWrapper"""
    return torch.stack(batch, 0)


def float_collate(batch, *, collate_fn_map=None):
    """Auto convert float to float32"""
    return torch.tensor(batch, dtype=torch.float32)


def list_dict_collate(batch, *, collate_fn_map=None):
    """collate lists; handles the case where the lists in the batch are
    expressing a dict via List[Tuble[key, value]] and returns a Dict[key, value]
    in that case."""
    if len(batch) > 0:
        list_0 = batch[0]
        if len(list_0) > 0:
            elem_0 = list_0[0]
            if isinstance(elem_0, tuple) and len(elem_0) == 2:
                # the lists in each batch sample are (key, value) pairs and we hence return a dictionary
                for i in range(len(batch)):
                    batch[i] = {k: v for k, v in batch[i]}
    return batch


def tensor_wrapper_collate_cat(batch, *, collate_fn_map=None):
    """Simply call cat for TensorWrapper"""
    return torch.cat(batch, 0)


def tensor_collate_cat(batch, *, collate_fn_map=None):
    """identical to "collate_tensor_fn" but replace torch.stack with torch.cat"""
    elem = batch[0]
    # NOTE(dd): skip this assert for now, sequence_id is sometimes shaped [B,1]
    # assert elem.shape[0] == 1
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        # Note(nyn): pytorch 1.12 doesn't have the _typed_storage() interface. Need to use storage() instead.
        # storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        storage = elem.storage()._new_shared(numel, device=elem.device)

        # since we are using torch.cat, we don't need to add a new dimension here
        dims_from_one = list(elem.size())[1:]
        out = elem.new(storage).resize_(len(batch), *dims_from_one)
    return torch.cat(batch, 0, out=out)  # concatenate instead of stack


def custom_collate_fn(batch):
    # Get the common keys between samples. This is required when we train with
    # multiple datasets with samples having different keys.
    if isinstance(batch, list) and isinstance(batch[0], dict):
        common_keys = set(batch[0].keys())

        for sample in batch[1:]:
            common_keys &= set(sample.keys())

        # update the batch with new samples with only the common keys
        new_batch = []
        for sample in batch:
            new_sample = {k: v for k, v in sample.items() if k in common_keys}
            new_batch.append(new_sample)
        batch = new_batch

    """Custom collate function for tensor wrapper"""
    default_collate_fn_map[TensorWrapper] = tensor_wrapper_collate
    default_collate_fn_map[float] = float_collate
    default_collate_fn_map[list] = list_dict_collate
    default_collate_fn_map[torch.Tensor] = collate_tensor_fn
    if "already_collated" in batch[0]:
        # Use torch.cat instead of torch.stack
        default_collate_fn_map[torch.Tensor] = tensor_collate_cat
        default_collate_fn_map[TensorWrapper] = tensor_wrapper_collate_cat
    batch = collate(batch, collate_fn_map=default_collate_fn_map)
    return batch


class TensorWrapper:
    """Base class for making "smart" tensor objects that behave like pytorch tensors
    Inpired by Paul-Edouard Sarlin's code here in pixloc:
    https://github.com/cvg/pixloc/blob/master/pixloc/pixlib/geometry/wrappers.py
    Adopted and modified by Daniel DeTone.
    """

    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    def dim(self):
        return self._data.dim()

    def nelement(self):
        return self._data.nelement()

    def numel(self):
        return self._data.numel()

    @property
    def collate_fn(self):
        return custom_collate_fn

    @property
    def is_cuda(self):
        return self._data.is_cuda

    @property
    def is_contiguous(self):
        return self._data.is_contiguous

    @property
    def requires_grad(self):
        return self._data.requires_grad

    @property
    def grad(self):
        return self._data.grad

    @property
    def grad_fn(self):
        return self._data.grad_fn

    def requires_grad_(self, requires_grad: bool = True):
        self._data.requires_grad_(requires_grad)

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        if isinstance(item, TensorWrapper):
            item = item._data
        self._data[index] = item

    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))

    def reshape(self, *args, **kwargs):
        return self.__class__(self._data.reshape(*args, **kwargs))

    def repeat(self, *args, **kwargs):
        return self.__class__(self._data.repeat(*args, **kwargs))

    def expand(self, *args, **kwargs):
        return self.__class__(self._data.expand(*args, **kwargs))

    def clone(self):
        return self.__class__(self._data.clone())

    def cpu(self):
        return self.__class__(self._data.cpu())

    def cuda(self, gpu_id=0):
        return self.__class__(self._data.cuda(gpu_id))

    def contiguous(self):
        return self.__class__(self._data.contiguous())

    def pin_memory(self):
        return self.__class__(self._data.pin_memory())

    def float(self):
        return self.__class__(self._data.float())

    def double(self):
        return self.__class__(self._data.double())

    def detach(self):
        return self.__class__(self._data.detach())

    def numpy(self):
        return self._data.numpy()

    def tensor(self):
        return self._data

    def tolist(self):
        return self._data.tolist()

    def squeeze(self, dim=None):
        assert dim != -1 and dim != self._data.dim() - 1
        if dim is None:
            return self.__class__(self._data.squeeze())
        return self.__class__(self._data.squeeze(dim=dim))

    def unsqueeze(self, dim=None):
        assert dim != -1 and dim != self._data.dim()
        return self.__class__(self._data.unsqueeze(dim=dim))

    def view(self, *shape):
        assert shape[-1] == -1 or shape[-1] == self._data.shape[-1]
        return self.__class__(self._data.view(*shape))

    def __len__(self):
        return self._data.shape[0]

    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def cat(cls, objects: List, dim=0, *, out=None):
        data = torch.cat([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def allclose(
        cls,
        input: torch.Tensor,
        other: torch.Tensor,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    ):
        return torch.allclose(
            # pyre-fixme[16]: `Tensor` has no attribute `_data`.
            input._data,
            other._data,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    @classmethod
    def take_along_dim(cls, obj, indices, dim, *, out=None):
        data = torch.take_along_dim(obj._data, indices, dim, out=out)
        return cls(data)

    @classmethod
    def flatten(cls, obj, start_dim=0, end_dim=-1):
        data = torch.flatten(obj._data, start_dim=start_dim, end_dim=end_dim)
        return cls(data)

    @classmethod
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return self.stack(*args, **kwargs)
        elif func is torch.cat:
            return self.cat(*args, **kwargs)
        elif func is torch.allclose:
            return self.allclose(*args, **kwargs)
        elif func is torch.take_along_dim:
            return self.take_along_dim(*args, **kwargs)
        elif func is torch.flatten:
            return self.flatten(*args, **kwargs)
        else:
            return NotImplemented

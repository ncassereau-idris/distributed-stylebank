# /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import pyarrow as pa
from pyarrow import plasma
import torch.distributed as dist
import functools
import subprocess
from . import tools


class PlasmaServer:

    def __init__(self):
        self._server = None
        self._client = None

    def connect(self):
        if tools.rank == 0:
            GB200 = 200 * (1024 ** 3)
            self._server = subprocess.Popen([
                "plasma_store", "-m", str(GB200), "-s", "/tmp/plasma"
            ])

    @property
    def client(self):
        if self._client is None:
            self._client = plasma.connect("/tmp/plasma", num_retries=200)
        return self._client

    def kill(self):
        if self._client is not None:
            self._client.disconnect()
        if self._server is not None:
            assert tools.rank == 0
            self._server.kill()


plasma_server = PlasmaServer()


class AbstractStorage:

    def __init__(self, autocuda=False):
        self._storage = dict()
        self.autocuda = autocuda

    def _maybe_cuda(self, value):
        if self.autocuda and isinstance(value, torch.Tensor):
            return value.cuda()
        else:
            return value

    def _retrieve(self, idx):
        raise NotImplementedError()

    def _register(self, idx, value):
        raise NotImplementedError()

    def __getitem__(self, ids):
        if isinstance(ids, int):
            ret = self._retrieve(ids)
        else:
            ret = torch.stack([
                self._retrieve(idx) for idx in ids
            ])
        return self._maybe_cuda(ret)

    def __setitem__(self, ids, value):
        if getattr(value, "is_cuda", False):
            value = value.cpu()

        if isinstance(ids, (list, tuple, torch.Tensor)):
            chunks = torch.unbind(value, dim=0)
            for i in range(len(ids)):
                self._register(ids[i], chunks[i])
        else:
            self._register(ids, value)

    def merge(self):
        storages_list = [None] * tools.size
        dist.all_gather_object(storages_list, self._storage)
        self._storage = functools.reduce(dict.__or__, storages_list)
        return self

    def __len__(self):
        return len(self._storage.keys())


class Storage(AbstractStorage):

    def _retrieve(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self._storage.get(idx, None)

    def _register(self, idx, value):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        self._storage[idx] = value


class PlasmaStorage(AbstractStorage):

    def __init__(self, autocuda=False):
        self._storage = dict()
        self.autocuda = autocuda
        self.gen = np.random.RandomState(None)

    @property
    def client(self):
        return plasma_server.client

    def _retrieve(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        object_id = self._storage.get(idx, None)
        if object_id is None:
            return None

        [buf] = self.client.get_buffers([object_id])
        reader = pa.BufferReader(buf)
        tensor = pa.ipc.read_tensor(reader)
        return torch.from_numpy(np.array(tensor.to_numpy()))

    def _register(self, idx, value):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        tensor = pa.Tensor.from_numpy(value.numpy())
        object_id = pa.plasma.ObjectID(self.gen.bytes(20))
        data_size = pa.ipc.get_tensor_size(tensor)
        buf = self.client.create(object_id, data_size)
        stream = pa.FixedSizeBufferWriter(buf)
        pa.ipc.write_tensor(tensor, stream)
        self.client.seal(object_id)
        self._storage[idx] = object_id

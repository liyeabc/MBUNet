# encoding: utf-8
"""
credit:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
"""

import logging
import numpy as np
import time
import weakref
import torch
import fastreid.utils.comm as comm
from fastreid.utils.events import EventStorage
import pdb
__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]


class HookBase:

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:

    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):

        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)

            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        self.storage.step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):

    def __init__(self, model, data_loader, optimizer):

        super().__init__()

        model.train()

        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode![SimpleTrainer]"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = self.data_loader.next()
        data_time = time.perf_counter() - start
        """
        If your want to do something with the heads, you can wrap the model.
        """

        outputs = self.model(data)
        # Compute loss
        loss_dict = self.model.module.losses(outputs)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()


    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite无限 or NaN非数（not a number） at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        all_metrics_dict = comm.gather(metrics_dict)

        # if comm.is_main_process():
        if "data_time" in all_metrics_dict[0]:
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            self.storage.put_scalar("data_time", data_time)

        # average the rest metrics
        metrics_dict = {
            k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())

        self.storage.put_scalar("total_loss", total_losses_reduced)
        if len(metrics_dict) > 1:
            self.storage.put_scalars(**metrics_dict)


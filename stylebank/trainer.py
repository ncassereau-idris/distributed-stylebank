# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import logging
import time
from . import tools
from .dataclasses import TrainingData


log = logging.getLogger(__name__)


class Trainer:

    def __init__(self, cfg, data_manager, network_manager):
        self.cfg = cfg
        assert cfg.training.train
        self.data_manager = data_manager
        self.network_manager = network_manager

        self.optimizer = optim.Adam(self.network_manager.model.parameters())

        self.effective_batch_size = cfg.training.batch_size * tools.size
        self.training_data = TrainingData()
        if self.cfg.vgg_layers.store:
            log.info("Preloading styles and contents")
            self.network_manager.loss_network.preload(
                *self.data_manager.make_preload_dataloaders()
            )

    def adjust_learning_rate(self, step):
        lr = self.cfg.training.learning_rate * tools.size
        lr = max(lr * (0.8 ** (step)), 1e-6)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def synchronise_data(self):
        self.training_data = self.training_data.merge()

    def log(self, epoch, step, steps_per_epoch):
        self.synchronise_data()
        duration_epoch = self.current_time - self.epoch_beginning
        duration_training = self.current_time - self.train_beginning
        log.info(
            f"Epoch {epoch} | " +
            f"Step {step} / {steps_per_epoch} | " +
            self.training_data.log() +
            f" | Batch size: {self.effective_batch_size}" +
            f" | Wall (epoch): {tools.format_duration(duration_epoch)}" +
            f" | Wall (training): {tools.format_duration(duration_training)}"
        )

    def log_epoch(self, epoch):
        self.synchronise_data()
        duration_epoch = self.current_time - self.epoch_beginning
        duration_training = self.current_time - self.train_beginning
        log.info(
            f"Epoch {epoch} / {self.cfg.training.epochs} | " +
            self.training_data.log_epoch() +
            f" | Batch size: {self.effective_batch_size}" +
            f" | Wall (epoch): {tools.format_duration(duration_epoch)}" +
            f" | Wall (training): {tools.format_duration(duration_training)}"
        )

    def train(self):
        dataloader = self.data_manager.training_dataloader
        step = 0
        self.adjust_learning_rate(step)
        T = self.cfg.training.consecutive_style_step + 1

        self.train_beginning = time.perf_counter()

        log.info("Beginning training")
        for epoch in range(1, self.cfg.training.epochs + 1):
            log.info(f"Epoch {epoch} / {self.cfg.training.epochs}")
            self.epoch_beginning = time.perf_counter()
            local_step = 0
            for (content_id, content), (style_id, style) in dataloader:
                step += 1
                local_step += 1

                content = content.cuda()
                style = style.cuda()

                if step % T != 0:
                    self._train_style_bank(content_id, content, style_id, style)
                else:
                    self._train_auto_encoder(content)

                self.current_time = time.perf_counter()

                if local_step % self.cfg.training.log_interval == 0:
                    self.log(epoch, local_step, len(dataloader))
                    self.training_data.reset()

                if step % self.cfg.training.save_interval == 0:
                    self.network_manager.save_models()

                if step % self.cfg.training.adjust_learning_rate_interval == 0:
                    lr_step = (
                        step /
                        self.cfg.training.adjust_learning_rate_interval
                    )
                    new_lr = self.adjust_learning_rate(lr_step)
                    log.info(f"Learning rate decay: {new_lr:.6f}")

            self.log_epoch(epoch)
            self.training_data.reset(reset_epoch_average=True)

        total_duration = self.current_time - self.train_beginning
        log.info(
            "End of training (Total duration: "
            f"{tools.format_duration(total_duration)})"
        )

    def _train_style_bank(self, content_id, content, style_id, style):
        self.optimizer.zero_grad()
        output_image = self.network_manager.model(content, style_id)
        content_loss, style_loss, tv_loss = self.network_manager.loss_network(
            output_image, content, style, content_id, style_id
        )
        total_loss = content_loss + style_loss + tv_loss
        total_loss.backward()
        self.optimizer.step()
        self.training_data.update(
            total_loss=total_loss,
            content_loss=content_loss,
            style_loss=style_loss,
            regularizer_loss=tv_loss
        )

    def _train_auto_encoder(self, content):
        self.optimizer.zero_grad()
        output_image = self.network_manager.model(content)
        loss = self.network_manager.loss_network(output_image, content)
        loss.backward()
        self.optimizer.step()
        self.training_data.update(reconstruction_loss=loss)

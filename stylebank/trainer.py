# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from dataclasses import dataclass
import logging
import time
from datetime import timedelta


log = logging.getLogger(__name__)


class MovingAverage:

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = 0.
        self.n = 0

    def update(self, new_data):
        self.data = self.n * self.data + new_data
        self.n += 1
        self.data /= self.n
        return self.data

    def __str__(self):
        return f"{self.data:.6f}"


@dataclass
class TrainingData:

    style_loss: float = 0.
    content_loss: float = 0.
    total_loss: float = 0.
    reconstruction_loss: float = 0.
    regularizer_loss: float = 0.

    epoch_style_loss = MovingAverage()
    epoch_content_loss = MovingAverage()
    epoch_total_loss = MovingAverage()
    epoch_reconstruction_loss = MovingAverage()
    epoch_regularizer_loss = MovingAverage()

    def reset(self, reset_epoch_average=False):
        if reset_epoch_average:
            self.epoch_style_loss.reset()
            self.epoch_content_loss.reset()
            self.epoch_total_loss.reset()
            self.epoch_reconstruction_loss.reset()
            self.epoch_regularizer_loss.reset()
        else:
            self.epoch_style_loss.update(self.style_loss)
            self.epoch_content_loss.update(self.content_loss)
            self.epoch_total_loss.update(self.total_loss)
            self.epoch_reconstruction_loss.update(self.reconstruction_loss)
            self.epoch_regularizer_loss.update(self.regularizer_loss)

        self.style_loss = 0.
        self.content_loss = 0.
        self.total_loss = 0.
        self.reconstruction_loss = 0.
        self.regularizer_loss = 0.


    def update(
        self,
        style_loss=None,
        content_loss=None,
        total_loss=None,
        reconstruction_loss=None,
        regularizer_loss=None
    ):
        if style_loss is not None:
            self.style_loss += style_loss.item()
        if content_loss is not None:
            self.content_loss += content_loss.item()
        if total_loss is not None:
            self.total_loss += total_loss.item()
        if reconstruction_loss is not None:
            self.reconstruction_loss += reconstruction_loss.item()
        if regularizer_loss is not None:
            self.regularizer_loss += regularizer_loss.item()

    def log(self):
        losses = [
            f"Total loss: {self.total_loss:.6f}",
            f"Content loss: {self.content_loss:.6f}",
            f"Style loss: {self.style_loss:.6f}",
            f"Regularizer loss: {self.regularizer_loss:.6f}",
            f"Reconstruction loss: {self.reconstruction_loss:.6f}",
        ]
        return " | ".join(losses)

    def log_epoch(self):
        losses = [
            f"Total loss: {str(self.epoch_total_loss)}",
            f"Content loss: {str(self.epoch_content_loss)}",
            f"Style loss: {str(self.epoch_style_loss)}",
            f"Regularizer loss: {str(self.epoch_regularizer_loss)}",
            f"Reconstruction loss: {str(self.epoch_reconstruction_loss)}",
        ]
        return " | ".join(losses)


class Trainer:

    def __init__(self, cfg, data_manager, network_manager):
        self.cfg = cfg
        assert cfg.training.train
        self.data_manager = data_manager
        self.network_manager = network_manager

        self.optimizer = optim.Adam(self.network_manager.model.parameters())
        self.optimizer_ae = optim.Adam([
            {'params': self.network_manager.model.encoder_net.parameters()},
            {'params': self.network_manager.model.decoder_net.parameters()}
        ], lr=cfg.training.learning_rate)

        self.training_data = TrainingData()

    def adjust_learning_rate(self, optimizer, step):
        lr = self.cfg.training.learning_rate
        lr = max(lr * (0.8 ** (step)), 1e-6)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def format_duration(self, seconds):
        return str(timedelta(seconds=int(seconds)))

    def log(self, epoch, step, steps_per_epoch):
        duration_epoch = self.current_time - self.epoch_beginning
        duration_training = self.current_time - self.train_beginning
        log.info(
            f"Epoch {epoch} | Step {step % steps_per_epoch} / {steps_per_epoch} | " +
            self.training_data.log() +
            f" | Wall (epoch): {self.format_duration(duration_epoch)}" +
            f" | Wall (training): {self.format_duration(duration_training)}"
        )

    def log_epoch(self, epoch):
        duration_epoch = self.current_time - self.epoch_beginning
        duration_training = self.current_time - self.train_beginning
        log.info(
            f"Epoch {epoch} / {self.cfg.training.epochs} | " +
            self.training_data.log_epoch() +
            f" | Wall (epoch): {self.format_duration(duration_epoch)}" +
            f" | Wall (training): {self.format_duration(duration_training)}"
        )

    def train(self):
        dataloader = self.data_manager.training_dataloader
        step = 0
        T = self.cfg.training.consecutive_style_step +1

        self.train_beginning = time.perf_counter()

        log.info("Beginning training")
        for epoch in range(1, self.cfg.training.epochs + 1):
            log.info(f"Epoch {epoch} / {self.cfg.training.epochs}")
            self.epoch_beginning = time.perf_counter()
            for content, (style_id, style) in dataloader:
                step += 1

                batch_size = content.shape[0]
                assert style.shape[0] == batch_size and style_id.shape[0] == batch_size

                content = content.cuda()
                style = style.cuda()

                if step % T != 0:
                    self._train_style_bank(content, style_id, style)
                else:
                    self._train_auto_encoder(content)

                self.current_time = time.perf_counter()

                if step % self.cfg.training.log_interval == 0:
                    self.log(epoch, step, len(dataloader))
                    self.training_data.reset()

                if step % self.cfg.training.save_interval == 0:
                    self.network_manager.save_models()

                if step % self.cfg.training.adjust_learning_rate_interval == 0:
                    lr_step = step / self.cfg.training.adjust_learning_rate_interval
                    self.adjust_learning_rate(self.optimizer, lr_step)
                    new_lr = self.adjust_learning_rate(self.optimizer_ae, lr_step)
                    log.info(f"Learning rate decay: {new_lr:.6f}")

            self.log_epoch(epoch)
            self.training_data.reset(reset_epoch_average=True)

        log.info(
            "End of training (Total duration: "
            f"{self.format_duration(self.current_time - self.train_beginning)})"
        )

    def _train_style_bank(self, content, style_id, style):
        self.optimizer.zero_grad()
        output_image = self.network_manager.model(content, style_id)

        content_loss, style_loss = self.network_manager.loss_network(output_image, content, style)
        content_loss *= self.cfg.training.content_weight
        style_loss *= self.cfg.training.style_weight

        diff_i = torch.sum(torch.abs(output_image[:, :, :, 1:] - output_image[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(output_image[:, :, 1:, :] - output_image[:, :, :-1, :]))
        tv_loss = self.cfg.training.reg_weight * (diff_i + diff_j)

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
        self.optimizer_ae.zero_grad()
        output_image = self.network_manager.model(content)
        loss = self.network_manager.loss_network(output_image, content)
        loss.backward()
        self.optimizer_ae.step()
        self.training_data.update(reconstruction_loss=loss)

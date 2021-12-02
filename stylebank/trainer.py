# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import logging
import time
from . import tools
from .dataclasses import TrainingData


log = logging.getLogger(__name__)


class MultiOptimizer:

    def __init__(self, cfg, network_manager):
        self.cfg = cfg
        self.network_manager = network_manager

        self.encoder_ae_optim = self._make_optimizer(
            self.network_manager.model.module.encoder_net
        )
        self.encoder_sb_optim = self._make_optimizer(
            self.network_manager.model.module.encoder_net
        )
        self.decoder_ae_optim = self._make_optimizer(
            self.network_manager.model.module.decoder_net
        )
        self.decoder_sb_optim = self._make_optimizer(
            self.network_manager.model.module.decoder_net
        )
        self.stylebank_optim = [
            self._make_optimizer(
                self.network_manager.model.module.style_bank[i]
            ) for i in range(len(
                self.network_manager.model.module.style_bank
            ))
        ]
        self.all_optim = [
            self.encoder_ae_optim, self.encoder_sb_optim,
            self.decoder_ae_optim, self.decoder_sb_optim
        ] + self.stylebank_optim
        self.scaler = GradScaler()

    def _make_optimizer(self, model):
        return optim.Adam(
            model.parameters(),
            lr=self.cfg.training.learning_rate
        )

    def adjust_learning_rate(self, step):
        lr_step = step / self.cfg.training.adjust_learning_rate_interval
        lr = self.cfg.training.learning_rate * tools.size
        lr = max(lr * (0.8 ** (lr_step)), 1e-6)
        for optimizer in self.all_optim:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        return lr

    def zero_grad(self):
        for optimizer in self.all_optim:
            optimizer.zero_grad()

    def _step_ae(self):
        self.scaler.step(self.encoder_ae_optim)
        self.scaler.step(self.decoder_ae_optim)
        # self.encoder_ae_optim.step()
        # self.decoder_ae_optim.step()

    def _step_sb(self, style_id):
        self.scaler.step(self.encoder_sb_optim)
        self.scaler.step(self.decoder_sb_optim)
        # self.encoder_sb_optim.step()
        # self.decoder_sb_optim.step()
        for idx in style_id:
            # self.stylebank_optim[idx].step()
            self.scaler.step(self.stylebank_optim[idx])

    def step(self, style_id=None):
        if style_id is None:
            self._step_ae()
        else:
            self._step_sb(style_id)
        self.scaler.update()


class Trainer:

    def __init__(self, cfg, data_manager, network_manager):
        self.cfg = cfg
        assert cfg.training.train
        self.data_manager = data_manager
        self.network_manager = network_manager

        self.optimizer = MultiOptimizer(cfg=cfg, network_manager=network_manager)

        self.effective_batch_size = cfg.training.batch_size * tools.size
        self.training_data = TrainingData()
        if self.cfg.vgg_layers.store:
            log.info("Preloading styles and contents")
            self.network_manager.loss_network.preload(
                *self.data_manager.make_preload_dataloaders()
            )

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
        dataloader = self.data_manager.make_training_dataloader()
        step = 0
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

                content_id = [idx.item() for idx in content_id]
                style_id = [idx.item() for idx in style_id]

                if step % T != 0:
                    self._train_style_bank(
                        content_id, content, style_id, style
                    )
                else:
                    self._train_auto_encoder(content)

                self.current_time = time.perf_counter()

                if local_step % self.cfg.training.log_interval == 0:
                    self.log(epoch, local_step, len(dataloader))
                    self.training_data.reset()

                if step % self.cfg.training.adjust_learning_rate_interval == 0:
                    new_lr = self.optimizer.adjust_learning_rate(step)
                    log.info(f"Learning rate decay: {new_lr:.6f}")

            self.log_epoch(epoch)
            self.network_manager.save_models(epoch, self.training_data)
            self.training_data.reset(reset_epoch_average=True)

        total_duration = self.current_time - self.train_beginning
        log.info(
            "End of training (Total duration: "
            f"{tools.format_duration(total_duration)})"
        )

    def _train_style_bank(self, content_id, content, style_id, style):
        self.optimizer.zero_grad()
        with autocast():
            output_image = self.network_manager.model(content, style_id)
            content_loss, style_loss, tv_loss = self.network_manager.loss_network(
                output_image, content, style, content_id, style_id
            )
            total_loss = content_loss + style_loss + tv_loss
        self.optimizer.scaler.scale(total_loss).backward()
        # total_loss.backward()
        self.optimizer.step(style_id=style_id)
        self.training_data.update(
            total_loss=total_loss,
            content_loss=content_loss,
            style_loss=style_loss,
            regularizer_loss=tv_loss
        )

    def _train_auto_encoder(self, content):
        self.optimizer.zero_grad()
        with autocast():
            output_image = self.network_manager.model(content)
            loss = self.network_manager.loss_network(output_image, content)
        self.optimizer.scaler.scale(loss).backward()
        # loss.backward()
        self.optimizer.step(style_id=None)
        self.training_data.update(reconstruction_loss=loss)

from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, L1Loss
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.text import Perplexity
from .components.model import Instructor
import soundfile as sf
from transformers import get_cosine_schedule_with_warmup
import wandb
import uuid

class InstructMusicGenAdapterLitModule(LightningModule):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            tmp_dir: str,
            compile: bool,
            instructor: Instructor,
            audio_regularization: float,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = self.hparams.instructor()

        self.criterion_1 = CrossEntropyLoss()
        self.criterion_2 = L1Loss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Perplexity(ignore_index=-100)
        self.val_acc = Perplexity(ignore_index=-100)
        self.test_acc = Perplexity(ignore_index=-100)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MinMetric()

        self.generation_sample = []

        self.uuid = str(uuid.uuid4())

        # self.model.musicgen.compression_model.train()
        # for param in self.model.musicgen.compression_model.parameters():
        #     param.requires_grad = False


    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        input_stems_mix, output_stems_mix, instruction_text = batch


        # if isinstance(input_stems_mix, torch.Tensor):
        #     input_stems_mix_np = input_stems_mix.detach().cpu().numpy()
        #     input_stems_mix = [input_stems_mix_np[i, :] for i in range(input_stems_mix_np.shape[0])]
        # if isinstance(output_stems_mix, torch.Tensor):
        #     output_stems_mix_np = output_stems_mix.detach().cpu().numpy()
        #     output_stems_mix = [output_stems_mix_np[i, :] for i in range(output_stems_mix_np.shape[0])]

        # add channel dimension
        input_stems_mix = input_stems_mix.unsqueeze(1).float()
        output_stems_mix = output_stems_mix.unsqueeze(1).float()

        description, cond_code = self.model.musicgen._prepare_tokens_and_attributes(instruction_text, input_stems_mix)
        _, label = self.model.musicgen._prepare_tokens_and_attributes(instruction_text, output_stems_mix)

        batch_size, codebook_size, seq_len = cond_code.shape

        output = self.model(
            input_code=label,
            text_description=description,
            condition_audio_code=cond_code,
            mode='train')

        logit = output.logits
        mask = output.mask
        logit_masked = logit[mask]
        label_masked = label[mask]
        loss_token = (self.criterion_1(logit_masked.reshape(-1, 2048),  # prediction loss
                              label_masked.reshape(-1)))

        loss = loss_token

        return loss, logit, label

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.size())

    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        loss, preds, y = self.forward(batch)
        return loss, preds, y

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        torch.cuda.empty_cache()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        # self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

        if len(self.generation_sample) < 10:
            input_stems_mix, output_stems_mix, instruction_text = batch

            input_stems_mix = input_stems_mix.unsqueeze(1).float()
            output_stems_mix = output_stems_mix.unsqueeze(1).float()

            description, cond_code = self.model.musicgen._prepare_tokens_and_attributes(instruction_text, input_stems_mix)

            cond_code = torch.cat(
                [cond_code, torch.ones_like(cond_code[:, :, 0:1]) * 2048], dim=-1)

            audio_values = self.model.generate(
                text_description=instruction_text,
                condition_audio_code=cond_code,
                num_samples=input_stems_mix.shape[0],
            )

            audio_values = self.model.musicgen.compression_model.decode(audio_values, None)

            input_path_list, output_path_list, ground_truth_list = [], [], []
            # save input_stems_mix and audio_values to tmp output folder
            for i, (input, output, ground_truth) in enumerate(zip(input_stems_mix, audio_values, output_stems_mix)):
                input_path = f"{self.hparams.tmp_dir}/{self.uuid}_input_{batch_idx}_{i}.wav"
                output_path = f"{self.hparams.tmp_dir}/{self.uuid}_output_{batch_idx}_{i}.wav"
                ground_truth_path = f"{self.hparams.tmp_dir}/{self.uuid}_ground_truth_{batch_idx}_{i}.wav"
                sf.write(input_path, input.float().squeeze(0).cpu().numpy(), 32000)
                # output needs to cut off the input length
                output = output.reshape(-1)[len(input):]
                sf.write(output_path, output.float().squeeze(0).cpu().numpy(), 32000)
                sf.write(ground_truth_path, ground_truth.float().squeeze(0).cpu().numpy(), 32000)
                input_path_list.append(input_path)
                output_path_list.append(output_path)
                ground_truth_list.append(ground_truth_path)

            # zip input, output, instruct, convert to wandb.Audio, wandb.Audio, text
            generation_sample = list(zip(input_path_list, output_path_list, ground_truth_list,instruction_text))

            generation_sample_wandb = [(wandb.Audio(input, sample_rate=32000),
                                       wandb.Audio(output, sample_rate=32000),
                                       wandb.Audio(ground_truth, sample_rate=32000),
                                       instruct) for input, output, ground_truth, instruct in generation_sample]

            self.generation_sample.extend(generation_sample_wandb)



    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.logger.log_table(key="val/audio",
                              columns=['input', 'output', 'ground_truth', 'instruct'],
                              data=self.generation_sample)
        self.generation_sample = []
        torch.cuda.empty_cache()



    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str, str], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        input_stems_mix, output_stems_mix, instruction_text, data_idx_str = batch
        batch_for_model = (input_stems_mix, output_stems_mix, instruction_text)
        loss, preds, targets = self.model_step(batch_for_model)
        # update and log metrics
        self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)



        input_stems_mix = input_stems_mix.unsqueeze(1).float()
        output_stems_mix = output_stems_mix.unsqueeze(1).float()

        description, cond_code = self.model.musicgen._prepare_tokens_and_attributes(instruction_text, input_stems_mix)

        cond_code = torch.cat(
            [cond_code, torch.ones_like(cond_code[:, :, 0:1]) * 2048], dim=-1)

        audio_values = self.model.generate(
            text_description=instruction_text,
            condition_audio_code=cond_code,
            num_samples=input_stems_mix.shape[0],
        )

        cond_code = torch.cat(
            [cond_code, torch.ones_like(cond_code[:, :, 0:1]) * 2048], dim=-1)

        audio_values = self.model.musicgen.compression_model.decode(audio_values, None)

        input_path_list, output_path_list, ground_truth_list = [], [], []
        # save input_stems_mix and audio_values to tmp output folder
        for i, (input, output, ground_truth, idx) in enumerate(zip(input_stems_mix, audio_values, output_stems_mix, data_idx_str)):
            input_path = f"{self.hparams.tmp_dir}/{self.uuid}_input_{batch_idx}_{i}.wav"
            output_path = f"{self.hparams.tmp_dir}/{self.uuid}_output_{batch_idx}_{i}.wav"
            ground_truth_path = f"{self.hparams.tmp_dir}/{self.uuid}_ground_truth_{batch_idx}_{i}.wav"
            sf.write(input_path, input.float().squeeze(0).cpu().numpy(), 32000)
            # output needs to cut off the input length
            output = output.reshape(-1)[len(input):]
            sf.write(output_path, output.float().squeeze(0).cpu().numpy(), 32000)
            eval_path = f"/weka2/home-yixiao/instruct-MusicGen/test_data/slakh_4stems/add/output/no_tune/{idx}.wav"
            sf.write(eval_path, output.float().squeeze(0).cpu().numpy(), 32000)
            print(f"Saved to {eval_path}")
            sf.write(ground_truth_path, ground_truth.float().squeeze(0).cpu().numpy(), 32000)
            input_path_list.append(input_path)
            output_path_list.append(output_path)
            ground_truth_list.append(ground_truth_path)





        # zip input, output, instruct, convert to wandb.Audio, wandb.Audio, text
        generation_sample = list(zip(input_path_list, output_path_list, ground_truth_list,instruction_text))

        generation_sample_wandb = [(wandb.Audio(input, sample_rate=32000),
                                   wandb.Audio(output, sample_rate=32000),
                                   wandb.Audio(ground_truth, sample_rate=32000),
                                   instruct) for input, output, ground_truth, instruct in generation_sample]

        self.generation_sample.extend(generation_sample_wandb)



    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.logger.log_table(key="test/audio",
                              columns=['input', 'output', 'ground_truth', 'instruct'],
                              data=self.generation_sample)
        self.generation_sample = []
        torch.cuda.empty_cache()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        # MAGIC
        if self.hparams.scheduler is not None:
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=10000)
            # scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class ReplaceNaN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, value=0):
        ctx.save_for_backward(input)
        output = input.clone()
        output[output != output] = value  # replace NaN with value
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input != input] = 0  # replace NaN gradient with 0
        return grad_input, None


class ArgmaxSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        # Perform the argmax operation and return indices
        return logits.argmax(dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


if __name__ == "__main__":
    s = InstructMusicGenAdapterLitModule(None, None, None, False)


    decoder_input_ids = (
            torch.ones((2, 4, 500), dtype=torch.long)
    )

    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],

    logits = s.model(
        input_code=decoder_input_ids,
        text_description=text,
        condition_audio_code=decoder_input_ids
    )

    print(logits.shape)



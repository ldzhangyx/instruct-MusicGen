"""PyTorch Lightning DataModule for the Slakh dataset with instruction-based music editing.

This module handles loading and preprocessing of the Slakh dataset for training
InstructMusicGen with add, remove, and extract instructions.
"""
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import mirdata
import random
import resampy
import time
import torch
import soundfile as sf
from tqdm import tqdm
import os
import concurrent
from concurrent.futures import ThreadPoolExecutor

class SlakhDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        time_in_seconds: int = 5,
        stereo_to_mono: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = False,
        volume_normalization: bool = True,
        average: bool = False,
    ) -> None:

        super().__init__()

        self.data_train = SlakhInstructDataset(data_path=data_dir,
                                               sample_rate=32000,
                                               time_in_seconds=time_in_seconds,
                                               stereo_to_mono=stereo_to_mono,
                                               volume_normalization=volume_normalization,
                                               average=average,
                                               split='train', )
        self.data_val = SlakhInstructDataset(data_path=data_dir,
                                             sample_rate=32000,
                                             time_in_seconds=time_in_seconds,
                                             stereo_to_mono=stereo_to_mono,
                                             volume_normalization=volume_normalization,
                                             average=average,
                                             split='validation', )
        self.data_test = SlakhInstructDataset(data_path=data_dir,
                                              sample_rate=32000,
                                              time_in_seconds=time_in_seconds,
                                              stereo_to_mono=stereo_to_mono,
                                              volume_normalization=volume_normalization,
                                              average=average,
                                              split='test', )


        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


class SlakhInstructDataset(Dataset):
    def __init__(self, data_path: str,
                 sample_rate: int,
                 time_in_seconds: int = 5,
                 stereo_to_mono: bool = True,
                 volume_normalization: bool = False,
                 average: bool = False,
                 split: str = 'train'):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.data = mirdata.initialize('slakh', data_home=data_path)
        self.indexes = self.data.get_mtrack_splits()[split]  # ['Track00001'~'Track02100']
        self.volume_normalization = volume_normalization
        self.average = average
        self.split = split

        self.instruct_set = [
            # 'generate',  # only output target stem
            'add',       # output target stem + input mix
            # 'drum_condition',
            'remove',    # output input mix - target stem
            # 'repeat',    # output input mix
            'extract',   # output target stem
            # 'replace',   # output target stem + input mix - replaced stem
            # 'remix'      # output new mix with target stem
        ]
        self.time_in_seconds = time_in_seconds
        self.stereo_to_mono = stereo_to_mono

        if split != 'train':  # remove 'repeat' instruction from validation and test set
            self.instruct_set = [i for i in self.instruct_set if i != 'repeat']

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx, retry_count=0, assigned_instruction=None, return_json=False, option='4_stems'):

        # evaluation?
        eval = False

        if self.split == 'test' and eval:
            # evaluation mode
            root_dir = "test_data"
            dataset = 'slakh_4stems'
            self.operation = operation = 'add'
            model = 'instruct-MusicGen'

            input_dir = f"{root_dir}/{dataset}/{operation}/input/"
            self.prediction_dir = prediction_dir = f"{root_dir}/{dataset}/{operation}/output/{model}/"
            ground_truth_dir = f"{root_dir}/{dataset}/{operation}/ground_truth/"
            instruction_dir = f"{root_dir}/{dataset}/{operation}/instruction/"

            os.makedirs(prediction_dir, exist_ok=True)

            with open(os.path.join(instruction_dir, f"{idx}.txt"), 'r') as f:
                instruction = f.readlines()[0].strip()[12:]

            audio_file = os.path.join(input_dir, f"{idx}.wav")
            output_file = os.path.join(ground_truth_dir, f"{idx}.wav")
            input_audio, sr = sf.read(audio_file)
            output_audio, sr = sf.read(output_file)

            return input_audio, output_audio, instruction, str(idx)

        else:

            raw_data = self.data.multitrack(self.indexes[idx])
            stem_list = raw_data.track_ids



            # we find that some audio_path is not exist, so we need to exclude it.
            stem_list = [
                stem for stem in stem_list if raw_data.tracks[stem].audio_path is not None
            ]


            stem_list_text = [raw_data.tracks[stem].instrument for stem in stem_list]

            stem_list_text_no_overlapping = []
            for stem_text in stem_list_text:
                if stem_text not in stem_list_text_no_overlapping:
                    stem_list_text_no_overlapping.append(stem_text)

            grouped_stem_dict = {}
            for stem_text in stem_list_text_no_overlapping:
                grouped_stem_dict[stem_text] = [stem for stem in stem_list if raw_data.tracks[stem].instrument == stem_text]


            # select random instruction
            if assigned_instruction is not None:
                instruction = assigned_instruction
            else:
                instruction = random.choice(self.instruct_set)
            # determine the target stem. all stem include 'other' should be filtered out. Then choose one.
            # if 'Drums' in stem_list_text:
            #     target_stem_key = stem_list[stem_list_text.index('Drums')]
            # else:
            #     target_stem_key = random.choice(stem_list)

            if option == '4_stems': # source and target is limited
                inters = set(stem_list_text_no_overlapping).intersection({'Drums', 'Bass', 'Piano', 'Guitar'})
                stem_list_text_no_overlapping = inters
                # we want the target stem to be one of the four instruments
                target_stem_key_text = random.choice(list(inters))
            else:
                target_stem_key_text = random.choice(stem_list_text_no_overlapping)

            input_stems_mix, output_stems_mix, instruction_text = None, None, None

            if instruction == 'repeat':
                # choose [1, N-1] stems in the rest of the stems as input mix
                input_stems_keys_text = random.sample(stem_list_text_no_overlapping, k=random.randint(1, len(stem_list_text_no_overlapping)))
                # input_stems_keys = [stem for stem in stem_list if stem != target_stem_key]
                output_stems_keys_text = input_stems_keys_text

                input_stems_keys = []
                for stem_text in input_stems_keys_text:
                    input_stems_keys += grouped_stem_dict[stem_text]
                output_stems_keys = input_stems_keys
                # mix input
                input_stems_mix = raw_data.get_target(input_stems_keys, average=self.average)
                # mix output
                output_stems_mix = input_stems_mix

                instruction_text = f"Music piece."

            elif instruction == 'add':
                # choose [1, N-1] stems in the rest of the stems as input mix
                input_stems_keys_text = random.sample([stem for stem in stem_list_text_no_overlapping if stem != target_stem_key_text], k=random.randint(1, len(stem_list_text_no_overlapping) - 1))
                # input_stems_keys = [stem for stem in stem_list if stem != target_stem_key]
                output_stems_keys_text = input_stems_keys_text + [target_stem_key_text]

                input_stems_keys = []
                for stem_text in input_stems_keys_text:
                    input_stems_keys += grouped_stem_dict[stem_text]

                output_stems_keys = []
                for stem_text in output_stems_keys_text:
                    output_stems_keys += grouped_stem_dict[stem_text]

                # mix input
                input_stems_mix = raw_data.get_target(input_stems_keys, average=self.average)
                # mix output
                output_stems_mix = raw_data.get_target(output_stems_keys, average=self.average)
                # instruction_text = f"instruction: Add {target_stem_key_text}."
                instruction_text = f"Music piece. Instruct: Add {target_stem_key_text}."

            elif instruction == 'remove':
                output_stems_keys_text = random.sample([stem for stem in stem_list_text_no_overlapping if stem != target_stem_key_text], k=random.randint(1, len(stem_list_text_no_overlapping) - 1))
                # choose [2, N] stems, must include target stem
                input_stems_keys_text = output_stems_keys_text + [target_stem_key_text]

                input_stems_keys = []
                for stem_text in input_stems_keys_text:
                    input_stems_keys += grouped_stem_dict[stem_text]

                output_stems_keys = []
                for stem_text in output_stems_keys_text:
                    output_stems_keys += grouped_stem_dict[stem_text]

                # mix input
                input_stems_mix = raw_data.get_target(input_stems_keys, average=self.average)
                # mix output
                output_stems_mix = raw_data.get_target(output_stems_keys, average=self.average)
                instruction_text = f"Music piece. Instruct: No {target_stem_key_text}."

            elif instruction == 'extract':
                # choose [1, N-1] stems in the rest of the stems as input mix
                input_stems_keys_text = [target_stem_key_text] + random.sample([stem for stem in stem_list_text_no_overlapping if stem != target_stem_key_text], k=random.randint(1, len(stem_list_text_no_overlapping) - 1))
                output_stems_keys_text = [target_stem_key_text]

                input_stems_keys = []
                for stem_text in input_stems_keys_text:
                    input_stems_keys += grouped_stem_dict[stem_text]

                output_stems_keys = []
                for stem_text in output_stems_keys_text:
                    output_stems_keys += grouped_stem_dict[stem_text]

                # mix input
                input_stems_mix = raw_data.get_target(input_stems_keys, average=self.average)
                # mix output
                output_stems_mix = raw_data.get_target(output_stems_keys, average=self.average)
                instruction_text = f"Music piece. Instruct: Only {target_stem_key_text}."

            elif instruction == 'generate':
                input_stems_keys_text = random.sample([stem for stem in stem_list_text_no_overlapping if stem != target_stem_key_text], k=random.randint(1, len(stem_list_text_no_overlapping) - 1))
                output_stems_keys_text = [target_stem_key_text]

                input_stems_keys = []
                for stem_text in input_stems_keys_text:
                    input_stems_keys += grouped_stem_dict[stem_text]

                output_stems_keys = []
                for stem_text in output_stems_keys_text:
                    output_stems_keys += grouped_stem_dict[stem_text]

                # mix input
                input_stems_mix = raw_data.get_target(input_stems_keys, average=self.average)
                # mix output
                output_stems_mix = raw_data.get_target(output_stems_keys, average=self.average)
                instruction_text = f"Music piece. Instruct: Generate {target_stem_key_text}. "

            else:
                raise NotImplementedError

            # post-processing
            if raw_data.audio[1] != self.sample_rate:
                input_stems_mix = resampy.resample(input_stems_mix, raw_data.audio[1], self.sample_rate, filter='kaiser_fast')
                output_stems_mix = resampy.resample(output_stems_mix, raw_data.audio[1], self.sample_rate, filter='kaiser_fast')

            if self.time_in_seconds is not None:
                num_samples = self.sample_rate * self.time_in_seconds
                # random pick the same offset. Consider min length.
                min_length = min(input_stems_mix.shape[1], output_stems_mix.shape[1])
                offset = random.randint(0, min_length - num_samples)
                input_stems_mix = input_stems_mix[:, offset:offset + num_samples]
                output_stems_mix = output_stems_mix[:, offset:offset + num_samples]

            target_mix = output_stems_mix - input_stems_mix

            # if more than half of the target mix is silence, we need to re-pick a sample
            # if volume is too low, we need to re-pick a sample
            if (np.max(np.abs(input_stems_mix)) < 0.1
                    or np.max(np.abs(output_stems_mix)) < 0.1
            or (instruction != 'repeat'
                and (
                        np.sum(np.abs(target_mix) < 0.01)
                        > (0.7 * target_mix.shape[1])
                    )
                )
            ):
                if retry_count < 10:  # limit the number of retries to 10
                    return self.__getitem__(idx, retry_count + 1, assigned_instruction=instruction, return_json=return_json)  # avoid wrong ratio
                else:
                    print(f"Retry count exceeds 10 times for {self.indexes[idx]}.")
                    pass

            if self.stereo_to_mono:
                input_stems_mix = input_stems_mix.mean(axis=0)
                output_stems_mix = output_stems_mix.mean(axis=0)

            if self.volume_normalization:
                if instruction == 'add':
                    input_stems_mix *= (len(stem_list) / len(output_stems_keys))  # n/N * N/(n+1) -> n/(n+1)
                    output_stems_mix *= (len(stem_list) / len(output_stems_keys)) # n+1/N * N/(n+1) -> 1
                elif instruction == 'remove':
                    input_stems_mix *= (len(stem_list) / len(input_stems_keys))   # n/N * N/n -> 1
                    output_stems_mix *= (len(stem_list) / len(input_stems_keys))  # (n-1)/N * N/n -> (n-1)/n
                elif instruction == 'extract':
                    input_stems_mix *= (len(stem_list) / len(input_stems_keys))   # n/N * N/n -> 1
                    output_stems_mix *= (len(stem_list) / len(input_stems_keys))  # 1/N * N/n -> 1/n
                elif instruction == 'repeat':
                    input_stems_mix *= (len(stem_list) / len(input_stems_keys))   # n/N * N/n -> 1
                    output_stems_mix *= (len(stem_list) / len(input_stems_keys))  # n/N * N/n -> 1
                elif instruction == 'generate':
                    input_stems_mix *= (len(stem_list) / len(input_stems_keys))  # n/N * N/n -> 1
                    output_stems_mix *= (len(stem_list) / len(input_stems_keys)) # 1/N * N/n -> 1/n

                # avoid clipping
                max_volume = max(np.max(np.abs(input_stems_mix)), np.max(np.abs(output_stems_mix)))
                if max_volume > 1.0:
                    input_stems_mix /= max_volume
                    output_stems_mix /= max_volume

            if return_json:
                instruction_json = {
                    "instruction_text": instruction_text,
                    "input_stems_list": input_stems_keys_text,
                }
                return input_stems_mix, output_stems_mix, instruction_json

            return input_stems_mix, output_stems_mix, instruction_text

    def get_test_files(self):
        root_folder = '/import/c4dm-04/yz007/instruct-MusicGen/test_data/slakh_allstems'

        def process_file(idx, instruction):
            input_stems_mix, output_stems_mix, instruction_json = self.__getitem__(idx,
                                                                                   assigned_instruction=instruction,
                                                                                   return_json=True,
                                                                                   option='all_stems')
            instruction_text = instruction_json['instruction_text']
            input_stems_keys_text = instruction_json['input_stems_list']
            os.makedirs(os.path.join(root_folder, instruction, 'input'), exist_ok=True)
            os.makedirs(os.path.join(root_folder, instruction, 'ground_truth'), exist_ok=True)
            os.makedirs(os.path.join(root_folder, instruction, 'instruction'), exist_ok=True)
            sf.write(os.path.join(root_folder, instruction, 'input', f'{idx}.wav'), input_stems_mix, self.sample_rate)
            sf.write(os.path.join(root_folder, instruction, 'ground_truth', f'{idx}.wav'), output_stems_mix, self.sample_rate)
            print(f"Finish processing {root_folder}/{instruction}/{idx}.wav")

            # add a line of instruction text to file
            with open(os.path.join(root_folder, instruction, 'instruction', f'{idx}.txt'), 'w') as f:
                f.write(f"Instruction: {instruction_text}\n")
                f.write(f"Stems: {', '.join(input_stems_keys_text)}\n")

            print(f"Finish processing {root_folder}/{instruction}/{idx}.wav")


        with ThreadPoolExecutor() as executor:
            for instruction in ['add', 'remove', 'extract']:
                futures = [executor.submit(process_file, idx, instruction) for idx in range(len(self))]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    pass

if __name__ == "__main__":
    dataset = SlakhDataModule(data_dir='/import/c4dm-datasets/Slakh/')
    print(len(dataset.data_test))
    dataset.data_test.get_test_files()

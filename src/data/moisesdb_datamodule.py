from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import os
from moisesdb.dataset import MoisesDB
from moisesdb.track import pad_and_mix
import random
import resampy
import soundfile as sf
from tqdm import tqdm
import os
import concurrent
from concurrent.futures import ThreadPoolExecutor

class MoisesDBDataModule(LightningDataModule):

    def __init__(
        self,
        cache_dir: str,
        data_dir: str = "/home/yixiao/instruct-MusicGen/data/moisesdb/",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        time_in_seconds: int = 5,
        stereo_to_mono: bool = True,

    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.dataset = MoisesDBInstructDataset(data_path=data_dir,
                                               sample_rate=32000,
                                               time_in_seconds=time_in_seconds,
                                               cache_dir=cache_dir,
                                               stereo_to_mono=stereo_to_mono,)  # MusicGen uses 32kHz sample rate

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        train_val_test_split_ratio = [0.8, 0.1, 0.1]
        dataset_size = len(self.dataset)
        train_size = int(train_val_test_split_ratio[0] * dataset_size)
        val_size = int(train_val_test_split_ratio[1] * dataset_size)
        test_size = dataset_size - train_size - val_size


        self.data_train, self.data_val, self.data_test = random_split(
            dataset=self.dataset,
            lengths=[train_size, val_size, test_size],
            # generator=torch.Generator().manual_seed(42),
        )


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


class MoisesDBInstructDataset(Dataset):
    def __init__(self, data_path: str,
                 sample_rate: int,
                 cache_dir: str,
                 time_in_seconds: int = 5,
                 stereo_to_mono: bool = True):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.data = MoisesDB(data_path=self.data_path)
        self.instruct_set = [
            # 'generate',  # only output target stem
            'add',       # output target stem + input mix
            'remove',    # output input mix - target stem
            'extract',   # output target stem
            # 'replace',   # output target stem + input mix - replaced stem
            # 'remix'      # output new mix with target stem
            # 'repeat',
        ]
        self.time_in_seconds = time_in_seconds
        self.stereo_to_mono = stereo_to_mono
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx, retry_count=0, assigned_instruction='add', return_json=True):

        raw_data = self.data[idx]
        cache_file = os.path.join(self.cache_dir, f"{idx}.pt")

        # get stems. We assume there are N stems
        # TODO: For the current experiment, substems are not being considered. We should consider them in the future.
        if os.path.exists(cache_file):
            stems = torch.load(cache_file)  # stems = {'vocals': ..., 'bass': ...}
        else:
            stems = raw_data.stems
            torch.save(stems, cache_file)
        stems.pop('vocals', None)

        # select random instruction
        instruction = random.choice(self.instruct_set)
        # determine the target stem. all stem include 'other' should be filtered out. Then choose one.
        target_stem_key = random.choice([stem for stem in stems.keys() if 'other' not in stem])
        target_subtrack_key_str = ', '.join(raw_data.sources[target_stem_key].keys())

        target_stem_mix = stems[target_stem_key]

        input_stems_mix, output_stems_mix, instruction_text = None, None, None

        if assigned_instruction is not None:
            instruction = assigned_instruction

        if instruction == 'add':
            # choose [1, N-1] stems in the rest of the stems as input mix
            input_stems_keys = [stem for stem in stems.keys() if stem != target_stem_key] # random.sample([stem for stem in stems.keys() if stem != target_stem_key], k=random.randint(1, len(stems) - 1))
            output_stems_keys = [target_stem_key] + input_stems_keys
            # mix input
            input_stems_mix = pad_and_mix([
                stems[stem] for stem in input_stems_keys
            ])
            # mix output
            output_stems_mix = pad_and_mix([
                stems[stem] for stem in output_stems_keys
            ])
            instruction_text = f"Music piece. Instruction: Add {target_stem_key}."

        elif instruction == 'remove':
            # choose [2, N] stems, must include target stem
            input_stems_keys = [target_stem_key] + random.sample([stem for stem in stems.keys() if stem != target_stem_key], k=random.randint(1, len(stems) - 1))
            # mix input
            input_stems_mix = pad_and_mix([
                stems[stem] for stem in input_stems_keys
            ])
            # mix output
            output_stems_keys = [stem for stem in input_stems_keys if stem != target_stem_key]
            output_stems_mix = pad_and_mix([
                stems[stem] for stem in input_stems_keys if stem != target_stem_key
            ])
            instruction_text = f"Music piece. Instruction: No {target_stem_key}"
        elif instruction == 'extract':
            input_stems_keys = [target_stem_key] + random.sample(
                [stem for stem in stems.keys() if stem != target_stem_key], k=random.randint(1, len(stems) - 1))
            # mix input
            input_stems_mix = pad_and_mix([
                stems[stem] for stem in input_stems_keys
            ])
            # mix output
            output_stems_keys = [target_stem_key]
            output_stems_mix = stems[target_stem_key]
            instruction_text = f"Music piece. Instruction: Only {target_stem_key}."

        elif instruction == 'repeat':
            # choose [1, N-1] stems in the rest of the stems as input mix
            input_stems_keys = random.sample([stem for stem in stems.keys()], k=random.randint(1, len(stems)))
            output_stems_keys = input_stems_keys
            # mix input
            input_stems_mix = pad_and_mix([
                stems[stem] for stem in input_stems_keys
            ])
            # mix output
            output_stems_mix = input_stems_mix
            instruction_text = f"Music piece. Instruction: Repeat."
        else:
            # TODO: implement other instructions
            raise NotImplementedError

        # post-processing
        if raw_data.sr != self.sample_rate:
            input_stems_mix = resampy.resample(input_stems_mix, raw_data.sr, self.sample_rate, filter='kaiser_fast')
            output_stems_mix = resampy.resample(output_stems_mix, raw_data.sr, self.sample_rate, filter='kaiser_fast')

        if self.time_in_seconds is not None:
            num_samples = self.sample_rate * self.time_in_seconds
            # random pick the same offset. Consider min length.
            min_length = min(input_stems_mix.shape[1], output_stems_mix.shape[1])
            offset = random.randint(0, min_length - num_samples)
            input_stems_mix = input_stems_mix[:, offset:offset + num_samples]
            output_stems_mix = output_stems_mix[:, offset:offset + num_samples]

        # if a lot of silence appears, retry
        if (np.max(np.abs(input_stems_mix)) < 0.1
            or np.max(np.abs(output_stems_mix)) < 0.1
            or np.max(np.abs(target_stem_mix)) < 0.1
        ):
            if retry_count < 10:  # limit the number of retries to 5
                return self.__getitem__(idx, retry_count + 1, assigned_instruction=instruction, return_json=return_json)
            else:
                print(f"Retry count exceeds 10 times for {idx}.")
                pass

        if self.stereo_to_mono:
            input_stems_mix = input_stems_mix.mean(axis=0)
            output_stems_mix = output_stems_mix.mean(axis=0)

        self.volume_normalization = True

        if self.volume_normalization:
            if instruction == 'add':
                input_stems_mix *= (len(stems) / len(output_stems_keys))  # n/N * N/(n+1) -> n/(n+1)
                output_stems_mix *= (len(stems) / len(output_stems_keys)) # n+1/N * N/(n+1) -> 1
            elif instruction == 'remove':
                input_stems_mix *= (len(stems) / len(input_stems_keys))   # n/N * N/n -> 1
                output_stems_mix *= (len(stems) / len(input_stems_keys))  # (n-1)/N * N/n -> (n-1)/n
            elif instruction == 'extract':
                input_stems_mix *= (len(stems) / len(input_stems_keys))   # n/N * N/n -> 1
                output_stems_mix *= (len(stems) / len(input_stems_keys))  # 1/N * N/n -> 1/n
            elif instruction == 'repeat':
                input_stems_mix *= (len(stems) / len(input_stems_keys))   # n/N * N/n -> 1
                output_stems_mix *= (len(stems) / len(input_stems_keys))  # n/N * N/n -> 1
            elif instruction == 'generate':
                input_stems_mix *= (len(stems) / len(input_stems_keys))  # n/N * N/n -> 1
                output_stems_mix *= (len(stems) / len(input_stems_keys)) # 1/N * N/n -> 1/n

            # avoid clipping
            max_volume = max(np.max(np.abs(input_stems_mix)), np.max(np.abs(output_stems_mix)))
            if max_volume > 1.0:
                input_stems_mix /= max_volume
                output_stems_mix /= max_volume

        if return_json:
            instruction_json = {
                "instruction_text": instruction_text,
                "input_stems_list": input_stems_keys,
            }
            return input_stems_mix, output_stems_mix, instruction_json

        return input_stems_mix, output_stems_mix, instruction_text


    def get_test_files(self):
        root_folder = '/data2/yixiao/test_data/moisesDB/'

        def process_file(idx, instruction):
            input_stems_mix, output_stems_mix, instruction_json = self.__getitem__(idx, assigned_instruction=instruction, return_json=True)

    def get_test_files(self):
        root_folder = '/import/c4dm-04/yz007/instruct-MusicGen/test_data/moisesDB/'

        def process_file(idx, instruction):
            input_stems_mix, output_stems_mix, instruction_json = self.__getitem__(idx,
                                                                                   assigned_instruction=instruction,
                                                                                   return_json=True)

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
    dataset = MoisesDBDataModule(data_dir='/import/c4dm-datasets/Slakh/')
    print(len(dataset.data_test))
    dataset.data_test.get_test_files()


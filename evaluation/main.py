"""Evaluation script for measuring audio generation quality metrics.

This script computes various metrics including CLAP score, FAD, KL divergence,
Inception Score, SSIM, SI-SDR, and SI-SDRi for generated audio.
"""
import argparse
import os
from typing import List

import torch
import torchaudio
from audioldm_eval import EvaluationHelper
from frechet_audio_distance import CLAPScore
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate audio generation quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing test data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="slakh_4stems",
        help="Dataset name"
    )
    parser.add_argument(
        "--operation",
        type=str,
        default="add",
        choices=["add", "remove", "extract"],
        help="Operation type"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="no_tune",
        help="Model name/identifier"
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device ID to use"
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    root_dir = args.root_dir
    dataset = args.dataset
    operation = args.operation
    model = args.model

    # GPU acceleration is preferred
    device = torch.device(f"cuda:{args.cuda_device}")

    input_dir = f"{root_dir}/{dataset}/{operation}/input/"
    prediction_dir = f"{root_dir}/{dataset}/{operation}/output/{model}/"
    ground_truth_dir = f"{root_dir}/{dataset}/{operation}/ground_truth/"
    instruction_dir = f"{root_dir}/{dataset}/{operation}/instruction/"

    file_number = len(os.listdir(prediction_dir))

    # --- CLAP ---
    clap = CLAPScore(
        submodel_name="music_audioset",
        verbose=True,
        enable_fusion=False,
    )
    clap_score = clap.score(
        text_path=f"{instruction_dir}/text.csv",
        audio_dir=prediction_dir,
        text_column="caption",
    )
    print(f"CLAP Score: {clap_score}")

    # Initialize a helper instance
    evaluator = EvaluationHelper(16000, device)

    # --- FAD, KL, IS---
    # Perform evaluation, result will be printed out and saved as json
    metrics = evaluator.main(
        prediction_dir,
        ground_truth_dir,
        limit_num=None  # If you only intend to evaluate X (int) pairs of data, set limit_num=X
    )

    # Clean up temporary files
    temp_files = [
        prediction_dir + "/classifier_logits_feature_cache.pkl",
        ground_truth_dir + "/classifier_logits_feature_cache.pkl",
        prediction_dir + "/_fad_feature_cache.npy",
        ground_truth_dir + "/_fad_feature_cache.npy",
    ]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # --- SSIM ---
    print("Calculating SSIM")
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_measures: List[float] = []

    for i in tqdm(range(file_number)):
        prediction_file = f"{prediction_dir}/{i}.wav"
        ground_truth_file = f"{ground_truth_dir}/{i}.wav"
        prediction_waveform, sr1 = torchaudio.load(prediction_file)
        ground_truth_waveform, sr2 = torchaudio.load(ground_truth_file)

        # Resample to 16000 Hz
        prediction_waveform = torchaudio.transforms.Resample(sr1, 16000)(prediction_waveform)
        ground_truth_waveform = torchaudio.transforms.Resample(sr2, 16000)(ground_truth_waveform)

        # Compute mel-spectrogram
        prediction_mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000)(prediction_waveform)
        ground_truth_mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000)(ground_truth_waveform)

        prediction_mel_expand = prediction_mel.reshape(1, *prediction_mel.shape)
        ground_truth_mel_expand = ground_truth_mel.reshape(1, *ground_truth_mel.shape)

        if prediction_mel_expand.shape[-1] > ground_truth_mel_expand.shape[-1]:
            prediction_mel_expand = prediction_mel_expand[:, :, :, :ground_truth_mel_expand.shape[-1]]

        ssim_measures.append(ssim(prediction_mel_expand, ground_truth_mel_expand).item())

    print(f"SSIM: {sum(ssim_measures) / len(ssim_measures)}")

    if operation in ['remove', 'extract']:
        # --- SI-SDR & SI-SDRi ---
        print("Calculating SI-SDR and SI-SDRi")
        sdr = ScaleInvariantSignalDistortionRatio()
        sdr_measures: List[float] = []
        sdri_measures: List[float] = []

        for i in tqdm(range(file_number)):
            prediction_file = f"{prediction_dir}/{i}.wav"
            ground_truth_file = f"{ground_truth_dir}/{i}.wav"
            input_file = f"{input_dir}/{i}.wav"
            prediction_waveform, sr1 = torchaudio.load(prediction_file)
            ground_truth_waveform, sr2 = torchaudio.load(ground_truth_file)
            input_waveform, sr3 = torchaudio.load(input_file)

            # Resample to 16000 Hz
            prediction_waveform = torchaudio.transforms.Resample(sr1, 16000)(prediction_waveform)
            ground_truth_waveform = torchaudio.transforms.Resample(sr2, 16000)(ground_truth_waveform)
            input_waveform = torchaudio.transforms.Resample(sr3, 16000)(input_waveform)

            if prediction_waveform.shape[-1] > ground_truth_waveform.shape[-1]:
                prediction_waveform = prediction_waveform[:, :ground_truth_waveform.shape[-1]]

            sdr_1 = sdr(prediction_waveform, ground_truth_waveform)
            sdr_0 = sdr(input_waveform, ground_truth_waveform)

            sdr_measures.append(sdr_1.item())
            sdri_measures.append((sdr_1 - sdr_0).item())

        print(f"SI-SDR: {sum(sdr_measures) / len(sdr_measures)}")
        print(f"SI-SDRi: {sum(sdri_measures) / len(sdri_measures)}")


if __name__ == "__main__":
    main()

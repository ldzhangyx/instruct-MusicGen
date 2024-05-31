from frechet_audio_distance import FrechetAudioDistance, CLAPScore
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import torch
from audioldm_eval import EvaluationHelper
import shutil
import torchaudio
import os
from tqdm import tqdm

# SETTINGS
root_dir = "/weka2/home-yixiao/instruct-MusicGen/test_data"

dataset = 'slakh_4stems'
operation = 'add'
model = 'no_tune'

# GPU acceleration is preferred

device = torch.device(f"cuda:{0}")

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
print(clap_score)


# Initialize a helper instance
evaluator = EvaluationHelper(16000, device)

# --- FAD, KL, IS---
# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    prediction_dir,
    ground_truth_dir,
    limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
)
# check temp files and delete
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
# Initialize the metric
print("Calculating SSIM")
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
# Load the audio files
measure = []
for i in tqdm(range(file_number)):
    prediction_file = f"{prediction_dir}/{i}.wav"
    ground_truth_file = f"{ground_truth_dir}/{i}.wav"
    prediction_waveform, sr1 = torchaudio.load(prediction_file)
    ground_truth_waveform, sr2 = torchaudio.load(ground_truth_file)
    # resample to 16000 Hz
    prediction_waveform = torchaudio.transforms.Resample(sr1, 16000)(prediction_waveform)
    ground_truth_waveform = torchaudio.transforms.Resample(sr2, 16000)(ground_truth_waveform)


    # mel-spectrogram
    prediction_mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000)(prediction_waveform)
    ground_truth_mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000)(ground_truth_waveform)

    prediction_mel_expand = prediction_mel.reshape(1, *prediction_mel.shape)
    ground_truth_mel_expand = ground_truth_mel.reshape(1, *ground_truth_mel.shape)
    
    if prediction_mel_expand.shape[-1] > ground_truth_mel_expand.shape[-1]:
      prediction_mel_expand = prediction_mel_expand[:, :, :, :ground_truth_mel_expand.shape[-1]]

    # ssim
    measure.append(ssim(prediction_mel_expand, ground_truth_mel_expand))

print(f"SSIM: {sum(measure) / len(measure)}")

if operation in ['remove', 'extract']:
    # --- SI-SDR & SI-SDRi ---
    # Initialize the metric
    print("Calculating SI-SDR")
    sdr = ScaleInvariantSignalDistortionRatio()
    # Load the audio files
    sdr_measure = []
    sdri_measure = []
    for i in tqdm(range(file_number)):
        prediction_file = f"{prediction_dir}/{i}.wav"
        ground_truth_file = f"{ground_truth_dir}/{i}.wav"
        input_file = f"{input_dir}/{i}.wav"
        prediction_waveform, sr1 = torchaudio.load(prediction_file)
        ground_truth_waveform, sr2 = torchaudio.load(ground_truth_file)
        input_waveform, sr3 = torchaudio.load(input_file)
    
        
        # resample to 16000 Hz
        prediction_waveform = torchaudio.transforms.Resample(sr1, 16000)(prediction_waveform)
        ground_truth_waveform = torchaudio.transforms.Resample(sr2, 16000)(ground_truth_waveform)
        input_waveform = torchaudio.transforms.Resample(sr3, 16000)(input_waveform)
        
        
        if prediction_waveform.shape[-1] > ground_truth_waveform.shape[-1]:
            prediction_waveform = prediction_waveform[:, :ground_truth_waveform.shape[-1]]

        sdr_1 = sdr(prediction_waveform, ground_truth_waveform)
        sdr_0 = sdr(input_waveform, ground_truth_waveform)

        # si-sdr
        sdr_measure.append(sdr_1)

        # si-sdri
        sdri_measure.append(sdr_1 - sdr_0)

    print(f"SI-SDR: {sum(sdr_measure) / len(sdr_measure)}")
    print(f"SI-SDRi: {sum(sdri_measure) / len(sdri_measure)}")

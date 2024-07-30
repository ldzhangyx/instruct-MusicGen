import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import soundfile as sf
from typing import Tuple
from src.models.components.model import Instructor
from src.models.instructmusicgenadapter_module import InstructMusicGenAdapterLitModule

from lightning import LightningModule
from hydra.experimental import compose, initialize
from omegaconf import DictConfig



def generate_edited_audio(model_ckpt_path: str, input_audio_path: str, instruction: str, output_audio_path: str) -> None:
    """
    Generate edited audio based on the input audio and instruction.

    :param model_ckpt_path: Path to the model checkpoint file
    :param config_path: Path to the Hydra configuration directory
    :param input_audio_path: Path to the input audio file
    :param instruction: Instruction string for editing
    :param output_audio_path: Path to save the output audio file
    """

    with torch.no_grad():
        # Load the model from the checkpoint
        model = InstructMusicGenAdapterLitModule.load_from_checkpoint(model_ckpt_path)
        model.eval()

        instruction = [instruction]

        # Read the input audio file
        input_audio, sample_rate = sf.read(input_audio_path)

        # Convert the audio to a tensor
        input_audio_tensor = torch.tensor(input_audio).unsqueeze(0).unsqueeze(0).float()

        # Prepare model inputs
        with torch.cuda.amp.autocast():
            description, cond_code = model.model.musicgen._prepare_tokens_and_attributes(instruction, input_audio_tensor)

        # Expand cond_code to match model requirements
        cond_code = torch.cat([cond_code, torch.ones_like(cond_code[:, :, 0:1]) * 2048], dim=-1)

        # Generate audio using the model
        with torch.cuda.amp.autocast():
            audio_values = model.model.generate(
                text_description=instruction,
                condition_audio_code=cond_code,
                num_samples=1,
            )

        # Decode the generated audio
        generated_audio = model.model.musicgen.compression_model.decode(audio_values, None).squeeze().cpu().detach().numpy()

        # Save the generated audio file
        sf.write(output_audio_path, generated_audio, sample_rate)

if __name__ == "__main__":
    model_ckpt_path = "/weka2/home-yixiao/instruct-MusicGen/ckpts/epoch_161.ckpt"
    input_audio_path = "/weka2/home-yixiao/instruct-MusicGen/test_data/slakh_4stems/extract/input/0.wav"
    instruction = "Music piece. Instruct: Only Drums."
    output_audio_path = "/weka2/home-yixiao/instruct-MusicGen/output_audio.wav"

    generate_edited_audio(model_ckpt_path, input_audio_path, instruction, output_audio_path)

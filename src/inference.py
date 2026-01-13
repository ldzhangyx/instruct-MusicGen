"""Inference script for generating edited audio using InstructMusicGen model."""
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import soundfile as sf
from src.models.instructmusicgenadapter_module import InstructMusicGenAdapterLitModule
from src.constants import CODEBOOK_SIZE, DEFAULT_NUM_SAMPLES, SAMPLE_RATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def generate_edited_audio(
    model_ckpt_path: str,
    input_audio_path: str,
    instruction: str,
    output_audio_path: str
) -> None:
    """Generate edited audio based on the input audio and instruction.

    Args:
        model_ckpt_path: Path to the model checkpoint file
        input_audio_path: Path to the input audio file
        instruction: Instruction string for editing (e.g., "Music piece. Instruct: Only Drums.")
        output_audio_path: Path to save the output audio file

    Raises:
        FileNotFoundError: If the model checkpoint or input audio file doesn't exist
        RuntimeError: If audio generation fails
        ValueError: If the instruction is empty or invalid
    """
    # Validate inputs
    if not instruction or not instruction.strip():
        raise ValueError("Instruction cannot be empty")

    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_ckpt_path}")

    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"Input audio file not found at: {input_audio_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_audio_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    try:
        logger.info(f"Loading model checkpoint from: {model_ckpt_path}")
        with torch.no_grad():
            # Load the model from the checkpoint
            model = InstructMusicGenAdapterLitModule.load_from_checkpoint(model_ckpt_path)
            model.eval()

            instruction_list = [instruction]

            # Read the input audio file
            logger.info(f"Reading input audio from: {input_audio_path}")
            input_audio, sample_rate = sf.read(input_audio_path)

            # Convert the audio to a tensor
            input_audio_tensor = torch.tensor(input_audio).unsqueeze(0).unsqueeze(0).float()

            # Prepare model inputs
            logger.info("Preparing model inputs...")
            with torch.cuda.amp.autocast():
                description, cond_code = model.model.musicgen._prepare_tokens_and_attributes(
                    instruction_list, input_audio_tensor
                )

            # Expand cond_code to match model requirements
            cond_code = torch.cat(
                [cond_code, torch.ones_like(cond_code[:, :, 0:1]) * CODEBOOK_SIZE], dim=-1
            )

            # Generate audio using the model
            logger.info("Generating edited audio...")
            with torch.cuda.amp.autocast():
                audio_values = model.model.generate(
                    text_description=instruction_list,
                    condition_audio_code=cond_code,
                    num_samples=DEFAULT_NUM_SAMPLES,
                )

            # Decode the generated audio
            logger.info("Decoding generated audio...")
            generated_audio = (
                model.model.musicgen.compression_model.decode(audio_values, None)
                .squeeze()
                .cpu()
                .detach()
                .numpy()
            )

            # Save the generated audio file
            logger.info(f"Saving output audio to: {output_audio_path}")
            sf.write(output_audio_path, generated_audio, SAMPLE_RATE)
            logger.info("Audio generation completed successfully!")

    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to generate audio: {str(e)}") from e

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate edited audio using InstructMusicGen model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file"
    )
    parser.add_argument(
        "--input_audio_path",
        type=str,
        required=True,
        help="Path to the input audio file"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Instruction string for editing (e.g., 'Music piece. Instruct: Only Drums.')"
    )
    parser.add_argument(
        "--output_audio_path",
        type=str,
        default="output_audio.wav",
        help="Path to save the output audio file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_edited_audio(
        model_ckpt_path=args.model_ckpt_path,
        input_audio_path=args.input_audio_path,
        instruction=args.instruction,
        output_audio_path=args.output_audio_path
    )

"""Constants used throughout the InstructMusicGen project."""

# Audio processing constants
SAMPLE_RATE = 32000  # MusicGen uses 32kHz sample rate
DEFAULT_RESAMPLING_RATE = 16000  # Standard resampling rate for evaluation metrics

# Model constants
CODEBOOK_SIZE = 2048  # Size of the codebook used in compression model
MAX_SEQUENCE_LENGTH = 1500  # Maximum sequence length for generation

# Training constants
DEFAULT_WARMUP_STEPS = 100  # Number of warmup steps for learning rate scheduler
DEFAULT_TRAINING_STEPS = 10000  # Default total training steps

# Generation constants
DEFAULT_NUM_SAMPLES = 1  # Default number of samples to generate
MAX_GENERATION_SAMPLES = 10  # Maximum number of generation samples to log

# Data processing constants
DEFAULT_TIME_IN_SECONDS = 5  # Default audio duration in seconds
MONO_CHANNELS = 1  # Number of channels for mono audio
STEREO_CHANNELS = 2  # Number of channels for stereo audio

# Evaluation constants
EVAL_SAMPLE_RATE = 16000  # Sample rate for evaluation metrics
DEFAULT_MEL_BINS = 128  # Default number of mel bins for spectrogram

# Retry constants
MAX_RETRIES = 3  # Maximum number of retries for operations that may fail
RETRY_DELAY = 1.0  # Delay in seconds between retries

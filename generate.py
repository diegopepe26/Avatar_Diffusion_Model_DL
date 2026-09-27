"""Generation with a trained model: load an experiment, then turn a caption and some seeds into images.

Used by the web demo (app.py) and, later, by the evaluation. The network is rebuilt with the settings saved
in the checkpoint of the experiment, so a 32x32 run works whatever IMAGE_SIZE says in config.py.
"""

import torch

from config import Config
from models.diffusion import DiffusionModel
from models.noise_scheduler import NoiseScheduler
from models.sampling import sample
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.vocabulary import Vocabulary

# the settings that decide the shape of the network: they come from the checkpoint, not from config.py
NETWORK_SETTINGS = ['IMAGE_SIZE', 'TEXT_CONDITIONING', 'D_MODEL', 'NUM_HEADS', 'NUM_ENCODER_BLOCKS', 'FFN_DIM',
                    'DROPOUT', 'ATTENTION', 'UNET_CHANNELS', 'TIME_DIM', 'NUM_TIMESTEPS']


def run_folders(config):
    """The trained experiments: the folders of RUNS_DIR that contain a last.pt.

    Args:
        config: the project Config (uses RUNS_DIR).

    Returns:
        List of folders, in alphabetical order (empty if nothing has been trained yet).
    """
    if not config.RUNS_DIR.exists():
        return []
    return sorted(folder for folder in config.RUNS_DIR.iterdir() if (folder / 'last.pt').exists())


def load_model(run_dir):
    """Rebuild the model of an experiment with the settings and the EMA weights of its checkpoint.

    Args:
        run_dir: folder of the experiment, e.g. runs/conditional_32.

    Returns:
        (model in eval() on the GPU if there is one, NoiseScheduler, Config of the run, Vocabulary).
    """
    checkpoint = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
    config = Config()
    for name in NETWORK_SETTINGS:
        setattr(config, name, checkpoint['settings'][name])
    vocabulary = Vocabulary(config)
    vocabulary.load(config.VOCABULARY_FILE)
    model = DiffusionModel(config, vocabulary)
    model.load_state_dict(checkpoint['ema'])        # the EMA weights: the ones that give the best images
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device).eval(), NoiseScheduler(config), config, vocabulary


def generate(model, scheduler, config, vocabulary, caption, seeds, guidance):
    """Generate one image per seed, each with its own random generator.

    One sample() per image: the image of a seed is the same whether it is generated alone or with others.

    Args:
        model, scheduler, config, vocabulary: as returned by load_model.
        caption: the prompt, e.g. 'a cartoon avatar with pale skin, long blonde hair, no glasses and a beard'.
        seeds: list of integers, one per image.
        guidance: strength of classifier-free guidance (ignored by a model without text).

    Returns:
        List of PIL images (values 0-255, IMAGE_SIZE x IMAGE_SIZE), in the order of the seeds.
    """
    device = next(model.parameters()).device
    tokens = torch.tensor([vocabulary.encode(caption)], device=device)
    config.GUIDANCE_SCALE = guidance
    return [ImagePreprocessor.denormalize(sample(model, scheduler, tokens, config, seed)[0]) for seed in seeds]


def is_held_out(words, config):
    """Tell whether a choice of words contains one of the pairs held out of training.

    Args:
        words: {attribute: word}, e.g. {'face_color': 'dark', 'hair': 'long', ...}.
        config: the project Config (uses OOD_PAIRS).

    Returns:
        True for a combination never seen in training (OOD).
    """
    return any(all(words[attribute] == word for attribute, word in pair.items()) for pair in config.OOD_PAIRS)

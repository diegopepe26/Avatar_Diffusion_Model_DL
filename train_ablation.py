"""Pooled ablation, the conditional model with the caption pooled into one vector and given to the UNet with FiLM (a
scale and a shift for every channel) in the blocks that would have the cross-attention (16x16 and 8x8).

It works at 32x32, to be compared with runs/conditional_32. The experiment goes to runs/ablation_pooled/conditional_32
(runs/conditional_32 stays as it is).
"""

from config import Config
from train import main


class ConfigPooled(Config):
    """The project Config with the pooled caption. All the other settings stay the same."""

    CONDITIONING = 'pooled'
    RUNS_DIR = Config.RUNS_DIR / 'ablation_pooled'


if __name__ == '__main__':
    main(ConfigPooled())

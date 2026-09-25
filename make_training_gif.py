"""Turns the control grids of a training (samples_epochXXX.png) into a GIF, with the epoch written on top.

Run it with:  python make_training_gif.py runs/conditional
The result is runs/conditional/training.gif: one frame per grid, in epoch order, to see how the training
improves the images.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import Config

FRAME_SECONDS = 0.5        # how long every grid stays on screen
LAST_FRAME_SECONDS = 2.0   # the last grid stays longer, then the GIF starts again
BAND_HEIGHT = 60           # white band above the grid, with the epoch


def epoch_of(grid_file):
    """Read the epoch from the name of a grid.

    Args:
        grid_file: path like runs/conditional/samples_epoch025.png.

    Returns:
        The epoch as a number, e.g. 25 (numbers, not text: 1000 comes after 200).
    """
    return int(grid_file.stem.removeprefix('samples_epoch'))


def make_frame(grid, epoch):
    """Put a white band with the epoch above a grid.

    Args:
        grid: PIL image of a control grid.
        epoch: epoch of the grid.

    Returns:
        PIL image, BAND_HEIGHT pixels taller than the grid.
    """
    frame = Image.new('RGB', (grid.width, grid.height + BAND_HEIGHT), 'white')
    frame.paste(grid, (0, BAND_HEIGHT))     # the band is above: it never covers the avatars
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=36)
    draw.text((frame.width / 2, BAND_HEIGHT / 2), f'Epoch {epoch}', fill='black', font=font, anchor='mm')
    return frame


def main(folder):
    """Collect the grids of one experiment and save them as training.gif in the same folder.

    Args:
        folder: the folder of the experiment, e.g. runs/conditional.
    """
    grids = sorted(folder.glob('samples_epoch*.png'), key=epoch_of)
    if not grids:
        raise ValueError(f'no samples_epoch*.png in {folder}: is it the folder of a training?')
    frames = [make_frame(Image.open(grid).convert('RGB'), epoch_of(grid)) for grid in grids]
    durations = [int(FRAME_SECONDS * 1000)] * (len(frames) - 1) + [int(LAST_FRAME_SECONDS * 1000)]   # milliseconds
    gif_file = folder / 'training.gif'
    frames[0].save(gif_file, save_all=True, append_images=frames[1:], duration=durations, loop=0)   # loop=0: forever
    print(f'{len(frames)} grids, epochs {epoch_of(grids[0])}-{epoch_of(grids[-1])}: {gif_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn the control grids of a training into a GIF.')
    parser.add_argument('folder', nargs='?', type=Path, default=Config.RUNS_DIR / 'conditional',
                        help='folder of the experiment (default: runs/conditional)')
    main(parser.parse_args().folder)

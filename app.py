"""Web demo: choose the attributes, the seed and the experiment, then generate and look at the avatars.

Run it with:  python app.py            (then open the address it prints, e.g. http://127.0.0.1:7860)
On Colab:     python app.py --share    (it also prints a public link, e.g. to show the demo at the exam)
"""

import argparse
import time

import gradio as gr
from PIL import Image

from config import Config
from generate import generate, is_held_out, load_model, run_folders
from preprocessing.caption_generator import CaptionGenerator

# label in the page -> attribute of Config.MAPPING, in the order of the caption and of the file names
ATTRIBUTES = {
    'Skin': 'face_color',
    'Hair length': 'hair',
    'Hair color': 'hair_color',
    'Glasses': 'glasses',
    'Beard': 'facial_hair',
}
config = Config()
models = {}     # experiment -> (model, scheduler, config, vocabulary): every experiment is loaded only once


def caption_and_warning(*words):
    """Compose the caption of the chosen words, as in training, and warn about a held-out combination.

    Args:
        words: the words chosen in the menus, in the order of ATTRIBUTES.

    Returns:
        (caption, warning text: empty for a combination seen in training).
    """
    chosen = dict(zip(ATTRIBUTES.values(), words))
    caption = CaptionGenerator(config).compose_caption(chosen)
    warning = '**⚠ Combination never seen in training (OOD).**' if is_held_out(chosen, config) else ''
    return caption, warning


def file_name(words, seed, guidance, text_conditioning):
    """Name of a saved image, with everything that decides it, so that no two settings overwrite each other.

    Args:
        words: the words chosen in the menus, in the order of ATTRIBUTES.
        seed: seed of the image.
        guidance: strength of classifier-free guidance.
        text_conditioning: False for the baseline, which ignores the words and the guidance.

    Returns:
        e.g. 'pale_long_blonde_no-glasses_a-beard_guidance3_seed7.png', or 'unconditional_seed7.png'.
    """
    if not text_conditioning:
        return f'unconditional_seed{seed}.png'
    return '_'.join(word.replace(' ', '-') for word in words) + f'_guidance{guidance:g}_seed{seed}.png'


def on_generate(experiment, seed, count, guidance, save, *words):
    """Generate the images, show them and, if asked, save them in the folder of the experiment.

    Args:
        experiment: name of a folder of RUNS_DIR, e.g. 'conditional_32'.
        seed: seed of the first image; the others get seed + 1, seed + 2, ...
        count: number of images.
        guidance: strength of classifier-free guidance.
        save: True to save every image in runs/<experiment>/generated/.
        words: the words chosen in the menus, in the order of ATTRIBUTES.

    Returns:
        (list of (image enlarged 4 times, 'seed N'), message with the time and the folder).
    """
    if not experiment:
        return [], 'No trained experiment in runs/: run train.py first.'
    if experiment not in models:
        models[experiment] = load_model(config.RUNS_DIR / experiment)
    model, scheduler, run_config, vocabulary = models[experiment]
    caption, _ = caption_and_warning(*words)
    seeds = [int(seed) + i for i in range(int(count))]
    start = time.time()
    images = generate(model, scheduler, run_config, vocabulary, caption, seeds, guidance)
    # 4 times bigger, keeping the real pixels (no smoothing), as in the control grids
    images = [image.resize((image.width * 4, image.height * 4), Image.NEAREST) for image in images]
    message = f'{len(images)} images in {time.time() - start:.0f} s.'
    if not run_config.TEXT_CONDITIONING:
        message += ' Baseline without text: the attributes and the guidance are ignored.'
    if save:
        folder = config.RUNS_DIR / experiment / 'generated'
        folder.mkdir(exist_ok=True)
        for image, image_seed in zip(images, seeds):
            image.save(folder / file_name(words, image_seed, guidance, run_config.TEXT_CONDITIONING))
        message += f' Saved in {folder}.'
    return [(image, f'seed {image_seed}') for image, image_seed in zip(images, seeds)], message


def build_page():
    """The page: experiment, attributes, prompt, settings, button and the generated images.

    Returns:
        The Gradio page (gr.Blocks), ready to launch.
    """
    experiments = [folder.name for folder in run_folders(config)]
    with gr.Blocks(title='Avatar Diffusion') as page:
        gr.Markdown('# Avatar Diffusion\nChoose the attributes and the settings, then press **Generate**.')
        experiment = gr.Dropdown(experiments, value=experiments[0] if experiments else None,
                                 label='Experiment (folders of runs/)')
        with gr.Row():
            menus = [gr.Dropdown(list(config.MAPPING[attribute]), value=list(config.MAPPING[attribute])[0], label=label)
                     for label, attribute in ATTRIBUTES.items()]
        caption = gr.Textbox(label='Prompt', interactive=False)
        warning = gr.Markdown()
        with gr.Row():
            seed = gr.Number(value=0, precision=0, label='Seed (of the first image)')
            count = gr.Slider(1, 8, value=4, step=1, label='Images')
            guidance = gr.Slider(1, 7, value=config.GUIDANCE_SCALE, step=0.5, label='Guidance')
        save = gr.Checkbox(value=True, label='Save the images in the folder of the experiment')
        button = gr.Button('Generate', variant='primary')
        # png: the default webp would blur the enlarged pixels and change their colors
        gallery = gr.Gallery(label='Generated images', columns=4, format='png')
        message = gr.Markdown()

        # the prompt follows the menus; it is also written when the page opens
        for menu in menus:
            menu.change(caption_and_warning, inputs=menus, outputs=[caption, warning])
        page.load(caption_and_warning, inputs=menus, outputs=[caption, warning])
        button.click(on_generate, inputs=[experiment, seed, count, guidance, save, *menus], outputs=[gallery, message])
    return page


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web demo of the cartoon diffusion model.')
    parser.add_argument('--share', action='store_true', help='also create a public link (e.g. on Colab)')
    build_page().launch(share=parser.parse_args().share)

"""Web demo: choose the attributes, the seed and the experiment, then generate and look at the avatars
and at what the attribute classifier sees in them.

Run it with:  python app.py            (then open the address it prints, e.g. http://127.0.0.1:7860)
On Colab:     python app.py --share    (it also prints a public link, e.g. to show the demo at the exam)
"""

import argparse
import time

import gradio as gr
import torch
from PIL import Image

from config import Config
from generate import generate, is_held_out, load_model, run_folders
from models.attribute_classifier import load_classifier, predict
from preprocessing.caption_generator import CaptionGenerator
from preprocessing.cartoon_dataset import attribute_words
from preprocessing.image_preprocessor import ImagePreprocessor

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
classifiers = {}    # image size -> AttributeClassifier, the judge of the images of that size (None if not trained)

# ---- Look of the page: warm paper, ginger (a hair color of the dataset) for the actions, and the generated
# avatars on the pastel tiles of the illustration of the assignment ----
PAPER, CARD, LINE, INK, MUTED, GINGER = '#FAF6EF', '#FFFDF9', '#E6DBCB', '#3B2F2A', '#6E5F54', '#BF521D'
COLORS = {
    'body_background_fill': PAPER, 'body_text_color': INK, 'body_text_color_subdued': MUTED,
    'background_fill_primary': CARD, 'background_fill_secondary': PAPER, 'block_background_fill': 'transparent',
    'block_border_color': LINE, 'border_color_primary': LINE, 'block_label_text_color': MUTED,
    'block_title_text_color': INK, 'block_label_background_fill': 'transparent',
    'input_background_fill': PAPER, 'input_border_color': LINE,
    'button_primary_background_fill': GINGER, 'button_primary_background_fill_hover': '#A8461A',
    'button_primary_text_color': '#FFFFFF', 'button_primary_border_color': GINGER,
    'color_accent_soft': '#F6DFD2', 'border_color_accent': GINGER, 'slider_color': GINGER,
    'checkbox_background_color_selected': GINGER, 'checkbox_border_color_selected': GINGER,
    'link_text_color': '#4F6F92',
}
THEME = gr.themes.Base(font=[gr.themes.GoogleFont('Atkinson Hyperlegible'), 'system-ui', 'sans-serif'],
                       radius_size=gr.themes.sizes.radius_lg).set(
    color_accent=GINGER, block_border_width='0px', block_shadow='none',
    **COLORS, **{name + '_dark': value for name, value in COLORS.items()})   # the same paper in dark mode
# the display face of the title (the body face comes with the theme)
HEAD = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque'
        ':opsz,wght@12..96,700;12..96,800&display=swap">')
CSS = f"""
#title h1 {{ font-family: 'Bricolage Grotesque', system-ui, sans-serif; font-weight: 800; letter-spacing: -0.02em;
            font-size: clamp(2.2rem, 4.5vw, 3.4rem); line-height: 1.02; color: {INK}; margin: 0 0 0.5rem; }}
#title p {{ max-width: 62ch; font-size: 1.05rem; line-height: 1.5; color: {MUTED}; margin: 0; }}
#controls {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 24px; padding: 20px; }}
/* Gradio paints the groups of fields with the border color: inside the card they stay transparent */
#controls .form {{ background: none; border: none; box-shadow: none; gap: 14px; }}
.ood {{ display: inline-block; background: #F6D58E; color: {INK}; border-radius: 999px; padding: 4px 12px;
        font-weight: 700; }}
/* the one bold element: every avatar on a pastel tile, the colors of the circles in the assignment;
   every tile sits in its own wrapper, so the colors alternate on the wrappers */
#avatars .grid-container {{ grid-template-rows: none; grid-auto-rows: auto; }}
/* Gradio keeps the gallery at least 450 px high: only the room for an empty gallery here */
#avatars .fixed-height {{ min-height: 180px; max-height: none; }}
@media (max-width: 575px) {{ #avatars .grid-container {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
#avatars .thumbnail-item {{ aspect-ratio: auto; height: auto; padding: 12px 12px 36px; border: none;
                            border-radius: 26px; box-shadow: none; background: #CFE2F3; }}
#avatars .grid-container > :nth-child(6n+2) .thumbnail-item {{ background: #F8D5C6; }}
#avatars .grid-container > :nth-child(6n+3) .thumbnail-item {{ background: #D7E7C6; }}
#avatars .grid-container > :nth-child(6n+4) .thumbnail-item {{ background: #F9E3A9; }}
#avatars .grid-container > :nth-child(6n+5) .thumbnail-item {{ background: #DDD7EF; }}
#avatars .grid-container > :nth-child(6n+6) .thumbnail-item {{ background: #C3E5DD; }}
#avatars .thumbnail-lg > img {{ aspect-ratio: 1 / 1; height: auto; border-radius: 16px; background: #FFFFFF;
                               image-rendering: pixelated; }}
/* the seed and the score under the avatar, on the tile: nothing covers the face or the beard */
#avatars .caption-label, #avatars .thumbnail-lg:hover .caption-label {{
    left: 50%; right: auto; bottom: 8px; transform: translateX(-50%); opacity: 1; border: none;
    background: none; color: {INK}; font-weight: 700; padding: 0; }}
#verdict p {{ color: {MUTED}; }}
/* on a narrow screen the table scrolls sideways instead of breaking its words */
#verdict table {{ border-collapse: collapse; font-size: 0.95rem; border: none; display: block; overflow-x: auto; }}
#verdict th, #verdict td {{ white-space: nowrap; }}
#verdict tr {{ border: none; }}
#verdict th {{ text-align: left; color: {MUTED}; background: none; border: none; border-bottom: 2px solid {LINE};
              padding: 6px 10px; }}
#verdict td {{ border: none; border-bottom: 1px solid {LINE}; padding: 6px 10px; }}
#verdict tr:nth-child(even) td {{ background: #F4EDE2; }}
#verdict .ok {{ color: #2E7D4F; font-weight: 700; }}
#verdict .no {{ color: #B3261E; font-weight: 700; }}
#message {{ color: {MUTED}; font-size: 0.9rem; }}
"""


def caption_and_warning(*words):
    """Compose the caption of the chosen words, as in training, and warn about a held-out combination.

    Args:
        words: the words chosen in the menus, in the order of ATTRIBUTES.

    Returns:
        (caption, warning text: empty for a combination seen in training).
    """
    chosen = dict(zip(ATTRIBUTES.values(), words))
    caption = CaptionGenerator(config).compose_caption(chosen)
    held_out = '<span class="ood">These attributes were never seen together in training (OOD)</span>'
    warning = held_out if is_held_out(chosen, config) else ''
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


def verdict_table(chosen, seen, seeds):
    """Table of what the attribute classifier sees in every image: ✓ where it is the word of the menu, ✗ where not.

    Args:
        chosen: {attribute: word} chosen in the menus.
        seen: one {attribute: word} per image, the words the classifier sees in it.
        seeds: the seed of every image, in the same order as seen.

    Returns:
        One line that explains the table, then a Markdown table with one row per image, e.g.
        '| 7 | ✓ dark | ✗ medium | ✓ black | ✓ glasses | ✓ a beard |' (the ticks and crosses in a colored span).
    """
    rows = ['The attribute classifier reads each avatar: ✓ the word of the menu, ✗ a different one.', '',
            '| Seed | ' + ' | '.join(ATTRIBUTES) + ' |', '|---' * (len(ATTRIBUTES) + 1) + '|']
    for words, image_seed in zip(seen, seeds):
        # by the name of the attribute, never by position: the menus and MAPPING have different orders
        cells = [('<span class="ok">✓</span> ' if words[attribute] == chosen[attribute]
                  else '<span class="no">✗</span> ') + words[attribute] for attribute in ATTRIBUTES.values()]
        rows.append(f'| {image_seed} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def on_generate(experiment, seed, count, guidance, save, *words):
    """Generate the images, judge them with the attribute classifier, show them and, if asked, save them.

    Args:
        experiment: name of a folder of RUNS_DIR, e.g. 'conditional_32'.
        seed: seed of the first image; the others get seed + 1, seed + 2, ...
        count: number of images.
        guidance: strength of classifier-free guidance.
        save: True to save every image in runs/<experiment>/generated/.
        words: the words chosen in the menus, in the order of ATTRIBUTES.

    Returns:
        (list of (image enlarged 4 times, 'seed N · right words/5'), table of what the classifier sees
        (empty without a classifier), message with the time and the folder).
    """
    if not experiment:
        return [], '', 'No trained experiment in runs/: run train.py first.'
    if experiment not in models:
        models[experiment] = load_model(config.RUNS_DIR / experiment)
    model, scheduler, run_config, vocabulary = models[experiment]
    caption, _ = caption_and_warning(*words)
    seeds = [int(seed) + i for i in range(int(count))]
    start = time.time()
    images = generate(model, scheduler, run_config, vocabulary, caption, seeds, guidance)

    # the judge of the size of this experiment; while it is missing, it is looked for again at every click
    size = run_config.IMAGE_SIZE
    if classifiers.get(size) is None:
        classifiers[size] = load_classifier(config, size)
    chosen = dict(zip(ATTRIBUTES.values(), words))
    captions = [f'seed {image_seed}' for image_seed in seeds]
    table = ''
    if classifiers[size] is not None:
        # the images as the classifier saw them in training: (count, 3, size, size) in [-1, 1]
        pixels = torch.stack([ImagePreprocessor.normalize(image) for image in images])
        seen = [attribute_words(row, config) for row in predict(classifiers[size], pixels, config).tolist()]
        table = verdict_table(chosen, seen, seeds)
        captions = [f'{text} · {sum(found[a] == chosen[a] for a in chosen)}/5' for text, found in zip(captions, seen)]

    # 4 times bigger, keeping the real pixels (no smoothing), as in the control grids
    images = [image.resize((image.width * 4, image.height * 4), Image.NEAREST) for image in images]
    message = f'{len(images)} images in {time.time() - start:.0f} s.'
    if not run_config.TEXT_CONDITIONING:
        message += ' Baseline without text: the attributes and the guidance are ignored.'
    if classifiers[size] is None:
        message += f' No attribute classifier for {size}x{size}: run train_classifier.py with IMAGE_SIZE = {size}.'
    if save:
        folder = config.RUNS_DIR / experiment / 'generated'
        folder.mkdir(exist_ok=True)
        for image, image_seed in zip(images, seeds):
            image.save(folder / file_name(words, image_seed, guidance, run_config.TEXT_CONDITIONING))
        message += f' Saved in {folder}.'
    return list(zip(images, captions)), table, message


def build_page():
    """The page: experiment, attributes, prompt, settings, button and the generated images.

    Returns:
        The Gradio page (gr.Blocks), ready to launch.
    """
    experiments = [folder.name for folder in run_folders(config)]
    labels = list(ATTRIBUTES.items())
    with gr.Blocks(title='Avatar Diffusion') as page:
        gr.Markdown('# Avatar Diffusion\nChoose five attributes and a seed. A small diffusion model, trained from '
                    'scratch on the Google Cartoon Set, draws the avatars; the attribute classifier then checks each '
                    'one against the prompt.', elem_id='title')
        with gr.Row(equal_height=False):
            # left: everything the user chooses
            with gr.Column(scale=5, min_width=320, elem_id='controls'):
                experiment = gr.Dropdown(experiments, value=experiments[0] if experiments else None,
                                         label='Experiment (a trained model in runs/)')
                menus = []      # in the order of ATTRIBUTES, as the inputs of the callbacks expect
                for group in [labels[:3], labels[3:]]:
                    with gr.Row():
                        menus += [gr.Dropdown(list(config.MAPPING[attribute]), value=list(config.MAPPING[attribute])[0],
                                              label=label, min_width=120) for label, attribute in group]
                caption = gr.Textbox(label='Prompt', interactive=False, lines=2)
                # sanitize_html=False keeps the class of the badge: the text is ours, never typed by the user
                warning = gr.Markdown(sanitize_html=False, elem_id='warning')
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label='Seed of the first image')
                    count = gr.Slider(1, 8, value=4, step=1, label='Number of images')
                guidance = gr.Slider(1, 7, value=config.GUIDANCE_SCALE, step=0.5, label='Guidance')
                save = gr.Checkbox(value=True, label='Save the images in runs/<experiment>/generated/')
                button = gr.Button('Generate avatars', variant='primary', size='lg')
            # right: the avatars and what the classifier sees in them
            with gr.Column(scale=7, min_width=320):
                # png: the default webp would blur the enlarged pixels and change their colors
                gallery = gr.Gallery(label='Generated avatars', show_label=False, columns=4, format='png',
                                     height='auto', elem_id='avatars')
                # what the attribute classifier sees in every image (the colored ticks need their class)
                verdict = gr.Markdown(sanitize_html=False, elem_id='verdict')
                message = gr.Markdown(elem_id='message')

        # the prompt follows the menus; it is also written when the page opens
        for menu in menus:
            menu.change(caption_and_warning, inputs=menus, outputs=[caption, warning])
        page.load(caption_and_warning, inputs=menus, outputs=[caption, warning])
        button.click(on_generate, inputs=[experiment, seed, count, guidance, save, *menus],
                     outputs=[gallery, verdict, message])
    return page


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web demo of the cartoon diffusion model.')
    parser.add_argument('--share', action='store_true', help='also create a public link (e.g. on Colab)')
    # in Gradio 6 the theme, the style and the fonts go to launch(), not to gr.Blocks()
    build_page().launch(share=parser.parse_args().share, theme=THEME, css=CSS, head=HEAD)

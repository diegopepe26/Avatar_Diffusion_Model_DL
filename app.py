"""Web demo: choose the attributes, the seed and the experiment, then generate and look at the avatars
and at what the attribute classifier sees in them.

Run it with:  python app.py            (then open the address it prints, e.g. http://127.0.0.1:7860)
On Colab:     python app.py --share    (it also prints a public link, e.g. to show the demo at the exam)
"""

import argparse
import time
from functools import partial

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
ICONS = Config.ROOT / 'assets' / 'icons'     # one SVG per word of MAPPING, for the cards and the options
COUNTS = [1, 2, 4, 8]                         # the choices of the number of images
GUIDANCES = [1.0, 2.0, 3.0, 5.0, 7.0]         # the choices of the guidance: 1 = no push towards the caption

# ---- Look of the page: two panels on one white sheet, black for what is chosen and for the action ----
INK, MUTED, LINE, PANEL, PAGE = '#1F1F1F', '#6F6C66', '#E4E2DC', '#F5F4F1', '#ECEBE7'
COLORS = {
    'body_background_fill': PAGE, 'body_text_color': INK, 'body_text_color_subdued': MUTED,
    'background_fill_primary': '#FFFFFF', 'background_fill_secondary': PANEL, 'block_background_fill': 'transparent',
    'block_border_color': LINE, 'border_color_primary': LINE, 'block_label_text_color': INK,
    'block_title_text_color': INK, 'block_label_background_fill': 'transparent', 'block_info_text_color': MUTED,
    'input_background_fill': '#FFFFFF', 'input_border_color': LINE,
    'button_primary_background_fill': INK, 'button_primary_background_fill_hover': '#3A3A38',
    'button_primary_text_color': '#FFFFFF', 'button_primary_border_color': INK,
    'button_secondary_background_fill': '#FFFFFF', 'button_secondary_background_fill_hover': '#FAFAF8',
    'button_secondary_text_color': INK, 'button_secondary_border_color': LINE,
    'color_accent_soft': '#E9E8E4', 'border_color_accent': INK, 'slider_color': INK,
    'checkbox_background_color_selected': INK, 'checkbox_border_color_selected': INK, 'link_text_color': INK,
}
THEME = gr.themes.Base(font=[gr.themes.GoogleFont('Geist'), 'system-ui', 'sans-serif'],
                       radius_size=gr.themes.sizes.radius_lg).set(
    color_accent=INK, block_border_width='0px', block_shadow='none',
    **COLORS, **{name + '_dark': value for name, value in COLORS.items()})   # the same look in dark mode
# the name of each attribute, written on its card above the chosen word
CARD_NAMES = '\n'.join(f'#card-{attribute}::before {{ content: "{label}"; }}' for label, attribute in ATTRIBUTES.items())
CSS = f"""
.gradio-container {{ width: 100% !important; max-width: 1120px !important; margin: 0 auto !important; }}
#brand h1 {{ font-size: 1.3rem; font-weight: 600; letter-spacing: -0.01em; margin: 4px 0 2px; color: {INK}; }}
#brand p {{ color: {MUTED}; margin: 0 0 6px; font-size: 0.95rem; }}
#shell {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 28px; overflow: hidden; gap: 0;
          box-shadow: 0 18px 48px rgba(31, 31, 31, 0.07); }}
#choose {{ background: {PANEL}; padding: 28px; gap: 18px; border-right: 1px solid {LINE}; }}
#results {{ padding: 28px; gap: 18px; }}
/* Gradio paints the groups of fields with the border color: on the panels they stay transparent */
#choose .form, #results .form {{ background: none; border: none; box-shadow: none; gap: 18px; }}
/* the fields without their inner margin: every piece of a panel starts on the same left edge */
#shell .block.padded, #shell .html-container {{ padding: 0; }}
#warning:not(:has(.ood)) {{ display: none; }}
.step {{ display: flex; align-items: center; gap: 12px; }}
.step .badge {{ background: {INK}; color: #FFFFFF; font-size: 0.75rem; font-weight: 600; border-radius: 999px;
               padding: 3px 9px; }}
.step h2 {{ font-size: 1.25rem; font-weight: 600; letter-spacing: -0.01em; margin: 0; }}
.hint {{ color: {MUTED}; font-size: 0.92rem; line-height: 1.5; margin: 6px 0 0; max-width: 52ch; }}
/* the five cards: the icon of the chosen word, the name of the attribute, the word. A card is never narrower
   than 86 px, the room of the longest word (sunglasses, 77 px): on a narrow panel the last cards go to a new row.
   If a word still does not fit (e.g. the text zoomed in), it breaks instead of sticking out of its card */
#cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(86px, 1fr)); gap: 8px; align-items: stretch; }}
#cards .card {{ flex-direction: column; gap: 2px; padding: 12px 4px 10px; min-width: 0; min-height: 124px;
               border: 1px solid {LINE}; border-radius: 18px; font-weight: 600; font-size: 0.88rem;
               overflow-wrap: anywhere; }}
/* Gradio crops the icons to fill their square (cover): the whole face is shown instead, even if not square */
#cards .card img {{ width: 52px; height: 52px; margin: 0 0 6px; order: -2; object-fit: contain; }}
#cards .card::before {{ order: -1; font-size: 0.74rem; font-weight: 500; color: {MUTED}; }}
{CARD_NAMES}
#cards .card:hover {{ border-color: #BDBAB3; }}
@media (max-width: 640px) {{ #choose, #results {{ padding: 20px 16px; }} }}
#prompt textarea {{ font-size: 0.95rem; color: {INK}; }}
.ood {{ display: inline-block; background: #F6D58E; color: {INK}; border-radius: 999px; padding: 4px 12px;
        font-size: 0.88rem; font-weight: 600; }}
/* the choices as pills: black when chosen, the round button of the radio hidden */
.pills .wrap {{ gap: 8px; }}
.pills label {{ border: 1px solid {LINE}; border-radius: 999px; background: #FFFFFF; padding: 6px 14px;
               min-width: 44px; justify-content: center; box-shadow: none; font-weight: 500; }}
.pills label input {{ position: absolute; opacity: 0; width: 1px; height: 1px; }}
.pills label:has(input:checked) {{ background: {INK}; border-color: {INK}; color: #FFFFFF; }}
.toggle label {{ border: 1px solid {LINE}; border-radius: 999px; background: #FFFFFF; padding: 6px 15px;
                width: fit-content; font-weight: 500; }}
.toggle label input {{ position: absolute; opacity: 0; width: 1px; height: 1px; }}
.toggle label:has(input:checked) {{ background: {INK}; border-color: {INK}; color: #FFFFFF; }}
.toggle label:has(input:checked) .label-text {{ color: #FFFFFF; }}
.toggle label:has(input:checked) .label-text::before {{ content: "✓  "; }}
/* Gradio puts the seed and the toggle in one .form box: the toggle goes down, level with the box of the seed */
#seed-row .form {{ align-items: flex-end; }}
#generate {{ border-radius: 16px; padding: 16px; font-size: 1rem; font-weight: 600; }}
button:focus-visible, label:has(input:focus-visible) {{ outline: 2px solid {INK}; outline-offset: 2px; }}
/* before the first avatars: an empty frame that says what to do */
.empty {{ border: 1.5px dashed #CFCCC5; border-radius: 22px; min-height: 380px; display: flex;
         flex-direction: column; align-items: center; justify-content: center; gap: 8px; text-align: center;
         color: {MUTED}; background: #FCFCFB; padding: 24px; }}
.empty .glyph {{ width: 56px; height: 56px; border-radius: 16px; background: #FFFFFF; border: 1px solid {LINE};
                display: grid; place-items: center; margin-bottom: 6px; }}
.empty strong {{ color: {INK}; font-weight: 600; font-size: 1rem; }}
/* every avatar on a pastel tile; every tile sits in its own wrapper, so the colors alternate on the wrappers */
#avatars .grid-container {{ grid-template-rows: none; grid-auto-rows: auto; }}
/* Gradio keeps the gallery at least 450 px high: only the room for the avatars here */
#avatars .fixed-height {{ min-height: 180px; max-height: none; }}
@media (max-width: 575px) {{ #avatars .grid-container {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
#avatars .thumbnail-item {{ aspect-ratio: auto; height: auto; padding: 12px 12px 36px; border: none;
                            border-radius: 22px; box-shadow: none; background: #CFE2F3; }}
#avatars .grid-container > :nth-child(6n+2) .thumbnail-item {{ background: #F8D5C6; }}
#avatars .grid-container > :nth-child(6n+3) .thumbnail-item {{ background: #D7E7C6; }}
#avatars .grid-container > :nth-child(6n+4) .thumbnail-item {{ background: #F9E3A9; }}
#avatars .grid-container > :nth-child(6n+5) .thumbnail-item {{ background: #DDD7EF; }}
#avatars .grid-container > :nth-child(6n+6) .thumbnail-item {{ background: #C3E5DD; }}
#avatars .thumbnail-lg > img {{ aspect-ratio: 1 / 1; height: auto; border-radius: 14px; background: #FFFFFF;
                               image-rendering: pixelated; }}
/* the seed and the score under the avatar, on the tile: nothing covers the face or the beard */
#avatars .caption-label, #avatars .thumbnail-lg:hover .caption-label {{
    left: 50%; right: auto; bottom: 8px; transform: translateX(-50%); opacity: 1; border: none;
    background: none; color: {INK}; font-weight: 600; padding: 0; }}
#verdict p {{ color: {MUTED}; }}
/* on a narrow screen the table scrolls sideways instead of breaking its words */
#verdict table {{ border-collapse: collapse; font-size: 0.93rem; border: none; display: block; overflow-x: auto; }}
#verdict th, #verdict td {{ white-space: nowrap; }}
#verdict tr {{ border: none; }}
#verdict th {{ text-align: left; color: {MUTED}; font-weight: 500; background: none; border: none;
              border-bottom: 1px solid {LINE}; padding: 6px 7px; }}
#verdict td {{ border: none; border-bottom: 1px solid {LINE}; padding: 7px 7px; }}
#verdict .ok {{ color: #2E7D4F; font-weight: 700; }}
#verdict .no {{ color: #B3261E; font-weight: 700; }}
#message {{ color: {MUTED}; font-size: 0.9rem; }}
/* the options of an attribute: a sheet over the darkened page, shown only when its card is clicked */
.overlay {{ position: fixed !important; inset: 0; z-index: 1000; display: flex !important; align-items: center;
           justify-content: center; padding: 16px; background: rgba(24, 24, 24, 0.38); }}
.overlay .sheet {{ background: #FFFFFF; border-radius: 24px; padding: 18px 20px 22px; width: min(600px, 100%);
                  flex-grow: 0 !important; gap: 14px; box-shadow: 0 24px 64px rgba(0, 0, 0, 0.18); }}
.sheet-head {{ align-items: center; justify-content: space-between; flex-wrap: nowrap; }}
.sheet-head h3 {{ margin: 0; font-size: 1.1rem; font-weight: 600; }}
.sheet-head .close {{ flex: 0 0 38px; min-width: 38px; width: 38px; height: 38px; padding: 0; border-radius: 999px; }}
.options {{ display: grid !important; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr)); gap: 10px; }}
.options .option {{ flex-direction: column; gap: 8px; padding: 14px 6px; min-width: 0; border-radius: 18px;
                   font-weight: 600; }}
.options .option img {{ width: 56px; height: 56px; margin: 0; object-fit: contain; }}
/* the chosen word: a black outline, not a black fill, so its icon stays readable */
.options .option.primary {{ background: #FFFFFF; color: {INK}; border: 2px solid {INK}; }}
"""
# the empty frame of the results, with a small picture icon
EMPTY = ('<div class="empty"><div class="glyph"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
         f'stroke="{INK}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
         '<rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="2"/>'
         '<path d="M21 16l-5-5-8 9"/></svg></div><strong>Your avatars will appear here</strong>'
         '<span>Choose the attributes on the left and press Generate avatars.</span></div>')


def step(number, title, hint):
    """The heading of one of the two steps of the page.

    Args:
        number: '01' or '02', written in a black badge.
        title: e.g. 'Choose your avatar'.
        hint: one line under the title, about what to do.

    Returns:
        HTML text.
    """
    return f'<div class="step"><span class="badge">{number}</span><h2>{title}</h2></div><p class="hint">{hint}</p>'


def icon_file(attribute, word):
    """The SVG icon of a word, shown on its card and among the options.

    Args:
        attribute: an attribute of MAPPING, e.g. 'glasses'.
        word: one of its words, e.g. 'no glasses'.

    Returns:
        Path of the icon, e.g. assets/icons/glasses_no-glasses.svg.
    """
    return ICONS / f'{attribute}_{word.replace(" ", "-")}.svg'


def caption_and_warning(*words):
    """Compose the caption of the chosen words, as in training, and warn about a held-out combination.

    Args:
        words: the words chosen on the cards, in the order of ATTRIBUTES.

    Returns:
        (caption, warning text: empty for a combination seen in training).
    """
    chosen = dict(zip(ATTRIBUTES.values(), words))
    caption = CaptionGenerator(config).compose_caption(chosen)
    held_out = '<span class="ood">Never seen together in training (OOD)</span>'
    warning = held_out if is_held_out(chosen, config) else ''
    return caption, warning


def choose(attribute, word, words):
    """The words of the page after choosing one word of one attribute, with the new prompt and warning.

    Args:
        attribute: the attribute of the option clicked, e.g. 'hair'.
        word: the word chosen, e.g. 'long'.
        words: the words of the page before the click, in the order of ATTRIBUTES.

    Returns:
        (the new words in the order of ATTRIBUTES, caption, warning).
    """
    # by the name of the attribute, never by position: the page and MAPPING list the attributes in different orders
    new_words = tuple(word if name == attribute else old for name, old in zip(ATTRIBUTES.values(), words))
    return (new_words, *caption_and_warning(*new_words))


def file_name(words, seed, guidance, text_conditioning):
    """Name of a saved image, with everything that decides it, so that no two settings overwrite each other.

    Args:
        words: the words chosen on the cards, in the order of ATTRIBUTES.
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
    """Table of what the attribute classifier sees in every image: ✓ where it is the word of the card, ✗ where not.

    Args:
        chosen: {attribute: word} chosen on the cards.
        seen: one {attribute: word} per image, the words the classifier sees in it.
        seeds: the seed of every image, in the same order as seen.

    Returns:
        One line that explains the table, then a Markdown table with one row per image, e.g.
        '| 7 | ✓ dark | ✗ medium | ✓ black | ✓ glasses | ✓ a beard |' (the ticks and crosses in a colored span).
    """
    rows = ['The attribute classifier reads each avatar: ✓ the chosen word, ✗ a different one.', '',
            '| Seed | ' + ' | '.join(ATTRIBUTES) + ' |', '|---' * (len(ATTRIBUTES) + 1) + '|']
    for words, image_seed in zip(seen, seeds):
        # by the name of the attribute, never by position: the page and MAPPING have different orders
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
        words: the words chosen on the cards, in the order of ATTRIBUTES.

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


def on_option(attribute, word, *words):
    """Click on an option of a sheet: keep the word, show it on the card, close the sheet, rewrite the prompt.

    Args:
        attribute, word: the attribute of the sheet and the word clicked.
        words: the words of the page before the click, in the order of ATTRIBUTES.

    Returns:
        (the 5 words, the card, the sheet, caption, warning, the options of the sheet with the chosen one outlined).
    """
    new_words, caption, warning = choose(attribute, word, words)
    options = [gr.Button(variant='primary' if option == word else 'secondary') for option in config.MAPPING[attribute]]
    card = gr.Button(value=word, icon=str(icon_file(attribute, word)))
    return (*new_words, card, gr.Column(visible=False), caption, warning, *options)


def build_page():
    """The page: step 01 with the cards of the attributes and the settings, step 02 with the avatars, and one
    sheet of options per attribute, hidden until its card is clicked.

    Returns:
        The Gradio page (gr.Blocks), ready to launch.
    """
    experiments = [folder.name for folder in run_folders(config)]
    first = {attribute: list(config.MAPPING[attribute])[0] for attribute in ATTRIBUTES.values()}
    with gr.Blocks(title='Avatar Diffusion') as page:
        gr.HTML('<header id="brand"><h1>Avatar Diffusion</h1><p>A small text-to-image diffusion model, trained from '
                'scratch on the Google Cartoon Set.</p></header>')
        # the chosen words, in the order of ATTRIBUTES: the cards only show them
        words = [gr.State(first[attribute]) for attribute in ATTRIBUTES.values()]
        with gr.Row(elem_id='shell', equal_height=False):
            with gr.Column(scale=1, min_width=320, elem_id='choose'):
                gr.HTML(step('01', 'Choose your avatar', 'Pick the five attributes, then the model and the settings.'))
                with gr.Row(elem_id='cards'):
                    cards = {attribute: gr.Button(first[attribute], icon=str(icon_file(attribute, first[attribute])),
                                                  elem_id=f'card-{attribute}', elem_classes='card')
                             for attribute in ATTRIBUTES.values()}
                caption = gr.Textbox(label='Prompt', interactive=False, lines=2, elem_id='prompt')
                # sanitize_html=False keeps the class of the badge: the text is ours, never typed by the user
                warning = gr.Markdown(sanitize_html=False, elem_id='warning')
                experiment = gr.Radio(experiments, value=experiments[0] if experiments else None, label='Experiment',
                                      elem_classes='pills')
                count = gr.Radio(COUNTS, value=4, label='Images', elem_classes='pills')
                guidance = gr.Radio([(f'{value:g}', value) for value in GUIDANCES], value=config.GUIDANCE_SCALE,
                                    label='Guidance', elem_classes='pills')
                with gr.Row(elem_id='seed-row'):
                    seed = gr.Number(value=0, precision=0, label='Seed', min_width=120)
                    save = gr.Checkbox(value=True, label='Save images', elem_classes='toggle', min_width=120)
                button = gr.Button('Generate avatars', variant='primary', elem_id='generate')
            with gr.Column(scale=1, min_width=320, elem_id='results'):
                gr.HTML(step('02', 'Your avatars', 'The attribute classifier checks every avatar against the prompt.'))
                empty = gr.HTML(EMPTY)
                # png: the default webp would blur the enlarged pixels and change their colors
                # no buttons over the avatars: the images are saved with Save images
                gallery = gr.Gallery(label='Generated avatars', show_label=False, columns=4, format='png',
                                     height='auto', buttons=[], elem_id='avatars', visible=False)
                # what the attribute classifier sees in every image (the colored ticks need their class)
                verdict = gr.Markdown(sanitize_html=False, elem_id='verdict')
                message = gr.Markdown(elem_id='message')

        for label, attribute in ATTRIBUTES.items():
            with gr.Column(visible=False, elem_classes='overlay') as sheet:
                with gr.Column(elem_classes='sheet'):
                    with gr.Row(elem_classes='sheet-head'):
                        gr.HTML(f'<h3>{label}</h3>')
                        close = gr.Button('✕', elem_classes='close', min_width=0)
                    with gr.Row(elem_classes='options'):
                        options = [gr.Button(word, icon=str(icon_file(attribute, word)), elem_classes='option',
                                             variant='primary' if word == first[attribute] else 'secondary')
                                   for word in config.MAPPING[attribute]]
            cards[attribute].click(lambda: gr.Column(visible=True), None, sheet)
            close.click(lambda: gr.Column(visible=False), None, sheet)
            for word, option in zip(config.MAPPING[attribute], options):
                option.click(partial(on_option, attribute, word), inputs=words,
                             outputs=[*words, cards[attribute], sheet, caption, warning, *options])

        # the prompt is written when the page opens; the results replace the empty frame at the first click
        page.load(caption_and_warning, inputs=words, outputs=[caption, warning])
        button.click(lambda: (gr.HTML(visible=False), gr.Gallery(visible=True)), None, [empty, gallery]).then(
            on_generate, inputs=[experiment, seed, count, guidance, save, *words], outputs=[gallery, verdict, message])
    return page


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web demo of the cartoon diffusion model.')
    parser.add_argument('--share', action='store_true', help='also create a public link (e.g. on Colab)')
    # in Gradio 6 the theme and the style go to launch(), not to gr.Blocks()
    build_page().launch(share=parser.parse_args().share, theme=THEME, css=CSS)

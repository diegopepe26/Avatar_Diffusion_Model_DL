"""Web demo of the model. You choose the attributes, the model and the seed, then the page generates the avatars and
shows what the attribute classifier sees in them.

With --share it also creates a public link to the page.
"""

import argparse
import random
import time
from functools import partial

import gradio as gr
import torch
from PIL import Image

from config import Config
from generate import generate, is_held_out, load_model
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
models = {}     # folder of a model -> (model, scheduler, config, vocabulary), so every model is loaded only once
classifiers = {}    # image size -> AttributeClassifier, the judge of the images of that size (None if not trained)
ICONS = Config.ROOT / 'assets' / 'icons'     # one SVG per word of MAPPING, for the cards and the options
BASELINE_PROMPT = 'The baseline generates without text: the attributes and the guidance are ignored.'

COUNTS = [1, 2, 4, 8]                       # the choices of the number of images
GUIDANCES = [1.0, 2.0, 3.0, 5.0, 7.0]         # the choices of the guidance, where 1 means no push towards the caption

# the grid of the models has the size in the columns and the model with or without text in the rows. On its right,
# the number of training images chooses the folder of the runs (runs/ or runs_100/)
SIZES = [32, 64]
KINDS = {'Conditional': 'conditional', 'Unconditional': 'unconditional'}    # row of the grid -> start of the folder
DATASETS = {'10,000': Config.RUNS_DIR, '100,000': Config.ROOT / 'runs_100'}
FIRST_RUN = {'dataset': '10,000', 'kind': 'conditional', 'size': 32}    # the model chosen when the page opens

# the prompt is HTML, not a text field, so that the words of a held-out pair can be in bold
PROMPT = '<span class="prompt-label">Prompt</span><p class="prompt-text">${value}</p>'

# ---- Look of the page, two panels on one white sheet, black for what is chosen and for the main button ----
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
/* Gradio paints the groups of fields with the border color, here they stay transparent */
#choose .form, #results .form {{ background: none; border: none; box-shadow: none; gap: 18px; }}
/* no inner margin in the fields, so every piece of a panel starts on the same left edge */
#shell .block.padded, #shell .html-container {{ padding: 0; }}
#warning:not(:has(.ood)) {{ display: none; }}
.step {{ display: flex; align-items: center; gap: 12px; }}
.step .badge {{ background: {INK}; color: #FFFFFF; font-size: 0.75rem; font-weight: 600; border-radius: 999px;
               padding: 3px 9px; }}
.step h2 {{ font-size: 1.25rem; font-weight: 600; letter-spacing: -0.01em; margin: 0; }}
.hint {{ color: {MUTED}; font-size: 0.92rem; line-height: 1.5; margin: 6px 0 0; max-width: 52ch; }}
/* the five cards show the icon of the chosen word, the name of the attribute and the word. A card is at least
   86 px wide, enough for the longest word (sunglasses), so on a narrow panel the last cards go to a new row.
   A word that still does not fit breaks instead of going out of its card */
#cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(86px, 1fr)); gap: 8px; align-items: stretch; }}
#cards .card {{ flex-direction: column; gap: 2px; padding: 12px 4px 10px; min-width: 0; min-height: 124px;
               border: 1px solid {LINE}; border-radius: 18px; font-weight: 600; font-size: 0.88rem;
               overflow-wrap: anywhere; }}
/* Gradio crops the icons to fill their square, here the whole face is shown instead */
#cards .card img {{ width: 52px; height: 52px; margin: 0 0 6px; order: -2; object-fit: contain; }}
#cards .card::before {{ order: -1; font-size: 0.74rem; font-weight: 500; color: {MUTED}; }}
{CARD_NAMES}
#cards .card:hover {{ border-color: #BDBAB3; }}
/* the controls that the baseline ignores, and the models never trained, stay in place but faded */
#cards .card:disabled, #randoms button:disabled, .pills label:has(input:disabled), #models button:disabled {{
    opacity: 0.4; cursor: not-allowed; }}
@media (max-width: 640px) {{ #choose, #results {{ padding: 20px 16px; }} }}
#randoms {{ gap: 8px; flex-wrap: wrap; }}
#randoms button {{ flex: 0 0 auto; border-radius: 999px; font-weight: 500; padding: 5px 14px; }}
/* the prompt looks like a text field, with a label and a white box */
#prompt .prompt-label {{ display: inline-block; font-size: 14px; color: {INK}; margin-bottom: 8px; }}
#prompt .prompt-text {{ background: #FFFFFF; border-radius: 6px; padding: 10px; margin: 0; font-size: 0.95rem;
                       line-height: 1.4; color: {INK}; min-height: calc(2.8em + 20px); }}
#prompt .prompt-text strong {{ font-weight: 700; }}
.ood {{ display: inline-block; background: #F6D58E; color: {INK}; border-radius: 999px; padding: 4px 12px;
        font-size: 0.88rem; font-weight: 600; }}
/* the badge and its note on one line; on a narrow panel the whole note goes under the badge */
#warning p {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; }}
.ood-note {{ color: {MUTED}; font-size: 0.88rem; }}
/* the models are a grid of pills, with the sizes in the columns and the model with or without text in the rows.
   On its right, after a thin line, there is the number of training images. The chosen one is black, as in the
   other pills */
#models {{ gap: 14px; flex-wrap: nowrap; }}
#models > .column {{ display: grid; gap: 8px; align-items: center; flex: 0 1 auto; }}
#models #grid {{ grid-template-columns: auto auto auto; }}
#models #grid > :first-child {{ grid-column: 1 / -1; }}
#models #data {{ border-left: 1px solid {LINE}; padding-left: 14px; }}
#models .head {{ display: inline-block; font-size: 14px; color: {INK}; }}
#models .kind {{ font-size: 0.88rem; font-weight: 500; color: {MUTED}; padding-right: 4px; }}
#models button {{ border: 1px solid {LINE}; border-radius: 999px; padding: 6px 14px; font-size: 14px;
                 font-weight: 500; line-height: 21px; white-space: nowrap; }}
#models button.primary {{ border-color: {INK}; }}
/* the choices look like pills, black when chosen, and the round button of the radio is hidden */
.pills .wrap {{ gap: 8px; }}
.pills label {{ border: 1px solid {LINE}; border-radius: 999px; background: #FFFFFF; padding: 6px 14px;
               min-width: 44px; justify-content: center; box-shadow: none; font-weight: 500; }}
.pills label input {{ position: absolute; opacity: 0; width: 1px; height: 1px; }}
/* Gradio leaves room on the left of the text for the round button of the radio, which is hidden here */
.pills label span {{ margin-left: 0; }}
.pills label:has(input:checked) {{ background: {INK}; border-color: {INK}; color: #FFFFFF; }}
.toggle label {{ border: 1px solid {LINE}; border-radius: 999px; background: #FFFFFF; padding: 6px 15px;
                width: fit-content; font-weight: 500; }}
.toggle label input {{ position: absolute; opacity: 0; width: 1px; height: 1px; }}
.toggle label:has(input:checked) {{ background: {INK}; border-color: {INK}; color: #FFFFFF; }}
.toggle label:has(input:checked) .label-text {{ color: #FFFFFF; }}
.toggle label:has(input:checked) .label-text::before {{ content: "✓  "; }}
/* the Random seed button is a pill like the toggle next to it */
#random-seed {{ flex: 0 0 auto; border: 1px solid {LINE}; border-radius: 999px; padding: 6px 15px; font-size: 14px;
                font-weight: 500; }}
/* the button and the toggle go down, level with the box of the seed */
#seed-row, #seed-row .form {{ align-items: flex-end; }}
#generate {{ border-radius: 16px; padding: 16px; font-size: 1rem; font-weight: 600; }}
button:focus-visible, label:has(input:focus-visible) {{ outline: 2px solid {INK}; outline-offset: 2px; }}
/* before the first avatars, an empty frame says what to do */
.empty {{ border: 1.5px dashed #CFCCC5; border-radius: 22px; min-height: 380px; display: flex;
         flex-direction: column; align-items: center; justify-content: center; gap: 8px; text-align: center;
         color: {MUTED}; background: #FCFCFB; padding: 24px; }}
.empty .glyph {{ width: 56px; height: 56px; border-radius: 16px; background: #FFFFFF; border: 1px solid {LINE};
                display: grid; place-items: center; margin-bottom: 6px; }}
.empty strong {{ color: {INK}; font-weight: 600; font-size: 1rem; }}
/* every avatar on a pastel tile; every tile sits in its own wrapper, so the colors alternate on the wrappers */
#avatars .grid-container {{ grid-template-rows: none; grid-auto-rows: auto; }}
/* Gradio keeps the gallery at least 450 px high and always shows a scroll bar on its side. Here the gallery takes
   only the room of the avatars, without the scroll bar */
#avatars .fixed-height {{ min-height: 180px; max-height: none; overflow: visible; }}
@media (max-width: 575px) {{ #avatars .grid-container {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
#avatars .thumbnail-lg {{ aspect-ratio: auto; height: auto; padding: 12px 12px 36px; border: none;
                          border-radius: 22px; box-shadow: none; background: #CFE2F3; }}
#avatars .grid-container > :nth-child(6n+2) .thumbnail-item {{ background: #F8D5C6; }}
#avatars .grid-container > :nth-child(6n+3) .thumbnail-item {{ background: #D7E7C6; }}
#avatars .grid-container > :nth-child(6n+4) .thumbnail-item {{ background: #F9E3A9; }}
#avatars .grid-container > :nth-child(6n+5) .thumbnail-item {{ background: #DDD7EF; }}
#avatars .grid-container > :nth-child(6n+6) .thumbnail-item {{ background: #C3E5DD; }}
#avatars .thumbnail-lg > img {{ aspect-ratio: 1 / 1; height: auto; border-radius: 14px; background: #FFFFFF;
                               image-rendering: pixelated; }}
/* the seed and the score go under the avatar, on the tile, so they never cover the face or the beard */
#avatars .caption-label, #avatars .thumbnail-lg:hover .caption-label {{
    left: 50%; right: auto; bottom: 8px; transform: translateX(-50%); opacity: 1; border: none;
    background: none; color: {INK}; font-weight: 600; padding: 0; }}
/* a click on an avatar opens it big over the others. Gradio gives this preview only the height of the grid, so
   while it is open the gallery is 1.2 times as tall as it is wide and the big avatar takes the whole width */
#avatars .gallery-container:has(.preview) .grid-wrap {{ aspect-ratio: 5 / 6; }}
#avatars .preview img {{ image-rendering: pixelated; }}
/* a plain white sheet behind the big avatar, so the colored tiles do not show through */
#avatars .preview::before {{ opacity: 1; }}
#verdict p {{ color: {MUTED}; }}
/* small text and cells, without the 10 px that Gradio puts around a table, so the widest row fits in the panel
   of a computer screen. On a narrower screen the table scrolls sideways instead of breaking its words */
#verdict table {{ border-collapse: collapse; font-size: 0.85rem; border: none; padding: 0; display: block;
                 overflow-x: auto; width: fit-content; max-width: 100%; margin-left: auto; margin-right: auto; }}
#verdict th, #verdict td {{ white-space: nowrap; }}
#verdict tr {{ border: none; }}
#verdict th {{ text-align: left; color: {MUTED}; font-weight: 500; background: none; border: none;
              border-bottom: 1px solid {LINE}; padding: 6px 5px; }}
#verdict td {{ border: none; border-bottom: 1px solid {LINE}; padding: 7px 5px; }}
#verdict .ok {{ color: #2E7D4F; font-weight: 700; }}
#verdict .no {{ color: #B3261E; font-weight: 700; }}
#message {{ color: {MUTED}; font-size: 0.9rem; }}
/* the options of an attribute are in a sheet over the darkened page, shown only when its card is clicked */
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
/* the chosen word has a black outline, not a black fill, so its icon stays readable */
.options .option.primary {{ background: #FFFFFF; color: {INK}; border: 2px solid {INK}; }}
"""

# side by side, the grid of the models and the training images need about 380 px. On a narrower panel (e.g. on a
# phone) the training images go under the grid, on one row and without the line. These rules measure the panel, not
# the screen (@container), and Gradio drops them from the CSS, so they go in the head of the page
HEAD = """<style>
#choose { container-type: inline-size; }
@container (max-width: 390px) {
    #choose #models { flex-direction: column; }
    #choose #models #data { border-left: none; padding-left: 0; grid-template-columns: auto auto auto; }
}
</style>"""

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
    """Compose the caption of the chosen words, as in training, and warn about a held-out combination, whose
    words are in bold in the caption.

    Args:
        words: the words chosen on the cards, in the order of ATTRIBUTES.

    Returns:
        (caption, warning text, empty for a combination seen in training), e.g. for blonde hair and sunglasses
        'a cartoon avatar with pale skin, long <strong>blonde</strong> hair, <strong>sunglasses</strong> and a beard'.
    """
    chosen = dict(zip(ATTRIBUTES.values(), words))

    # the attributes of the held-out pairs found among the words, e.g. {'hair_color', 'glasses'} (four if both)
    held_out = {attribute for pair in config.OOD_PAIRS if all(chosen[name] == word for name, word in pair.items())
                for attribute in pair}
    shown = {attribute: f'<strong>{word}</strong>' if attribute in held_out else word
             for attribute, word in chosen.items()}

    caption = CaptionGenerator(config).compose_caption(shown)
    warning = ('<span class="ood">Never seen together in training (OOD)</span>'
               '<span class="ood-note">In bold: the OOD pair.</span>') if held_out else ''

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
    # by the name of the attribute and not by position, because the page and MAPPING list the attributes in
    # different orders
    new_words = tuple(word if name == attribute else old for name, old in zip(ATTRIBUTES.values(), words))

    return (new_words, *caption_and_warning(*new_words))


def random_words(held_out):
    """Words picked at random for the five cards, for the buttons Random prompt and Random OOD prompt.

    Args:
        held_out: False for a combination seen in training, True for one with a pair held out of it (OOD).

    Returns:
        The five words, in the order of ATTRIBUTES.
    """
    while True:
        # one word per attribute, all equally likely, drawn again until the combination is of the kind asked
        chosen = {attribute: random.choice(list(config.MAPPING[attribute])) for attribute in ATTRIBUTES.values()}
        if is_held_out(chosen, config) == held_out:
            return tuple(chosen.values())


def run_folder(run):
    """Folder of a model of the grid.

    Args:
        run: the model chosen on the grid, e.g. {'dataset': '100,000', 'kind': 'conditional', 'size': 64}.

    Returns:
        e.g. runs_100/conditional_64, which holds a last.pt only if that model was trained.
    """
    return DATASETS[run['dataset']] / f'{run["kind"]}_{run["size"]}'


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
    """Table of what the attribute classifier sees in every image, with ✓ for the word of the card and ✗ for another.

    Args:
        chosen: {attribute: word} chosen on the cards; None for the baseline, which had no words to follow, and
            then the table only describes the images, without ✓ and ✗.
        seen: one {attribute: word} per image, the words the classifier sees in it.
        seeds: the seed of every image, in the same order as seen.

    Returns:
        One line that explains the table, then a Markdown table with one row per image, e.g.
        '| 7 | ✓ dark | ✗ medium | ✓ black | ✓ glasses | ✓ a beard |' (the ticks and crosses in a colored span),
        or '| 7 | dark | medium | black | glasses | a beard |' for the baseline.
    """
    intro = ('The attribute classifier reads each avatar: ✓ the chosen word, ✗ a different one.' if chosen else
             'The baseline has no text to follow: the attribute classifier describes what it drew.')
    rows = [intro, '', '| Seed | ' + ' | '.join(ATTRIBUTES) + ' |', '|---' * (len(ATTRIBUTES) + 1) + '|']

    for words, image_seed in zip(seen, seeds):
        # by the name of the attribute and not by position, because the page and MAPPING have different orders
        cells = [words[attribute] if chosen is None else
                 ('<span class="ok">✓</span> ' if words[attribute] == chosen[attribute]
                  else '<span class="no">✗</span> ') + words[attribute] for attribute in ATTRIBUTES.values()]
        rows.append(f'| {image_seed} | ' + ' | '.join(cells) + ' |')

    return '\n'.join(rows)


def on_generate(run, seed, count, guidance, save, *words):
    """Generate the images, judge them with the attribute classifier, show them and, if asked, save them.

    Args:
        run: the model chosen on the grid, e.g. {'dataset': '10,000', 'kind': 'conditional', 'size': 32}.
        seed: seed of the first image; the others get seed + 1, seed + 2, ...
        count: number of images.
        guidance: strength of classifier-free guidance.
        save: True to save every image in the folder of the model, in generated/ (e.g. runs/conditional_32/generated/).
        words: the words chosen on the cards, in the order of ATTRIBUTES.

    Returns:
        (list of (image enlarged 4 times, 'seed N · right words/5', only 'seed N' for the baseline), table of what
        the classifier sees (empty without a classifier), message with the time and the folder).
    """
    run_dir = run_folder(run)
    if not (run_dir / 'last.pt').exists():
        return [], '', f'No trained model in {run_dir}: run train.py first.'

    if run_dir not in models:
        models[run_dir] = load_model(run_dir)
    model, scheduler, run_config, vocabulary = models[run_dir]

    chosen = dict(zip(ATTRIBUTES.values(), words))
    caption = CaptionGenerator(config).compose_caption(chosen)     # the plain caption, without the bold
    seeds = [int(seed) + i for i in range(int(count))]
    start = time.time()
    images = generate(model, scheduler, run_config, vocabulary, caption, seeds, guidance)

    # the judge of the size of this experiment; while it is missing, it is looked for again at every click
    size = run_config.IMAGE_SIZE
    if classifiers.get(size) is None:
        classifiers[size] = load_classifier(config, size)

    captions = [f'seed {image_seed}' for image_seed in seeds]
    table = ''
    if classifiers[size] is not None:
        # the images as the classifier saw them in training, (count, 3, size, size) in [-1, 1]
        pixels = torch.stack([ImagePreprocessor.normalize(image) for image in images])
        seen = [attribute_words(row, config) for row in predict(classifiers[size], pixels, config).tolist()]
        if run_config.TEXT_CONDITIONING:
            table = verdict_table(chosen, seen, seeds)
            captions = [f'{text} · {sum(found[a] == chosen[a] for a in chosen)}/5'
                        for text, found in zip(captions, seen)]
        else:
            # the baseline read no words, so there is no score and the classifier only describes what it drew
            table = verdict_table(None, seen, seeds)

    # 4 times bigger, keeping the real pixels (no smoothing), as in the control grids
    images = [image.resize((image.width * 4, image.height * 4), Image.NEAREST) for image in images]

    message = f'{len(images)} images in {time.time() - start:.0f} s.'
    if not run_config.TEXT_CONDITIONING:
        message += ' Baseline without text: the attributes and the guidance are ignored.'
    if classifiers[size] is None:
        message += f' No attribute classifier for {size}x{size}: run train_classifier.py with IMAGE_SIZE = {size}.'

    if save:
        folder = run_dir / 'generated'
        folder.mkdir(exist_ok=True)
        for image, image_seed in zip(images, seeds):
            image.save(folder / file_name(words, image_seed, guidance, run_config.TEXT_CONDITIONING))
        message += f' Saved in {folder}.'

    return list(zip(images, captions)), table, message


def on_option(attribute, word, *words):
    """Click on an option of a sheet. It keeps the word, shows it on the card, closes the sheet and rewrites the prompt.

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


def on_random(held_out):
    """Click on Random prompt or Random OOD prompt, which puts new words on all the cards, as if chosen by hand.

    Args:
        held_out: False for Random prompt, True for Random OOD prompt.

    Returns:
        (the 5 words, the 5 cards, caption, warning, the options of every sheet with the chosen ones outlined).
    """
    words = random_words(held_out)
    cards = [gr.Button(value=word, icon=str(icon_file(attribute, word)))
             for attribute, word in zip(ATTRIBUTES.values(), words)]
    options = [gr.Button(variant='primary' if option == word else 'secondary')
               for attribute, word in zip(ATTRIBUTES.values(), words) for option in config.MAPPING[attribute]]

    return (*words, *cards, *caption_and_warning(*words), *options)


def text_controls(text, *words):
    """The controls of the caption, usable for an experiment with text and faded for the baseline, which ignores them.

    Args:
        text: False for the unconditional baseline.
        words: the words chosen on the cards, in the order of ATTRIBUTES, used again for the caption of a model
            with text.

    Returns:
        (the 5 cards, the 2 random buttons, the guidance, caption, warning).
    """
    caption, warning = caption_and_warning(*words) if text else (BASELINE_PROMPT, '')
    buttons = [gr.Button(interactive=text) for _ in range(len(ATTRIBUTES) + 2)]

    return (*buttons, gr.Radio(interactive=text), caption, warning)


def on_model(change, run, *words):
    """Click on a model of the grid or on a number of training images. The new choice becomes black, the models
    never trained on those images are faded, and the controls of the caption are turned on or off (see
    text_controls).

    Args:
        change: what was clicked, e.g. {'kind': 'unconditional', 'size': 64} or {'dataset': '100,000'}; {} when the
            page opens.
        run: the model chosen before the click, e.g. {'dataset': '10,000', 'kind': 'conditional', 'size': 32}.
        words: the words chosen on the cards, in the order of ATTRIBUTES.

    Returns:
        (the new run, the 4 models of the grid, the 2 numbers of training images, then what text_controls returns).
    """
    run = {**run, **change}
    # a model never trained on these images (e.g. the baseline at 64x64 on 100,000) is replaced by the conditional
    # one of its size
    if not (run_folder(run) / 'last.pt').exists():
        run['kind'] = 'conditional'

    cells =[gr.Button(variant='primary' if (kind, size) == (run['kind'], run['size']) else 'secondary',
                       interactive=(run_folder({**run, 'kind': kind, 'size': size}) / 'last.pt').exists())
             for kind in KINDS.values() for size in SIZES]
    datasets = [gr.Button(variant='primary' if dataset == run['dataset'] else 'secondary') for dataset in DATASETS]

    return (run,*cells, *datasets, *text_controls(run['kind'] == 'conditional', *words))


def build_page():
    """Build the page, with step 01 (the cards of the attributes and the settings), step 02 (the avatars) and one
    sheet of options per attribute, hidden until its card is clicked.

    Returns:
        The Gradio page (gr.Blocks), ready to launch.
    """
    first = {attribute: list(config.MAPPING[attribute])[0] for attribute in ATTRIBUTES.values()}
    with gr.Blocks(title='Avatar Diffusion') as page:
        gr.HTML('<header id="brand"><h1>Avatar Diffusion</h1><p>A small text-to-image diffusion model, trained from '
                'scratch on the Google Cartoon Set.</p></header>')
        # the chosen words, in the order of ATTRIBUTES. The cards only show them
        words = [gr.State(first[attribute]) for attribute in ATTRIBUTES.values()]
        run = gr.State(FIRST_RUN)       # the model chosen on the grid. The grid only shows it
        with gr.Row(elem_id='shell', equal_height=False):
            with gr.Column(scale=1, min_width=320, elem_id='choose'):
                gr.HTML(step('01', 'Choose your avatar', 'Pick the five attributes, then the model and the settings.'))
                with gr.Row(elem_id='cards'):
                    cards = {attribute: gr.Button(first[attribute], icon=str(icon_file(attribute, first[attribute])),
                                                  elem_id=f'card-{attribute}', elem_classes='card')
                             for attribute in ATTRIBUTES.values()}
                with gr.Row(elem_id='randoms'):
                    random_seen = gr.Button('Random prompt', size='sm', min_width=0)
                    random_ood = gr.Button('Random OOD prompt', size='sm', min_width=0)
                caption = gr.HTML(html_template=PROMPT, elem_id='prompt')
                # sanitize_html=False keeps the class of the badge. The text is ours, never typed by the user
                warning = gr.Markdown(sanitize_html=False, elem_id='warning')
                with gr.Row(elem_id='models'):
                    with gr.Column(min_width=0, elem_id='grid'):
                        gr.HTML('<span class="head">Model</span>')
                        cells = {}      # (kind, size) -> its pill, in the order of on_model
                        for label, kind in KINDS.items():
                            gr.HTML(f'<span class="kind">{label}</span>')
                            for size in SIZES:
                                chosen = (kind, size) == (FIRST_RUN['kind'], FIRST_RUN['size'])
                                cells[kind, size] = gr.Button(f'{size}×{size}', min_width=0,
                                                              variant='primary' if chosen else 'secondary')
                    # the third column, with one number of training images per row
                    with gr.Column(min_width=0, elem_id='data'):
                        gr.HTML('<span class="head">Training images</span>')
                        numbers = {dataset: gr.Button(dataset, min_width=0, variant='primary'
                                                      if dataset == FIRST_RUN['dataset'] else 'secondary')
                                   for dataset in DATASETS}
                count = gr.Radio(COUNTS, value=4, label='Images', elem_classes='pills')
                guidance = gr.Radio([(f'{value:g}', value) for value in GUIDANCES], value=config.GUIDANCE_SCALE,
                                    label='Guidance', elem_classes='pills')
                with gr.Row(elem_id='seed-row'):
                    seed = gr.Number(value=0, precision=0, label='Seed', min_width=120)
                    random_seed = gr.Button('Random seed', min_width=0, elem_id='random-seed')
                    save = gr.Checkbox(value=False, label='Save images', elem_classes='toggle', min_width=120)
                button = gr.Button('Generate avatars', variant='primary', elem_id='generate')
            with gr.Column(scale=1, min_width=320, elem_id='results'):
                gr.HTML(step('02', 'Your avatars', 'The attribute classifier checks every avatar against the prompt.'))
                empty = gr.HTML(EMPTY)
                # png, because the default webp would blur the enlarged pixels and change their colors. No buttons
                # over the avatars, the images are saved with Save images
                gallery = gr.Gallery(label='Generated avatars', show_label=False, columns=4, format='png',
                                     height='auto', buttons=[], elem_id='avatars', visible=False)
                # what the attribute classifier sees in every image (the colored ticks need their class)
                verdict = gr.Markdown(sanitize_html=False, elem_id='verdict')
                message = gr.Markdown(elem_id='message')

        all_options = []     # the options of every sheet, in the order of ATTRIBUTES, for the random buttons
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
            all_options += options
        for random_button, held_out in [(random_seen, False), (random_ood, True)]:
            random_button.click(partial(on_random, held_out), None,
                                [*words, *cards.values(), caption, warning, *all_options])

        # the prompt is written when the page opens; the results replace the empty frame at the first click
        # the grid and the controls of the caption follow the model, also for the one chosen when the page opens
        controls = [run, *cells.values(), *numbers.values(), *cards.values(), random_seen, random_ood, guidance,
                    caption, warning]
        page.load(partial(on_model, {}), inputs=[run, *words], outputs=controls)
        for (kind, size), cell in cells.items():
            cell.click(partial(on_model, {'kind': kind, 'size': size}), inputs=[run, *words], outputs=controls)
        for dataset, number in numbers.items():
            number.click(partial(on_model, {'dataset': dataset}), inputs=[run, *words], outputs=controls)
        # a new seed at random, from 0 to 9,999, short numbers that are easy to read under the avatars
        random_seed.click(lambda: random.randint(0, 9999), None, seed)
        button.click(lambda: (gr.HTML(visible=False), gr.Gallery(visible=True)), None, [empty, gallery]).then(
            on_generate, inputs=[run, seed, count, guidance, save, *words], outputs=[gallery, verdict, message])
    return page


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web demo of the cartoon diffusion model.')
    parser.add_argument('--share', action='store_true', help='also create a public link (e.g. on Colab)')
    # in Gradio 6 the theme and the style go to launch(), not to gr.Blocks()
    build_page().launch(share=parser.parse_args().share, theme=THEME, css=CSS, head=HEAD)

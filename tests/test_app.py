"""Tests of the web demo. Run them with:  python -m pytest tests/"""

import gradio as gr

import app
from config import Config


def test_every_word_has_its_icon():
    # the cards and the options of the page show one icon per word: a word without icon would break the page
    for attribute, words in Config().MAPPING.items():
        for word in words:
            assert app.icon_file(attribute, word).is_file(), f'no icon for {attribute} = {word}'


def test_choosing_an_option_changes_only_its_attribute():
    # the words of the page, in the order of ATTRIBUTES: skin, hair length, hair color, glasses, beard
    words = ('dark', 'short', 'black', 'glasses', 'a beard')
    new_words, caption, warning = app.choose('hair', 'long', words)
    assert new_words == ('dark', 'long', 'black', 'glasses', 'a beard')
    assert caption == 'a cartoon avatar with dark skin, long black hair, glasses and a beard'
    assert warning                                   # dark skin + long hair: held out of training (OOD)


def test_gallery_shows_the_real_pixels():
    galleries = [block for block in app.build_page().blocks.values() if isinstance(block, gr.Gallery)]
    assert galleries[0].format == 'png'       # webp (the default) would blur the 4x enlarged pixels


def test_file_names_keep_the_guidance():
    words = ('pale', 'long', 'blonde', 'no glasses', 'a beard')
    assert app.file_name(words, 7, 3.0, True) == 'pale_long_blonde_no-glasses_a-beard_guidance3_seed7.png'
    assert app.file_name(words, 7, 5.5, True) != app.file_name(words, 7, 3.0, True)   # no overwriting
    assert app.file_name(words, 7, 3.0, False) == 'unconditional_seed7.png'           # baseline: no text


def test_verdict_table_marks_the_words_the_classifier_does_not_see():
    # the cards: hair length before hair color
    chosen = {'face_color': 'dark', 'hair': 'long', 'hair_color': 'black', 'glasses': 'glasses', 'facial_hair': 'a beard'}
    # the classifier: the order of MAPPING (hair color before hair length), wrong only on the hair length
    seen = {'face_color': 'dark', 'hair_color': 'black', 'hair': 'medium', 'glasses': 'glasses', 'facial_hair': 'a beard'}
    table = app.verdict_table(chosen, [seen], [7])
    ok, no = '<span class="ok">✓</span>', '<span class="no">✗</span>'     # green tick, red cross
    assert table.splitlines()[-1] == f'| 7 | {ok} dark | {no} medium | {ok} black | {ok} glasses | {ok} a beard |'

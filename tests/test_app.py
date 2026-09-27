"""Tests of the web demo. Run them with:  python -m pytest tests/"""

import gradio as gr

import app


def test_gallery_shows_the_real_pixels():
    galleries = [block for block in app.build_page().blocks.values() if isinstance(block, gr.Gallery)]
    assert galleries[0].format == 'png'       # webp (the default) would blur the 4x enlarged pixels


def test_file_names_keep_the_guidance():
    words = ('pale', 'long', 'blonde', 'no glasses', 'a beard')
    assert app.file_name(words, 7, 3.0, True) == 'pale_long_blonde_no-glasses_a-beard_guidance3_seed7.png'
    assert app.file_name(words, 7, 5.5, True) != app.file_name(words, 7, 3.0, True)   # no overwriting
    assert app.file_name(words, 7, 3.0, False) == 'unconditional_seed7.png'           # baseline: no text

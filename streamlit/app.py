
import sys, os, time
sys.path.append(os.path.abspath(".."))

import numpy as np
from PIL import Image
import streamlit as st
import color_palettes

from pathlib import Path

#######################

IMG_SHAPE = (500,500)

#######################

"""
# Algorithmic Color Palettes

_created by Austin Poor_

Using clustering algorithms to create algorithmic color palettes.

---
"""

st.image(
    "assets/sample-palette-1.png",
    caption="Example Color Palette",
    use_column_width=True,
)

"""
## Try it Out!

Select options in the sidebar and then press the "Generate Palette" button.

You can use one of the supplied images or test it out with your own!

---
"""

# Algorithm parameters
input_image = st.sidebar.selectbox(
    'Clustering algorithm:',
    sorted([p.name for p in Path("images").glob("*.jpeg")]) + ["<Custom Image>"]
)
input_method = "kmeans"
input_hsv = st.sidebar.checkbox(
    'Use HSV (rather than RGB):',
)
input_nclusters = st.sidebar.slider(
    'Color Palette Size:',
    min_value=1, 
    max_value=25,
    value=5
)
input_fitler_low = st.sidebar.slider(
    'Filter Out Colors Darker Than...',
    min_value=0, 
    max_value=100,
    value=5
)
input_fitler_high = st.sidebar.slider(
    'Filter Out Colors Brighter Than...',
    min_value=0, 
    max_value=100,
    value=80
)

st.sidebar.write(
    "*The low filter value should be less than "
    "the high filter value."
)


if str(input_image) == "<Custom Image>":
    input_my_photo = st.file_uploader(
        "Custom image path",
        type=["png", "jpg", "jpeg"]
    )
else:
    "Image Name: ", input_image

"Color Clustering Algorithm: ", "K-Means"
"Size of Color Palette", input_nclusters
"RGB or HSV?", "HSV" if input_hsv else "RGB"
(
    "Only using colors in the brightness range ",
    input_fitler_low, "% to ", input_fitler_high, "%"
)
if input_fitler_low >= input_fitler_high:
    st.write("")
    time.sleep(1)
    st.stop()

button_pressed = st.button("Generate Palette")
if st.button("Stop!"): st.stop()

bar = st.empty()

img = None

if input_image != "<Custom Image>":
    img = Image.open(
        Path("images") / input_image
    )
    st.image(
        img,
        use_column_width=True
    )
elif input_my_photo is not None:
    img = Image.open(input_my_photo)
    st.image(
        img,
        use_column_width=True
    )
else:
    img = None

palette_img = st.empty()

if button_pressed:

    if img is not None:
        bar = st.progress(0)

        palette = color_palettes.calc_palette_to_img(
            np.array(img.resize(IMG_SHAPE)),
            hsv=input_hsv,
            method=input_method,
            nclusters=input_nclusters,
            filter_low=input_fitler_low / 100,
            filter_high=input_fitler_high  / 100,
            callback=lambda n: bar.progress(n / 10)
        )

        bar.empty()

        palette_img = st.image(
            palette,
            use_column_width=True
        )

    else:
        st.write("Please Select an Image")


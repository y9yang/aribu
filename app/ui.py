"""Display helpers shared by both pages."""
import numpy as np
import pydeck as pdk
import streamlit as st
from aribu.viz import ERROR_COLOURS, stretch_rgb

COLOURS = (ERROR_COLOURS * 255).astype(np.uint8)
CLEAR_RGB = np.array([70, 90, 105])     # cloud probability 0; 100 is white
THRESHOLD_HELP = ("A pixel counts as water when the model's probability is above this value. "
                  "Lowering it detects more water, but also produces more false alarms.")

def error_image(pred, truth, valid):
    code = np.where(pred & truth, 1, np.where(pred & ~truth, 2, np.where(~pred & truth, 3, 0)))
    code[~valid] = 4
    return COLOURS[code]

def legend(items):
    swatch = ('<span style="margin-right:1.4em;white-space:nowrap"><span style="display:inline-block;width:.7em;'
              'height:.7em;margin-right:.4em;background:rgb{}"></span>{}</span>')
    # same 75% as the captions, on the text only so the swatches keep the map colours
    st.markdown('<div style="font-size:.85em;color:color-mix(in srgb, currentColor 75%, transparent)">' + "".join(swatch.format(tuple(int(v) for v in c), name) for c, name in items)
                + "</div>", unsafe_allow_html=True)

def fit_view(extent, height):
    """A map view that fits `extent` (left, bottom, right, top) into a map `height` px tall.

    The centred layout is 704 px wide, and we keep 20 px to spare on each side.
    On a Mercator map the world is 512 px wide at zoom 0, and latitudes stretch by 1 / cos(latitude).
    """
    l, b, r, t = extent
    zoom = np.log2(min(664 / (r - l), (height - 40) * np.cos(np.radians((b + t) / 2)) / (t - b)) * 360 / 512)
    return pdk.ViewState(longitude=(l + r) / 2, latitude=(b + t) / 2, zoom=zoom)

def optical_image(rgb):
    """Sentinel-2 bands (3, H, W) as true colour uint8, no data white."""
    return (stretch_rgb(rgb) * 255).astype(np.uint8)

def cloud_image(cloud, no_data):
    """Cloud probability from 0 to 100 as RGB uint8, from slate for clear sky to white, no data in the no-data grey."""
    v = np.clip(cloud / 100, 0, 1)[..., None]
    rgb = CLEAR_RGB * (1 - v) + 255 * v
    rgb[no_data] = COLOURS[4]
    return rgb.astype(np.uint8)

import aribu            # GDAL fix, keep first
import base64
import io
import json
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image
from aribu.dataset import LABEL_WATER
from aribu.exposure import tally
from aribu.metrics import confusion, metrics_from_counts
from aribu.paths import DEMO_DIR
from aribu.viz import ERROR_COLOURS

COLOURS = (ERROR_COLOURS * 255).astype(np.uint8)
FLOOD_RGBA, PERM_RGBA = (47, 111, 143, 230), (138, 155, 168, 150)

st.set_page_config(page_title="Aribu", layout="centered")

@st.cache_data
def load_manifest():
    return json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))

@st.cache_resource
def load_event(event):
    with np.load(DEMO_DIR / event / "chips.npz") as z:
        chips = {k: z[k] for k in z.files}
    return chips, json.loads((DEMO_DIR / event / "districts.geojson").read_text(encoding="utf-8"))

def masks(chips, model, threshold):
    """Flood water from the model and from the hand label, both without permanent water."""
    usable = chips["valid"] & ~chips["perm"]
    return (chips[f"prob_{model}"] > threshold * 255) & usable, (chips["label"] == LABEL_WATER) & usable

def exposure_table(event, model, threshold):
    chips, districts = load_event(event)
    flooded, truth = masks(chips, model, threshold)
    n = int(chips["district"].max())
    people, people_label, flooded_m2, mapped_m2 = (np.zeros(n + 1) for _ in range(4))
    for i, area in enumerate(chips["px_area"]):
        d, pop = chips["district"][i], chips["pop"][i]
        p, px = tally(flooded[i], d, pop, n)
        people += p
        flooded_m2 += px * area
        people_label += tally(truth[i], d, pop, n)[0]
        mapped_m2 += tally(chips["valid"][i], d, pop, n)[1] * area
    rows = [{"District": f["properties"]["name"],
             "People · model": round(people[f["properties"]["id"]], -1), # pyright: ignore[reportCallIssue, reportArgumentType]
             "People · hand label": round(people_label[f["properties"]["id"]], -1), # pyright: ignore[reportCallIssue, reportArgumentType]
             "Flooded km²": flooded_m2[f["properties"]["id"]] / 1e6,
             "Mapped km²": mapped_m2[f["properties"]["id"]] / 1e6,
             "Share mapped": mapped_m2[f["properties"]["id"]] / 1e6 / f["properties"]["area_km2"]}
            for f in districts["features"]]
    return pd.DataFrame(rows).sort_values("People · model", ascending=False)

@st.cache_data
def overlays(event, model, threshold):
    """One transparent PNG per chip, as data URLs: flood water blue, permanent water grey."""
    chips, _ = load_event(event)
    flooded, _ = masks(chips, model, threshold)
    urls = []
    for f, perm in zip(flooded, chips["perm"]):
        rgba = np.zeros((*f.shape, 4), np.uint8)
        rgba[perm] = PERM_RGBA
        rgba[f] = FLOOD_RGBA
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format="PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    return urls

def flood_map(event, model, threshold, table):
    chips, districts = load_event(event)
    most = max(table["People · model"].max(), 1)
    by_name = table.set_index("District")["People · model"]
    shaded = {"type": "FeatureCollection", "features": [
        {**f, "properties": {**f["properties"],
                             "fill": [47, 111, 143, int(12 + 70 * np.sqrt(by_name[f["properties"]["name"]] / most))],
                             "people": f"{by_name[f['properties']['name']]:,.0f} people exposed in mapped squares"}}
        for f in districts["features"]]}
    squares = [{"path": [[l, b], [r, b], [r, t], [l, t], [l, b]]} for l, b, r, t in chips["bounds"]]
    layers = [
        pdk.Layer("GeoJsonLayer", shaded, filled=True, stroked=True, pickable=True,
                  get_fill_color="properties.fill", get_line_color=[90, 105, 115, 140], line_width_min_pixels=1),
        *(pdk.Layer("BitmapLayer", image=pdk.types.String(url), bounds=list(b)) for url, b in zip(overlays(event, model, threshold), chips["bounds"])), # pyright: ignore[reportAttributeAccessIssue]
        pdk.Layer("PathLayer", squares, get_path="path", get_color=[31, 42, 51, 110], width_min_pixels=1),
    ]
    corners = [[x, y] for l, b, r, t in chips["bounds"] for x, y in ((l, b), (r, t))]
    view = pdk.data_utils.compute_view(corners, view_proportion=1)          # pyright: ignore[reportAttributeAccessIssue]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style=None, # pyright: ignore[reportArgumentType]
                             tooltip={"text": "{name}\n{people}"}), height=520) # pyright: ignore[reportArgumentType]

def error_image(pred, truth, valid):
    code = np.where(pred & truth, 1, np.where(pred & ~truth, 2, np.where(~pred & truth, 3, 0)))
    code[~valid] = 4
    return COLOURS[code]

def legend(items):
    swatch = ('<span style="margin-right:1.4em;white-space:nowrap"><span style="display:inline-block;width:.7em;'
              'height:.7em;margin-right:.4em;background:rgb{}"></span>{}</span>')
    st.markdown('<div style="font-size:.85em;opacity:.8">' + "".join(swatch.format(tuple(int(v) for v in c), name) for c, name in items)
                + "</div>", unsafe_allow_html=True)

def event_picker(events, key, other):
    """The flood event control. The page shows it twice; picking in one moves the other."""
    names = list(events)
    st.session_state.setdefault(key, names[0])
    return st.segmented_control("Flood event", names, required=True, key=key,
                                help="Events, and the chips within each, run from best to worst IoU.",
                                format_func=lambda e: f"{events[e]['country']} · {events[e]['s1_date'][:4]}",
                                on_change=lambda: st.session_state.update({other: st.session_state[key]}))

manifest = load_manifest()
events, models = manifest["events"], manifest["models"]

st.title("Aribu")
st.caption("Flood water mapped from satellite radar, and the people living where it reached.")
st.caption("*Aribu* is Akkadian for raven. In the Mesopotamian flood myth, Utnapishtim sent one out from his boat "
           "on Mount Nimush, and it did not return: the flood had receded.")

st.header("Flood maps")
cols = st.columns(len(models) + 1)
for col, m in zip(cols, models.values()):
    col.metric(f"{m['label']} · IoU", f"{m['micro_iou']['test']:.2f}",
               help=f"All test split chips, at threshold {m['threshold']}. Bolivia hold-out: {m['micro_iou']['bolivia']:.2f}.")
cols[-1].caption("IoU: the overlap between mapped and hand-labelled water, divided by their union. 1 is perfect.")

event = event_picker(events, "event", "event_exposure")
info = events[event]
chips, _ = load_event(event)

i = st.pagination(len(chips["ids"]), key=f"chip_{event}") - 1
chip_id, valid, truth = chips["ids"][i], chips["valid"][i], chips["label"][i] == LABEL_WATER
inputs = [(Image.open(DEMO_DIR / event / f"{chip_id}_vh.jpg"), "Radar (Sentinel-1, VH)"), # pyright: ignore[reportOperatorIssue]
          (Image.open(DEMO_DIR / event / f"{chip_id}_rgb.jpg"), "Optical (Sentinel-2)")] # pyright: ignore[reportOperatorIssue]
maps = [(error_image(truth, truth, valid), "Hand label")]
for name, m in models.items():
    pred = chips[f"prob_{name}"][i] > m["threshold"] * 255
    iou = metrics_from_counts(confusion(pred, truth, valid))["iou"]
    maps.append((error_image(pred, truth, valid), f"{m['label']} · IoU {iou:.2f}" if np.isfinite(iou) else m["label"]))
for row in (inputs, maps):
    for col, (image, caption) in zip(st.columns(len(row)), row):
        col.image(image, caption=caption, width="stretch")
legend([(COLOURS[1], "water"), (COLOURS[2], "false alarm"), (COLOURS[3], "missed"), (COLOURS[4], "no data")])

st.divider()
st.header("Exposure")
event_picker(events, "event_exposure", "event")
left, right = st.columns(2, gap="large")
model = left.segmented_control("Model", list(models), default="fusion", required=True,
                               format_func=lambda m: models[m]["label"])
threshold = right.slider("Decision threshold", 0.05, 0.95, models[model]["threshold"], 0.05, key=f"threshold_{model}",
                         help="Probability above which a pixel counts as water. Lower finds more water, and more false alarms.")
table = exposure_table(event, model, threshold)
mapped_km2 = sum(v.sum() * area for v, area in zip(chips["valid"], chips["px_area"])) / 1e6

st.caption(f"Totals cover only the {len(chips['ids'])} squares mapped for this event ({mapped_km2:,.0f} km²), not the whole flood.")
a, b, c = st.columns(3)
a.metric("Exposed people · model", f"{table['People · model'].sum():,.0f}")
b.metric("Exposed people · hand label", f"{table['People · hand label'].sum():,.0f}")
c.metric("Flooded area", f"{table['Flooded km²'].sum():,.1f} km²")

flood_map(event, model, threshold, table)
legend([(FLOOD_RGBA[:3], "flood water"), (PERM_RGBA[:3], "permanent water"), ((31, 42, 51), "mapped area")])

st.dataframe(table, hide_index=True, width="stretch", column_config={
    "People · model": st.column_config.NumberColumn(format="localized"),
    "People · hand label": st.column_config.NumberColumn(format="localized"),
    "Flooded km²": st.column_config.NumberColumn(format="%.2f"),
    "Mapped km²": st.column_config.NumberColumn(format="%.1f"),
    "Share mapped": st.column_config.NumberColumn(format="percent")})
st.download_button("Download table (CSV)", table.to_csv(index=False),
                   file_name=f"aribu-{event.lower()}-{model}-exposure.csv", mime="text/csv") # pyright: ignore[reportOptionalMemberAccess]

with st.expander("How this is computed"):
    st.markdown(f"""
- **Mapped area only**: {len(chips['ids'])} squares of 5 km from the {'Bolivia hold-out' if info['split'] == 'bolivia' else 'test split'}, which the models never trained on. They are a small sample picked by the dataset authors, not full coverage of the flood, so figures are not scaled up to the whole event or to whole districts.
- **Flood water**: pixels the model scores above the threshold, minus permanent water (JRC). Only pixels with a valid radar reading and a hand label count, so both columns compare like with like.
- **People**: WorldPop {info['worldpop_year']} counts per ~100 m cell, spread evenly over the 10 m pixels inside it.
- **Districts**: geoBoundaries ADM2 ({info['boundaries_license']}).
- **Exposed, not affected**: people living where water was mapped; not a count of people needing aid. Figures rounded to 10.
""")

st.divider()
st.caption("Data: Sen1Floods11 (Bonafilia et al., 2020) · WorldPop · geoBoundaries · JRC Global Surface Water. Predictions computed offline.")

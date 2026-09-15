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
from ui import COLOURS, THRESHOLD_HELP, error_image, fit_view, legend

FLOOD_RGBA, PERM_RGBA = (47, 111, 143, 230), (138, 155, 168, 150)

@st.cache_data
def load_manifest():
    return json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))

@st.cache_resource
def load_event(event):
    with np.load(DEMO_DIR / event / "chips.npz") as z:
        chips = {k: z[k] for k in z.files}
    return chips, json.loads((DEMO_DIR / event / "districts.geojson").read_text(encoding="utf-8"))

def flood_mask(chips, model, threshold):
    """Flood water from the model, without permanent water."""
    return (chips[f"prob_{model}"] > threshold * 255) & chips["valid"] & ~chips["perm"]

def exposure_table(event, model, threshold):
    chips, districts = load_event(event)
    flooded = flood_mask(chips, model, threshold)
    n = int(chips["district"].max())
    people, flooded_m2 = np.zeros(n + 1), np.zeros(n + 1)
    for i, area in enumerate(chips["px_area"]):
        p, px = tally(flooded[i], chips["district"][i], chips["pop"][i], n)
        people += p
        flooded_m2 += px * area
    rows = [{"District": f["properties"]["name"],
             "Exposed people": round(people[f["properties"]["id"]], -1), # pyright: ignore[reportCallIssue, reportArgumentType]
             "Flooded km²": flooded_m2[f["properties"]["id"]] / 1e6}
            for f in districts["features"]]
    return pd.DataFrame(rows).sort_values("Exposed people", ascending=False)

@st.cache_data
def overlays(event, model, threshold):
    """One transparent PNG per chip, as data URLs: flood water blue, permanent water grey."""
    chips, _ = load_event(event)
    flooded = flood_mask(chips, model, threshold)
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
    most = max(table["Exposed people"].max(), 1)
    by_name = table.set_index("District")["Exposed people"]
    shaded = {"type": "FeatureCollection", "features": [
        {**f, "properties": {**f["properties"],
                             "fill": [47, 111, 143, int(12 + 70 * np.sqrt(by_name[f["properties"]["name"]] / most))],
                             "people": f"{by_name[f['properties']['name']]:,.0f} people exposed in the mapped chips"}}
        for f in districts["features"]]}
    squares = [{"path": [[l, b], [r, b], [r, t], [l, t], [l, b]]} for l, b, r, t in chips["bounds"]]
    layers = [
        pdk.Layer("GeoJsonLayer", shaded, filled=True, stroked=True, pickable=True,
                  get_fill_color="properties.fill", get_line_color=[90, 105, 115, 140], line_width_min_pixels=1),
        *(pdk.Layer("BitmapLayer", image=pdk.types.String(url), bounds=list(b)) for url, b in zip(overlays(event, model, threshold), chips["bounds"])), # pyright: ignore[reportAttributeAccessIssue]
        pdk.Layer("PathLayer", squares, get_path="path", get_color=[31, 42, 51, 110], width_min_pixels=1),
    ]
    view = fit_view((*chips["bounds"][:, :2].min(0), *chips["bounds"][:, 2:].max(0)), 520)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style=None, # pyright: ignore[reportArgumentType]
                             tooltip={"text": "{name}\n{people}"}), height=520) # pyright: ignore[reportArgumentType]

def event_picker(events, key, other, help=None):
    """The flood event control. The page shows it twice; picking in one moves the other."""
    names = list(events)
    st.session_state.setdefault(key, names[0])
    return st.segmented_control("Flood event", names, required=True, key=key, help=help,
                                format_func=lambda e: f"{events[e]['country']} · {events[e]['s1_date'][:4]}",
                                on_change=lambda: st.session_state.update({other: st.session_state[key]}))

manifest = load_manifest()
events, models = manifest["events"], manifest["models"]
live_page = st.session_state.pages["live"]

st.title("Aribu")
st.caption("Deep learning models (U-Nets) map floodwater from satellite imagery, and the results are used to estimate how many people live in the affected areas.")
st.container(key="quiet_raven").caption("*Aribu* is Akkadian for raven. After the flood in the Epic of Gilgamesh, Utnapishtim sent out a dove "
                                        "and a swallow, and both returned. The raven never did: it had found dry land.")

st.header("Flood maps")
with st.container(border=True):
    for col, m in zip(st.columns(len(models)), models.values()):
        col.metric(f"{m['label']} · IoU", f"{m['micro_iou']['test']:.2f}",
                   help=f"Computed on all test chips pooled together, at threshold {m['threshold']}. "
                        f"On Bolivia, an event withheld entirely from training: {m['micro_iou']['bolivia']:.2f}.")
    st.caption("IoU, or Intersection over Union, measures the overlap with hand-labelled water. "
               "It ranges from 0 for no overlap to 1 for a perfect match.")

st.caption(f"We show a few 5 km × 5 km tiles, called chips, from each of {len(events)} flood events in Sen1Floods11, "
           "our training dataset. None of these chips were used in training, and Bolivia was withheld entirely.")
event = event_picker(events, "event", "event_exposure",
                     help="Events are sorted from best to worst IoU. The chips of each event follow the same order.")
info = events[event]
chips, _ = load_event(event)

i = st.pagination(int(chips["held_out"].sum()), key=f"chip_{event}") - 1
chip_id, valid, truth = chips["ids"][i], chips["valid"][i], chips["label"][i] == LABEL_WATER
inputs = [(Image.open(DEMO_DIR / event / f"{chip_id}_vh.jpg"), "Radar · Sentinel-1 VH"), # pyright: ignore[reportOperatorIssue]
          (Image.open(DEMO_DIR / event / f"{chip_id}_mndwi.jpg"), "Water index · Sentinel-2 MNDWI"), # pyright: ignore[reportOperatorIssue]
          (Image.open(DEMO_DIR / event / f"{chip_id}_rgb.jpg"), "Optical · Sentinel-2")] # pyright: ignore[reportOperatorIssue]
maps = []
for name, m in models.items():
    pred = chips[f"prob_{name}"][i] > m["threshold"] * 255
    iou = metrics_from_counts(confusion(pred, truth, valid))["iou"]
    maps.append((error_image(pred, truth, valid), f"{m['label']} · IoU {iou:.2f}" if np.isfinite(iou) else m["label"]))
maps.append((error_image(truth, truth, valid), "Hand label"))
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
                         help=THRESHOLD_HELP)
table = exposure_table(event, model, threshold)
mapped_km2 = sum(v.sum() * area for v, area in zip(chips["valid"], chips["px_area"])) / 1e6

# One card for the results, so they stand apart from the controls above, with the note that qualifies them
with st.container(border=True):
    a, b = st.columns(2)
    a.metric("Exposed people", f"{table['Exposed people'].sum():,.0f}")
    b.metric("Flooded area", f"{table['Flooded km²'].sum():,.1f} km²")
    st.caption(f"Note that these numbers cover all {len(chips['ids'])} hand-labelled chips of this event in Sen1Floods11, "
               f"{mapped_km2:,.0f} km² in total, which is still a small part of the whole flood.")

flood_map(event, model, threshold, table)
legend([(FLOOD_RGBA[:3], "flood water"), (PERM_RGBA[:3], "permanent water")])

st.dataframe(table, hide_index=True, width="stretch", column_config={
    "District": st.column_config.TextColumn(width="medium"),
    "Exposed people": st.column_config.NumberColumn(format="localized"),
    "Flooded km²": st.column_config.NumberColumn(format="%.2f")})
st.download_button("Download table as CSV", table.to_csv(index=False),
                   file_name=f"aribu-{event.lower()}-{model}-exposure.csv", mime="text/csv") # pyright: ignore[reportOptionalMemberAccess]

with st.expander("How we computed this"):
    st.markdown(f"""
- **Mapped area**: we use all {len(chips['ids'])} hand-labelled chips of this event, each 5 km × 5 km, {'from the Bolivia hold-out split, which the models never saw during training' if info['split'] == 'bolivia' else 'from the training, validation and test splits. Note that the models learned from the training chips, so on those the model numbers are closer to the hand label than they would be on a new flood'}. Since they cover only a small part of the flood, we do **not** scale the numbers up to the whole event or to entire districts.
- **Flood water**: pixels where the model's probability is above the threshold, excluding permanent water such as rivers and lakes, which we take from JRC Global Surface Water. We only count pixels with both a valid radar reading and a hand label.
- **People**: WorldPop {info['worldpop_year']} provides a head count for each grid cell of about 100 m, and we distribute it evenly over the 10 m pixels inside that cell.
- **Districts**: second-level administrative areas, known as ADM2, from geoBoundaries. License: {info['boundaries_license'].split(' (')[0]}.
- **Exposed versus affected**: *exposed* means living where we mapped flood water, which is not the same as needing aid. All head counts are rounded to the nearest 10.
""")

st.divider()
st.container(key="quiet_credits").caption(
    "Data: Sen1Floods11 by Bonafilia et al., 2020 · WorldPop · geoBoundaries · JRC Global Surface Water. "
    + ("Note that the predictions on this page were computed in advance: on the Live page, the models run when you ask them to."
       if live_page else "The predictions were computed in advance: the app does not run the models itself."))
if live_page:
    st.page_link(live_page, label="Run the models on recent floods →")

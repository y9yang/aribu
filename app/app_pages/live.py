import json
import time
from datetime import date, datetime
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from aribu import live
from aribu.dataset import normalised_indices
from aribu.exposure import districts_on_grid, pixel_area_m2, population_on_grid, tally
from aribu.metrics import confusion, metrics_from_counts
from aribu.paths import DEMO_DIR, LIVE_DIR, MODELS_DIR
from aribu.sources import population_window
from aribu.viz import mndwi_image, radar_image
from ui import COLOURS, THRESHOLD_HELP, cloud_image, error_image, fit_view, legend, optical_image

CHECKPOINTS = {"radar": "radar-only.pt", "fusion": "fusion.pt"}
OUTLINE_RGBA = (47, 111, 143, 140)
SHAPE = (live.SQUARE_SIZE, live.SQUARE_SIZE)

@st.cache_data
def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

@st.cache_data
def load_analysts(code, square_id):
    """The analysts' map of the square, with the water that was there before the flood added."""
    with np.load(LIVE_DIR / code / "analysts.npz") as flood, np.load(LIVE_DIR / code / "water_before.npz") as before:
        return flood[square_id] | before[square_id]

@st.cache_resource(show_spinner=False)
def earth_engine():
    live.init_earth_engine()

@st.cache_resource(show_spinner=False)
def load_model(name):
    # imported here, not at the top: PyTorch takes several seconds to import, so the page opens without it
    from aribu.model import load_checkpoint
    return load_checkpoint(MODELS_DIR / CHECKPOINTS[name])

@st.cache_data(ttl="6h", max_entries=100, show_spinner=False)
def find_scenes(left, top, day):
    earth_engine()
    return live.find_scenes(live.square_transform(left, top), day)

# about 10 MB per square, so fewer entries than for the scene search
@st.cache_data(ttl="6h", max_entries=20, show_spinner=False)
def fetch_square(left, top, scenes):
    earth_engine()
    return live.fetch_square(live.square_transform(left, top), scenes)

def short_date(d):
    return f"{d:%b} {d.day}"

def days_from(d, reference, name):
    gap = (d - reference).days
    if gap == 0:
        return f"same day as {name}"
    return f"{abs(gap)} day{'s' if abs(gap) > 1 else ''} {'after' if gap > 0 else 'before'} {name}"

def area_map(code, act):
    """The activation's squares under the analysts' outline. Returns the selection event."""
    squares = load_json(LIVE_DIR / code / "squares.geojson")
    tips = {"type": "FeatureCollection", "features": [
        {**f, "properties": {**f["properties"], "tip": f"Square {f['properties']['id']}\n"
                                                       f"{f['properties']['flood_share']:.0%} inside the analysts' flood outline"}}
        for f in squares["features"]]}
    # Once a square is picked, Streamlit draws it opaque and the other squares at 40%, in their own fill colour:
    # a light slate, under the outline, so the picked square stands out and the outline stays visible.
    layers = [
        pdk.Layer("GeoJsonLayer", tips, id="squares", pickable=True, auto_highlight=True, filled=True, stroked=True,
                  get_fill_color=[190, 200, 208, 40], highlight_color=[31, 42, 51, 40],
                  get_line_color=[31, 42, 51, 130], line_width_min_pixels=1),
        pdk.Layer("GeoJsonLayer", load_json(LIVE_DIR / code / "outline.geojson"), filled=True, stroked=False,
                  get_fill_color=list(OUTLINE_RGBA)),
    ]
    # as tall as the squares need at the page's width, from 320 to 640 px, plus the 40 px that fit_view keeps free
    l, b, r, t = act["extent"]
    height = int(np.clip(664 * (t - b) / ((r - l) * np.cos(np.radians((b + t) / 2))), 320, 640)) + 40
    deck = pdk.Deck(layers=layers, initial_view_state=fit_view(act["extent"], height), map_style=None, tooltip={"text": "{tip}"}) # pyright: ignore[reportArgumentType]
    return st.pydeck_chart(deck, height=height, on_select="rerun", selection_mode="single-object", key=f"map_{code}")

def run_square(code, square_id, bounds, scenes, name, day, models):
    """Download the square, run the model and put everything the page shows in st.session_state.live_result."""
    left, top = bounds[0], bounds[3]
    transform = live.square_transform(left, top)
    start = time.perf_counter()
    with st.status("Running...", expanded=True) as status:
        try:
            s1_day = date.fromisoformat(scenes["s1_date"])
            st.write(f"Found a Sentinel-1 image from {short_date(s1_day)}, {days_from(s1_day, day, 'the chosen date')}")
            t0 = time.perf_counter()
            square = fetch_square(left, top, scenes)
            st.write(f"Downloaded the radar and optical bands of square {square_id} in {time.perf_counter() - t0:.1f} s")
            status.update(label="Loading the model...")
            from aribu.model import DEVICE, prob_from_input     # as in load_model
            model, ckpt = load_model(name)
            status.update(label="Running...")
            norm = ckpt["normalisation"]
            t0 = time.perf_counter()
            prob = prob_from_input(model, live.square_input(square, ckpt["arm"], norm["mean"], norm["std"]))
            st.write(f"Ran the model ({models[name]['label'].lower()}) on the {'GPU' if DEVICE == 'cuda' else 'CPU'} in {time.perf_counter() - t0:.1f} s")
            pop, pop_transform = population_window(LIVE_DIR / code / "population.tif", bounds)
            districts = load_json(LIVE_DIR / code / "districts.geojson")["features"]
            st.session_state.live_result = {
                "key": (code, square_id, day, name), "square_id": square_id, "name": name, "s1_day": s1_day,
                "square": square, "prob": prob, "transform": transform, "arm": ckpt["arm"],
                "analysts": load_analysts(code, square_id),
                "pop": population_on_grid(pop, pop_transform, transform, SHAPE),
                "district": districts_on_grid([f["geometry"] for f in districts], transform, SHAPE)}
            st.write("Looked up how many people live in the square, and in which districts")
            status.update(label=f"Done in {time.perf_counter() - start:.1f} s", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"We could not map this square: {e}")

def show_result(result, act, code, models):
    square, prob, name = result["square"], result["prob"], result["name"]
    no_optical = live.no_optical_data(square)
    valid = np.isfinite(square["vv"]) & np.isfinite(square["vh"])
    pred, analysts = prob > models[name]["threshold"], result["analysts"]
    empty = no_optical.all()
    s2 = "no Sentinel-2 image" if empty else "Sentinel-2"
    if result["arm"] == "s1+s2":
        mndwi = normalised_indices(square["green"], square["nir"], square["swir"])[1]
        first = (mndwi_image(mndwi, no_optical), f"Water index · {s2}" if empty else "Water index · Sentinel-2 MNDWI")
    else:
        first = (radar_image(square["vh"]), "Radar · Sentinel-1 VH")
    inputs = [first,
              (cloud_image(square["cloud"], no_optical), f"Cloud probability · {s2}"),
              (optical_image(square["rgb"]), f"Optical · {s2}")]
    maps = [(error_image(pred, pred, valid), models[name]["label"]),
            (error_image(pred, analysts, valid), "Agreement"),
            (error_image(analysts, analysts, valid), "Analysts + water before the flood")]
    for row in (inputs, maps):
        for col, (image, caption) in zip(st.columns(3), row):
            col.image(image, caption=caption, width="stretch")
    legend([(COLOURS[1], "water"), (COLOURS[2], "model only"),
            (COLOURS[3], "analysts only"), (COLOURS[4], "no data")])

    iou = metrics_from_counts(confusion(pred, analysts, valid))["iou"]
    image_time = datetime.fromisoformat(act["image"]["time"])
    pre_event = datetime.fromisoformat(act["pre_event_image"]["time"])
    s1_day, analysts_day = result["s1_day"], image_time.date()
    gap = days_from(s1_day, analysts_day, "the analysts' image")
    with st.container(border=True):
        st.metric("Agreement with the analysts · IoU", f"{iou:.2f}" if np.isfinite(iou) else "no water in either map",
                  help="Water in both maps, divided by water in either map, over the pixels with a valid radar reading. "
                       "The analysts mapped only the flood, so we add the water in their Sentinel-2 image "
                       f"from before it ({short_date(pre_event)}, {pre_event.year}).")
        st.caption(f"The analysts drew their map from a {act['image']['sensor']} image of {short_date(image_time)}, "
                   f"{image_time.year}, and it has errors of its own. So this number tells us how well two maps agree, "
                   "which says less about accuracy than IoU against hand labels."
                   + (f" Our radar image is from {short_date(s1_day)}, {gap}, so the flood may have "
                      "changed in between." if s1_day != analysts_day else ""))

    st.divider()
    st.header("Exposure")
    threshold = st.slider("Decision threshold", 0.05, 0.95, models[name]["threshold"], 0.05, key=f"live_threshold_{name}",
                          help=THRESHOLD_HELP)
    districts = load_json(LIVE_DIR / code / "districts.geojson")["features"]
    n, area_m2 = len(districts), pixel_area_m2(result["transform"], SHAPE[0])

    def counts(t):
        return tally((prob > t) & valid & ~square["perm"], result["district"], result["pop"], n)

    low, _ = counts(min(threshold + 0.1, 0.95))
    high, _ = counts(max(threshold - 0.1, 0.05))
    _, pixels = counts(threshold)
    with st.container(border=True):
        a, b = st.columns(2)
        a.metric("Exposed people", f"{round(low.sum(), -1):,.0f} – {round(high.sum(), -1):,.0f}")
        b.metric("Flooded area", f"{pixels.sum() * area_m2 / 1e6:,.1f} km²")
        st.caption(f"Note that these numbers cover this 5 km × 5 km square only, and that the range of exposed people "
                   f"comes from moving the threshold 0.1 either side of {threshold:.2f}.")
    present = set(np.unique(result["district"]).tolist())
    table = pd.DataFrame([{"District": f["properties"]["name"], "Exposed people, low": round(low[i], -1),
                           "Exposed people, high": round(high[i], -1), "Flooded km²": pixels[i] * area_m2 / 1e6}
                          for i, f in enumerate(districts, start=1) if i in present])
    if len(table):
        table = table.sort_values("Exposed people, high", ascending=False)
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "District": st.column_config.TextColumn(width="medium"),
            "Exposed people, low": st.column_config.NumberColumn(format="localized"),
            "Exposed people, high": st.column_config.NumberColumn(format="localized"),
            "Flooded km²": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("Download table as CSV", table.to_csv(index=False),
                           file_name=f"aribu-{code.lower()}-{result['square_id'].lower()}-{name}-exposure.csv", mime="text/csv")

# Newest flood first, so it is also the one the page opens on.
activations = dict(sorted(load_json(LIVE_DIR / "activations.json").items(),
                          key=lambda item: item[1]["image"]["time"], reverse=True))
models = load_json(DEMO_DIR / "manifest.json")["models"]

st.title("Recent floods")
st.caption("On this page the models run when you press **Run**: we download the satellite images of a 5 km × 5 km square "
           "from a recent flood that Copernicus analysts have mapped. The models never saw these floods during training. "
           "There are no hand labels either, so we compute IoU, or Intersection over Union, against the analysts' map instead.")

st.header("Flood maps")
left, right = st.columns([2, 1])
code = left.selectbox("Flood event", list(activations),
                      format_func=lambda c: f"{activations[c]['area']}, {activations[c]['country']} · "
                                            f"{datetime.fromisoformat(activations[c]['image']['time']):%b %Y}",
                      help="Recent floods that the analysts of the Copernicus Emergency Management Service (EMS) have mapped.")
act = activations[code]
day = right.date_input("Image date", datetime.fromisoformat(act["image"]["time"]).date(),
                       min_value=date(2017, 1, 1), max_value="today", key=f"day_{code}",
                       help="By default, this is the date of the analysts' image. We take the Sentinel-1 image closest to "
                            "noon UTC (Coordinated Universal Time) on this date, at most 3 days away, and then the Sentinel-2 "
                            "image closest in time to that one, again at most 3 days away.")

picked = area_map(code, act).selection.objects.get("squares", [])
legend([(OUTLINE_RGBA[:3], "analysts' flood outline")])

if not picked:
    st.info("Click a square on the map to pick it.")
else:
    square_id, bounds = picked[0]["properties"]["id"], picked[0]["properties"]["bounds"]
    try:
        with st.spinner("Looking for satellite images..."):
            scenes = find_scenes(bounds[0], bounds[3], day)
    except Exception as e:
        scenes = None
        st.error(f"We could not look for satellite images: {e}")
    else:
        if scenes is None:
            st.info(f"There is no Sentinel-1 image of square {square_id} within 3 days of {short_date(day)}. "
                    "Sentinel-1 passes every few days, so a nearby date usually works.")
    if scenes:
        s1_day, auto = date.fromisoformat(scenes["s1_date"]), live.pick_model(scenes["cloud_share"])
        s2_day = date.fromisoformat(scenes["s2_date"]) if scenes["s2_date"] else None
        box = st.container(border=True, key="scene_box")     # filled below, once the model is chosen
        with st.container(horizontal=True, vertical_alignment="bottom"):
            choice = st.segmented_control("Model", ["auto", "radar", "fusion"], default="auto", required=True, key="live_model",
                                          format_func=lambda m: "Auto" if m == "auto" else models[m]["label"])
            run = st.button("Run", type="primary", icon=":material/play_arrow:")
        name = auto if choice == "auto" else choice
        # radar + optical leads with the optical image, radar only with the radar image
        lead, lead_day = ("Optical image", s2_day) if name == "fusion" and s2_day else ("Radar image", s1_day)
        note = f"optical image, {short_date(s2_day)}" if s2_day else "no optical image"
        with box:
            a, b, c = st.columns(3)
            a.metric(f"{lead} · square {square_id}", short_date(lead_day), delta=days_from(lead_day, day, "the chosen date"),
                     delta_color="off", delta_arrow="off")
            b.metric("Cloud over the square", f"{scenes['cloud_share']:.0%}", delta=note, delta_color="off", delta_arrow="off",
                     help=f"Share of the square where Sentinel-2's cloud probability is {live.CLOUD_PROBABILITY}% or more, "
                          "or where Sentinel-2 has no image.")
            c.metric("Model", models[name]["label"], delta="picked by the cloud rule" if name == auto else "your choice",
                     delta_color="off", delta_arrow="off")
            st.caption(f"In our cloud tests on Bolivia (withheld from training), radar + optical no longer beats radar only "
                       f"from {live.RADAR_ONLY_FROM:.0%} cloud. This square has {scenes['cloud_share']:.0%}, so the cloud rule "
                       f"picks **{models[auto]['label'].lower()}**. "
                       + ("You can override the choice below." if name == auto
                          else f"You picked **{models[name]['label'].lower()}** below."))
        if run:
            run_square(code, square_id, bounds, scenes, name, day, models)
        result = st.session_state.get("live_result")
        if result and result["key"] == (code, square_id, day, name):
            show_result(result, act, code, models)
        else:
            st.info("Press **Run** to see the maps and the exposure table.")

pre_event = datetime.fromisoformat(act["pre_event_image"]["time"])
with st.expander("How we computed this"):
    st.markdown(f"""
- **Images**: Sentinel-1 radar (VV and VH) and the Sentinel-2 bands of the square, downloaded from Google Earth Engine when you press **Run**. We prepare them as in Sen1Floods11, our training dataset: radar in dB, Sentinel-2 Level-1C, both on the same 10 m grid.
- **Model**: from {live.RADAR_ONLY_FROM:.0%} cloud over the square (pixels without a Sentinel-2 image count as cloud), we pick radar only, and below it radar + optical. You can override this choice. Note that radar + optical **always** gets 0 for both optical water indices where Sentinel-2's cloud probability is {live.CLOUD_PROBABILITY}% or more, or where Sentinel-2 has no image (as in our cloud tests), so on a cloudy square it mostly sees radar.
- **Analysts' map**: the flood outline (the observed flood extent) of the Copernicus EMS Rapid Mapping activation [{code}]({act['link']}), turned into pixels on our 10 m grid. The analysts mapped only the flood, so we add the water that was there before it: open water in their Sentinel-2 image of {short_date(pre_event)}, {pre_event.year} (MNDWI above {live.MNDWI_WATER}), and permanent water from JRC Global Surface Water where that image is cloudy. So both maps show all water, as the hand labels do.
- **Flood water**: pixels where the model's probability is above the threshold, excluding permanent water such as rivers and lakes, which we take from JRC Global Surface Water. We only count pixels with a valid radar reading.
- **People**: GHS-POP R2023A, from the Global Human Settlement Layer of the European Commission's Joint Research Centre, provides a head count for each grid cell of 3 arcseconds (about 90 m), and we distribute it evenly over the 10 m pixels inside that cell. We use its 2025 layer, a projection from earlier censuses.
- **Districts**: second-level administrative areas, known as ADM2, from geoBoundaries. License: {act['boundaries_license'].split(' (')[0]}.
- **Exposed versus affected**: *exposed* means living where we mapped flood water, which is not the same as needing aid. All head counts are rounded to the nearest 10.
""")

st.divider()
st.container(key="quiet_credits").caption(
    "Data: Copernicus Emergency Management Service © European Union, Rapid Mapping activations " + ", ".join(activations)
    + " · Copernicus Sentinel-1 and Sentinel-2 data, via Google Earth Engine · GHS-POP R2023A by Schiavina et al., 2023"
      " · JRC Global Surface Water · geoBoundaries.")
st.page_link(st.session_state.pages["past"], label="← Check the models against hand labels on past floods")

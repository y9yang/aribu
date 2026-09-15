import aribu            # GDAL fix, keep first
import importlib.util
import os
import streamlit as st

st.set_page_config(page_title="Aribu", layout="centered")
# Captions default to 60% of the text colour, too faint on this background. Informative ones get 75%,
# the raven story and the credits (in the "quiet" containers) a step lighter at 65%.
# The navigation links are muted, and the current page's link gets the full text colour and a thin blue underline.
# The values in the live page's scene box are a step smaller, so "Radar + optical" fits in a third of the page.
st.html("""<style>
[data-testid="stCaptionContainer"] { opacity: .75 }
[class*="st-key-quiet"] [data-testid="stCaptionContainer"] { opacity: .65 }
.st-key-scene_box [data-testid="stMetricValue"] { font-size: 1.75rem }
.st-key-nav a, .st-key-nav a:hover, .st-key-nav a:focus-visible { background: none; padding-inline: 0 }
.st-key-nav a p { font-size: .875rem; font-weight: 500; color: color-mix(in srgb, currentColor 55%, transparent) }
.st-key-nav a:hover p, .st-key-nav_current a p { color: inherit }
.st-key-nav_current a p { text-decoration: underline 1.5px #2F6F8F; text-underline-offset: .4em }
</style>""")

def live_available():
    """The live page needs torch and Earth Engine installed, and a service account key."""
    installed = all(importlib.util.find_spec(m) for m in ("torch", "ee"))
    return installed and bool(os.environ.get("EE_KEY_JSON") or os.environ.get("EE_KEY_FILE"))

past = st.Page("app_pages/past_floods.py", title="Past floods", default=True)
live = st.Page("app_pages/live.py", title="Live", url_path="live")
pages = [past, live] if live_available() else [past]
st.session_state.pages = {"past": past, "live": live if live in pages else None}

pg = st.navigation(pages, position="hidden")
if len(pages) > 1:
    with st.container(key="nav", horizontal=True, horizontal_alignment="right", gap="large"):
        for page in pages:
            with st.container(key="nav_current" if page is pg else None, width="content"):
                st.page_link(page)
pg.run()

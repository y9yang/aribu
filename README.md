# Aribu

Aribu maps floodwater from satellite imagery with deep learning, and estimates how many people live in the flooded areas. We train U-Nets (neural networks that label every pixel of an image) on the Sen1Floods11 dataset and compare them with a classical threshold baseline. A Streamlit app shows the results on two pages: one for past floods from the dataset, and one that runs the models on recent floods, downloading the satellite images on demand.

Try the app here: **[huggingface.co/spaces/Gwynpleina/aribu](https://huggingface.co/spaces/Gwynpleina/aribu)**

## Results

We report IoU, or Intersection over Union: the overlap between the predicted and the hand-labelled water, divided by their union. It ranges from 0 (no overlap) to 1 (a perfect match). All scores below are *micro* IoU: the pixels of all chips in a split are pooled before computing the score.

The test split has 90 chips from the same 10 flood events as the training split. Bolivia is an 11th event, left out of training entirely, so it shows how the models do on a flood they have never seen.

| Model | Input | Test IoU | Bolivia IoU |
| --- | --- | --- | --- |
| Threshold baseline | Sentinel-1 radar (VH) | 0.59 | 0.60 |
| U-Net, radar only | Sentinel-1 radar (VV, VH) | 0.68 | 0.72 |
| U-Net, radar + optical | Sentinel-1 radar (VV, VH) and Sentinel-2 water indices (NDWI, MNDWI) | 0.82 | 0.78 |

Findings from notebooks 02 and 03:

- Both U-Nets beat the baseline on both splits. Note that the radar-only U-Net gains more on Bolivia (+0.12) than on the test split (+0.09).
- Adding the optical channels makes the **biggest** difference: a U-Net trained on them alone scores 0.83 on the test split. But clouds often hide the ground during a flood, and optical satellites cannot see through them (radar can).
- To choose between the two optical models, we covered growing parts of the optical input with synthetic clouds. As expected, radar + optical degrades more gracefully than optical only. It still falls below the radar-only model once 50% of each chip is hidden on the test split (25% on Bolivia), so we keep both, and the Live page switches to radar only from 25% cloud.
- The *macro* IoU (each chip scored separately, then averaged) is much lower: 0.39 for radar only and 0.55 for radar + optical on the test split. Chips with very little water pull the average down, since a handful of wrong pixels already ruins their score.

## How it works

### Data

[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) contains 446 hand-labelled chips (tiles of 512 × 512 pixels at 10 m resolution, about 5 km × 5 km) from 11 flood events around the world. Each chip comes with:

- **Sentinel-1** (S1) radar: two bands, VV and VH, in decibels. The satellite sends out its own microwave pulses and measures what bounces back, so it works through clouds and at night. Calm water reflects the pulses away from the satellite, so it looks dark.
- **Sentinel-2** (S2) optical imagery: 13 bands, from visible light to shortwave infrared. We derive two standard water indices from them, NDWI (Normalized Difference Water Index) and MNDWI (Modified Normalized Difference Water Index), which are high over water because water absorbs infrared light.
- A hand label marking each pixel as water, land or no data.

We use the dataset's official train, validation, test and Bolivia splits (252, 89, 90 and 15 chips).

### Threshold baseline

Radar images are grainy (this is called *speckle*), so we first smooth the VH band with a 5 × 5 median filter. A pixel is then water if its value is below -23.5 dB. The cut-off was tuned on the unfiltered training split, and the filtered version beat three other variants on the validation split. See [notebooks/02_threshold-baseline.ipynb](notebooks/02_threshold-baseline.ipynb).

### U-Nets

We use `smp.Unet` from [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) with a ResNet-34 encoder pretrained on ImageNet. The training setup:

- The radar bands are clipped to -50 to 1 dB and standardised with the mean and standard deviation of the training split only (so nothing leaks from the other splits). The water indices already lie between -1 and 1 and go in as they are.
- The loss is binary cross-entropy over the labelled pixels only.
- AdamW, learning rate 3e-4, batch size 8, 30 epochs, mixed precision, keeping the epoch with the best validation IoU.
- The decision threshold (the probability above which a pixel counts as water) is tuned on the validation split: 0.40 for radar only, 0.35 for radar + optical.

We also built a U-Net by hand, following the original 2015 paper. With a third of the parameters, it came close: test IoU 0.66 against 0.67, both at the default threshold of 0.5. See [notebooks/03_unet-comparison.ipynb](notebooks/03_unet-comparison.ipynb).

### Exposure

For each chip, we count the people living where the model finds flood water:

1. **Flood water**: the model's water pixels minus the permanent water (rivers, lakes) from JRC Global Surface Water.
2. **People**: WorldPop estimates the number of people in grid cells of about 100 m, and we spread each estimate evenly over the 10 m pixels inside the cell.
3. **Districts**: second-level administrative areas (ADM2) from geoBoundaries. We sum the exposed people and the flooded area per district.

### Recent floods (the Live page)

We picked three recent floods mapped by the analysts of the Copernicus Emergency Management Service (EMS), and cut one mapped area of each into 5 km squares, the size of a chip. We keep the squares that are at least 5% inside the analysts' flood outline.

Most of what the page needs is prepared in advance by [src/aribu/activations.py](src/aribu/activations.py) and stored in [data/live/](data/live/): the squares, the analysts' map of each square, the water there before the flood, the population and the districts. When you pick a square, the page looks up the nearest images, and pressing **Run** downloads them (with the permanent water) and runs the model:

1. **Images**: the Sentinel-1 image closest to the chosen date (at most 3 days away), and the Sentinel-2 image closest to that one (again at most 3 days away), both from Google Earth Engine. We prepare them as in Sen1Floods11, which was itself exported from Earth Engine: radar in dB, Sentinel-2 Level-1C, on the same 10 m grid. Without a Sentinel-1 image the square cannot be mapped, and the page says so. As a check, the Bolivia chips downloaded this way give the same IoU as the dataset's own images (0.72 and 0.78).
2. **Model**: from 25% cloud over the square we pick radar only, and below that radar + optical (you can override this). Pixels without a Sentinel-2 image count as cloud. Where Sentinel-2 sees cloud, both water indices are set to 0, as in our cloud tests.
3. **Agreement**: without hand labels, we compute IoU against the analysts' map instead. The analysts mapped only the flood, so we add the water that was there before it: open water in their Sentinel-2 image from before the flood (MNDWI above 0.2), and JRC permanent water where that image is cloudy.
4. **Exposure**: as for the chips, but with GHS-POP R2023A (its 2025 layer, cells of about 90 m) in place of WorldPop. The exposed people are shown as a range, from moving the threshold 0.1 either side.

## The app

The [app](https://huggingface.co/spaces/Gwynpleina/aribu) has two pages:

- **Past floods**: browse the test and Bolivia chips of five flood events (Nigeria, Sri Lanka, Bolivia, Somalia and Pakistan), with a few swapped for validation chips. Each chip shows the radar, the water index and the true-colour image next to the predictions of both models, coloured by correct water, false alarm and missed water. Below, pick an event, a model and a decision threshold to see the exposed people and flooded area per district, on a map and in a table you can download as CSV. The exposure covers all hand-labelled chips of the event, training chips included. The predictions are computed in advance and stored in [data/demo/](data/demo/).
- **Live**: pick one of three recent floods (Pakistan, Mozambique or Colombia) and a date (by default the day of the analysts' image), then click a square on the map. The page shows the date of the image, the cloud over the square and the model it picked. Press **Run** to see the model's map next to the analysts' map, their agreement (IoU) and the exposure table for that square.

Neither page needs a GPU: Past floods shows predictions computed in advance, and the Live page is meant to run on the CPU (as on the Space), one square at a time.

Everything the app reads is in the repository (the two checkpoints in [models/](models/), and the prepared data in [data/demo/](data/demo/) and [data/live/](data/live/)), so it runs right after cloning, without the dataset. To use the Live page on your own machine, you need an Earth Engine key (see below); without one, the app shows only the Past floods page.

The Space sleeps after 48 hours without visitors, so the first visit after that takes a while to load.

## Getting started

### Requirements

- To use the app: nothing, it runs on the [Space](https://huggingface.co/spaces/Gwynpleina/aribu).
- To run the app yourself: Docker (see [Running with Docker](#running-with-docker)), or the conda environment below if you have an NVIDIA GPU. For the Live page, also a Google Earth Engine key.
- To retrain the models or rebuild the data: [conda](https://conda-forge.org/download/) (Miniforge or Miniconda) and an NVIDIA GPU, since the environment installs the GPU build of PyTorch.

### Installation

```bash
git clone https://github.com/y9yang/aribu.git
cd aribu
conda env create -f environment.yml
conda activate aribu
```

This also installs the `aribu` package from [src/](src/) in editable mode.

The environment has to be **activated**: calling its `python` directly skips the `PATH` setup, and PyTorch and rasterio then fail to load their libraries (at least on Windows).

### Earth Engine access

You need this only to run the Live page yourself (the Space has its own key) or to rebuild `data/live/`. The images come from Google Earth Engine, and the app logs in with a service account:

1. Create a Google Cloud project, register it for **noncommercial** Earth Engine use, and enable the Earth Engine API.
2. Create a service account in that project with the roles Earth Engine Resource Viewer and Service Usage Consumer, and download a JSON key for it. Keep the key outside the repository, since anyone with the file can use the project's quota.
3. Before starting the app, set the environment variable `EE_KEY_FILE` to the path of the key (or `EE_KEY_JSON` to its text). The app looks only for these two variables, so a personal Earth Engine login is not enough.

   ```bash
   export EE_KEY_FILE=/path/to/key.json         # bash
   $env:EE_KEY_FILE = "C:\path\to\key.json"     # PowerShell
   ```

### Running the app

```bash
streamlit run app/streamlit_app.py
```

### Running with Docker

The [Dockerfile](Dockerfile) builds the image that runs on Hugging Face, with the CPU build of PyTorch. The key never goes into the image: we mount it when the container starts.

```bash
docker build -t aribu .
docker run --rm -p 8080:8080 -v /path/to/key.json:/run/secrets/ee.key.json:ro -e EE_KEY_FILE=/run/secrets/ee.key.json aribu
```

Then open <http://localhost:8080>. Without `-v` and `-e`, the app shows the Past floods page only.

The Space is deployed with `deploy/hf-space/deploy_hf.py`, and reads its settings from `deploy/hf-space/README.md`.

### Reproducing the results

1. Download the hand-labelled part of Sen1Floods11 with [gsutil](https://cloud.google.com/storage/docs/gsutil_install):

   ```bash
   gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled data/raw/HandLabeled
   ```

2. Run the notebooks in order. Notebook 02 writes the baseline and the normalisation statistics to `data/results/`, which notebook 03 reads. Notebook 03 then writes the two model checkpoints to [models/](models/), and its scores to `data/results/`.
3. Rebuild the Past floods data. This downloads WorldPop and geoBoundaries into `data/cache/` on the first run:

   ```bash
   python -m aribu.demo
   ```

4. Rebuild the Live page's data. This needs Earth Engine access (the key from above, or your own login), and downloads the EMS products, GHS-POP and geoBoundaries into `data/cache/` on the first run:

   ```bash
   python -m aribu.activations
   ```

5. Run the tests (the Earth Engine test is skipped without a key or the raw chips):

   ```bash
   pytest
   ```

Retraining in a fresh kernel gives slightly different scores: in our runs, the validation IoU varied by about 0.02 between restarts, because some GPU operations are not deterministic.

## Project structure

```
aribu/
├── app/
│   ├── streamlit_app.py        # entry point and navigation
│   ├── app_pages/
│   │   ├── past_floods.py      # the Past floods page
│   │   └── live.py             # the Live page
│   └── ui.py                   # display helpers shared by both pages
├── data/
│   ├── demo/                   # precomputed predictions, population and districts for Past floods
│   ├── live/                   # flood outlines, squares, population and districts for Live
│   ├── external/               # Sen1Floods11 metadata (event dates and countries)
│   └── splits/                 # official Sen1Floods11 split CSVs
├── deploy/
│   ├── requirements.txt        # pinned packages for the Docker image
│   └── hf-space/               # Space settings and upload script
├── models/                     # frozen checkpoints: radar-only.pt, fusion.pt
├── notebooks/
│   ├── 01_data-exploration.ipynb
│   ├── 02_threshold-baseline.ipynb
│   └── 03_unet-comparison.ipynb
├── src/aribu/
│   ├── dataset.py              # loading chips, preprocessing, water indices, synthetic clouds
│   ├── baseline.py             # speckle filter and threshold predictor
│   ├── model.py                # building, loading and running the U-Nets
│   ├── train.py                # training loop and loss
│   ├── metrics.py              # confusion counts, IoU and friends, micro and macro
│   ├── exposure.py             # population and districts on the chip grid
│   ├── live.py                 # one square from Earth Engine, the cloud rule and the model input
│   ├── demo.py                 # builds data/demo/ for Past floods
│   ├── activations.py          # builds data/live/ for Live
│   ├── sources.py              # downloads and population windows
│   ├── viz.py                  # colour maps and error panels
│   ├── report.py               # JSON output
│   └── paths.py                # repository paths
├── tests/
├── Dockerfile
├── environment.yml
└── pyproject.toml
```

All the logic lives in `src/aribu/`. The notebooks import from it and are committed with their outputs, so the figures render on GitHub.

## Limitations

- The hand-labelled chips cover only a small part of each flood, so the exposure numbers describe the mapped chips (or the one square, on the Live page), and we do not scale them up to the whole event or to entire districts.
- The analysts' map has errors of its own, and it can be from a different day than our radar image. So the live IoU measures how well two maps agree, which says less about accuracy than IoU against hand labels.
- *Exposed* means living where we mapped flood water, which is not the same as needing aid.
- GHS-POP's 2025 layer is a projection from earlier censuses.
- Spreading each WorldPop cell evenly over its pixels ignores where people actually live within those 100 m.
- Our synthetic clouds are smoothed random noise. Real clouds (and their shadows) look different, so the cloud experiment gives only a rough picture.
- The decision thresholds were tuned on the validation split, and they do not always carry over to a new event: on Bolivia, the tuned threshold lowers the radar + optical IoU from 0.80 to 0.78.

## Data and credits

- **Sen1Floods11**: Bonafilia, D., Tellman, B., Anderson, T. and Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops.
- **Copernicus Sentinel-1 and Sentinel-2** data, downloaded for the Live page through [Google Earth Engine](https://earthengine.google.com/).
- **Copernicus Emergency Management Service**: flood outlines from the Rapid Mapping activations EMSR838, EMSR857 and EMSR865, © European Union, [mapping.emergency.copernicus.eu](https://mapping.emergency.copernicus.eu/).
- **WorldPop**: population counts at about 100 m, [worldpop.org](https://www.worldpop.org/).
- **GHS-POP R2023A**: Schiavina, M., Freire, S., Carioli, A. and MacManus, K. (2023). *GHS-POP R2023A: GHS population grid multitemporal (1975-2030).* European Commission, Joint Research Centre. doi:10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE.
- **geoBoundaries**: Runfola, D. et al. (2020). *geoBoundaries: a global database of political administrative boundaries.* PLOS ONE. The license differs per country and is shown in the app.
- **JRC Global Surface Water**: Pekel, J.-F., Cottam, A., Gorelick, N. and Belward, A. S. (2016). *High-resolution mapping of global surface water and its long-term changes.* Nature. The permanent water layer ships with Sen1Floods11, and the Live page takes it from Earth Engine.
- **U-Net**: Ronneberger, O., Fischer, P. and Brox, T. (2015). *U-Net: convolutional networks for biomedical image segmentation.* MICCAI.

## About the name

*Aribu* is Akkadian for raven. After the flood in the Epic of Gilgamesh, Utnapishtim sent out a dove and a swallow, and both returned. The raven never did: it had found dry land.

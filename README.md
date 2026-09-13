# Aribu

Aribu maps floodwater from satellite imagery with deep learning, and uses the maps to estimate how many people live in the flooded areas. We train U-Nets (neural networks that label every pixel of an image) on the Sen1Floods11 dataset, compare them with a classical threshold baseline, and present the results in a Streamlit app.

Try the app here: **[aribu-demo.streamlit.app](https://aribu-demo.streamlit.app/)**

## Results

We report IoU, or Intersection over Union: the overlap between the predicted and the hand-labelled water, divided by their union. It ranges from 0 for no overlap to 1 for a perfect match. All scores below are *micro* IoU, which means the pixels of all chips in a split are pooled together before computing the score.

The test split contains 90 chips from the same 10 flood events that the models were trained on. Bolivia is an 11th event, withheld from training entirely, so it shows how well the models carry over to a flood they have never seen.

| Model | Input | Test IoU | Bolivia IoU |
| --- | --- | --- | --- |
| Threshold baseline | Sentinel-1 radar (VH) | 0.59 | 0.60 |
| U-Net, radar only | Sentinel-1 radar (VV, VH) | 0.68 | 0.72 |
| U-Net, radar + optical | Sentinel-1 radar (VV, VH) and Sentinel-2 water indices (NDWI, MNDWI) | 0.82 | 0.78 |

What we learned along the way:

- Both U-Nets beat the baseline on both splits. Note that the radar-only U-Net improves more on Bolivia (+0.12) than on the test split (+0.09).
- Adding the optical channels makes the **biggest** difference. In fact, a U-Net trained on the optical channels alone scores even higher on the test split (0.83). But keep in mind that clouds often hide the ground during a flood, and optical satellites cannot see through them (radar can).
- To decide which optical model to keep, we covered growing parts of the optical input with synthetic clouds. As expected, the radar + optical model degrades more gracefully than the optical-only one. Note that it still falls below the radar-only model once 50% of each chip is hidden on the test split (25% on Bolivia), which is why we keep both models.
- The *macro* IoU, where we score each chip separately and then average, is much lower: 0.39 for radar only and 0.55 for radar + optical on the test split. Chips with very little water pull the average down, since a handful of wrong pixels already ruins their score.

## How it works

### Data

[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) contains 446 hand-labelled chips (tiles of 512 × 512 pixels at 10 m resolution, about 5 km × 5 km) from 11 flood events around the world. Each chip comes with:

- **Sentinel-1** (S1) radar: two bands, VV and VH, in decibels. The satellite sends out its own microwave pulses and measures what bounces back, so it works through clouds and at night. Calm water reflects the pulses away from the satellite and therefore looks dark.
- **Sentinel-2** (S2) optical imagery: 13 bands, from visible light to shortwave infrared. We derive two standard water indices from them, NDWI (Normalized Difference Water Index) and MNDWI (Modified Normalized Difference Water Index), which are high over water because water absorbs infrared light.
- A hand label marking each pixel as water, land or no data.

We use the dataset's official train, validation, test and Bolivia splits (252, 89, 90 and 15 chips).

### Threshold baseline

Radar images are grainy (this is called *speckle*), so we first smooth the VH band with a 5 × 5 median filter. A pixel is then water if its value is below -23.5 dB. We tuned this cut-off on the training split and chose the variant among four candidates on the validation split. See [notebooks/02_threshold-baseline.ipynb](notebooks/02_threshold-baseline.ipynb).

### U-Nets

We use `smp.Unet` from [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) with a ResNet-34 encoder pretrained on ImageNet. The training setup:

- Inputs are clipped and standardised with statistics from the training split only, to avoid leaking information from the other splits.
- The loss is binary cross-entropy, computed over labelled pixels only.
- AdamW optimiser, learning rate 3e-4, batch size 8, 30 epochs, mixed precision. We keep the epoch with the best validation IoU.
- The decision threshold (the probability above which a pixel counts as water) is tuned on the validation split: 0.40 for radar only, 0.35 for radar + optical.

We also built a U-Net by hand, following the original 2015 paper. It came close (test IoU 0.66 against 0.67 at the default threshold) with a third of the parameters. See [notebooks/03_unet-comparison.ipynb](notebooks/03_unet-comparison.ipynb).

### Exposure

For each chip, we count the people living where the model detects flood water:

1. **Flood water**: the model's water pixels, minus the permanent water (rivers, lakes) from JRC Global Surface Water.
2. **People**: WorldPop gives a head count for grid cells of about 100 m, which we spread evenly over the 10 m pixels inside each cell.
3. **Districts**: second-level administrative areas (ADM2) from geoBoundaries. We add up the exposed people and the flooded area per district.

## The app

The [Streamlit app](https://aribu-demo.streamlit.app/) has two sections:

- **Flood maps**: browse the held-out chips of five flood events (Nigeria, Sri Lanka, Bolivia, Somalia and Pakistan). For each chip, we show the radar, the water index and the true-colour image, next to the predictions of both models, coloured by correct water, false alarm and missed water.
- **Exposure**: pick an event, a model and a decision threshold, and see the exposed people and flooded area per district on a map and in a table (which can be downloaded as a CSV file).

The predictions were computed in advance and are stored in [data/demo/](data/demo/), so the app runs without a GPU and without the raw dataset.

## Getting started

### Requirements

- [conda](https://conda-forge.org/download/) (Miniforge or Miniconda)
- An NVIDIA GPU, since the environment installs the GPU build of PyTorch. Note that the app alone needs neither the GPU nor PyTorch (see below).

### Installation

```bash
git clone https://github.com/y9yang/aribu.git
cd aribu
conda env create -f environment.yml
conda activate aribu
```

This also installs the `aribu` package from [src/](src/) in editable mode.

Note that the environment has to be **activated**. Calling the environment's `python` directly skips the `PATH` setup, and PyTorch then fails to load its GPU libraries (at least on Windows).

### Running the app

```bash
streamlit run app/streamlit_app.py
```

To run the app without the full environment (for example on Streamlit Community Cloud), install the lighter set of packages in [app/requirements.txt](app/requirements.txt) from the repository root:

```bash
pip install -r app/requirements.txt
```

### Reproducing the results

1. Download the hand-labelled part of Sen1Floods11 with [gsutil](https://cloud.google.com/storage/docs/gsutil_install):

   ```bash
   gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled data/raw/HandLabeled
   ```

2. Run the notebooks in order. Notebook 02 writes the baseline and the normalisation statistics to `data/results/`, which notebook 03 reads. Notebook 03 then writes the two model checkpoints to [models/](models/).
3. Rebuild the app data. This downloads WorldPop and geoBoundaries into `data/cache/` on the first run:

   ```bash
   python -m aribu.demo
   ```

4. Run the tests:

   ```bash
   pytest
   ```

Note that retraining in a fresh kernel gives slightly different scores: in our runs, the validation IoU varied by about 0.02 between restarts, because some GPU operations are not deterministic.

## Project structure

```
aribu/
├── app/
│   ├── streamlit_app.py        # the Streamlit app
│   └── requirements.txt        # packages for running the app on its own
├── data/
│   ├── demo/                   # precomputed predictions, population and districts for the app
│   ├── external/               # Sen1Floods11 metadata (event dates and countries)
│   ├── splits/                 # official Sen1Floods11 split CSVs
│   └── raw/                    # the downloaded dataset (git-ignored)
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
│   ├── demo.py                 # builds data/demo/ for the app
│   ├── viz.py                  # colour maps and error panels
│   ├── report.py               # JSON output
│   └── paths.py                # repository paths
├── tests/
├── environment.yml
└── pyproject.toml
```

All the logic lives in `src/aribu/`. The notebooks import from it and are committed with their outputs, so the figures render on GitHub.

## Limitations

- The hand-labelled chips cover only a small part of each flood, so the exposure numbers describe the mapped chips. We do not scale them up to the whole event or to entire districts.
- *Exposed* means living where we mapped flood water, which is not the same as needing aid.
- Spreading each WorldPop cell evenly over its pixels ignores where people actually live within those 100 m.
- Our synthetic clouds are smoothed random noise. Real clouds (and their shadows) look different, so the cloud experiment gives a rough picture at best.
- The decision thresholds were tuned on the validation split, and they do not always carry over to a new event: on Bolivia, the tuned threshold lowers the radar + optical IoU from 0.80 to 0.78.

## Data and credits

- **Sen1Floods11**: Bonafilia, D., Tellman, B., Anderson, T. and Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops.
- **WorldPop**: population counts at about 100 m, [worldpop.org](https://www.worldpop.org/).
- **geoBoundaries**: Runfola, D. et al. (2020). *geoBoundaries: a global database of political administrative boundaries.* PLOS ONE. The license differs per country and is shown in the app.
- **JRC Global Surface Water**: Pekel, J.-F., Cottam, A., Gorelick, N. and Belward, A. S. (2016). *High-resolution mapping of global surface water and its long-term changes.* Nature. The permanent water layer ships with Sen1Floods11.
- **U-Net**: Ronneberger, O., Fischer, P. and Brox, T. (2015). *U-Net: convolutional networks for biomedical image segmentation.* MICCAI.

## About the name

*Aribu* is Akkadian for raven. After the flood in the Epic of Gilgamesh, Utnapishtim sent out a dove and a swallow, and both returned. The raven never did: it had found dry land.

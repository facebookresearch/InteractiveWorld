# Installation and asset preparation

This guide expands the four preparation steps of the [README](../README.md#1-preparation).
After it, `./launch/simulator.sh doctor` reports `"ok": true` and every command in the
documentation works.

## Requirements

| | Tested | Notes |
| :--- | :--- | :--- |
| OS | Linux x86_64 (CentOS Stream 9; Ubuntu 22.04+ is the Isaac Sim reference) | |
| GPU | NVIDIA A100 80 GB, driver 580 | One simulator worker uses about 4 GB of GPU memory. Isaac Sim's own [requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) apply. |
| Disk | about 100 GB | Isaac Sim environment ~22 GB, BEHAVIOR-1K dataset 33 GB (plus the 31 GB zip while installing), baked cache 3.4 GB |
| Tools | `git`, `conda`, optional `ffmpeg` | `ffmpeg` is needed only for `--video` |

## Step 1: Isaac Sim 5.1 and BEHAVIOR-1K 3.9.0

InteractiveWorld uses the scenes and objects of [BEHAVIOR-1K](https://behavior.stanford.edu)
and runs inside the same Python environment as OmniGibson. The official BEHAVIOR-1K
installer installs Isaac Sim 5.1 from NVIDIA's package index, OmniGibson 3.9.0 and the
3.9.0 dataset:

```bash
git clone -b v3.9.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --dataset \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
conda activate behavior
```

* `--new-env` creates the conda environment `behavior` (Python 3.11).
* `--dataset` downloads `behavior-1k-assets-3.9.0.zip` from the Hugging Face dataset
  `behavior-1k/zipped-datasets` and the dataset key `omnigibson.key`. Accepting
  `--accept-dataset-tos` means you accept the BEHAVIOR Data License.
* The dataset lands in `BEHAVIOR-1K/datasets` (`behavior-1k-assets/` and
  `omnigibson.key`). Export `OMNIGIBSON_DATA_PATH=/big/disk/behavior-data` before running
  the installer to put it elsewhere.

InteractiveWorld requires exactly BEHAVIOR-1K **3.9.0**: the release pins the encrypted
source of every scene-bound object, and `prepare_assets.sh` refuses any other version.

## Step 2: InteractiveWorld

```bash
git clone https://github.com/facebookresearch/InteractiveWorld.git
cd InteractiveWorld
pip install -r requirements.txt       # smplx; everything else comes with OmniGibson
export ISAAC_PYTHON=$(which python)   # the Python of the `behavior` environment
python3 -B validate_release.py        # static integrity check, no Isaac Sim needed
```

The launch scripts run Isaac Sim through `ISAAC_PYTHON`, so the shell that runs them does
not need to be inside the conda environment. `ISAAC_ENV_DIR` (optional) adds
`$ISAAC_ENV_DIR/lib` to the library path, which is needed on systems whose system
`libstdc++` is older than the one in the conda environment.

## Step 3: Human model files

The animated human uses the SMPL+H body model and one motion-capture walk. Both are
licensed for non-commercial research use and cannot be redistributed, so you download
them yourself:

| File | How to get it | Used for |
| :--- | :--- | :--- |
| `mano_v1_2.zip` | Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), open *Downloads* and take **Models & Code**. The archive contains `mano_v1_2/models/SMPLH_male.pkl`. | Body and hand mesh, skinning weights and joints |
| `ACCAD.tar.bz2` | Register at [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de), open *Downloads*, find **ACCAD** and take the **SMPL-X N** archive. It contains `ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz`. | The walking gait |

The grasp atlas (three hand-pose families and six object definitions) ships with the
repository under `simulator/human_model/assets/`.

## Step 4: Prepare the assets

```bash
./launch/prepare_assets.sh \
    --behavior-data ../BEHAVIOR-1K/datasets \
    --smplh ~/Downloads/mano_v1_2.zip \
    --accad ~/Downloads/ACCAD.tar.bz2 \
    --output ~/interactiveworld-assets \
    --workers 8 --gpus 0
```

| Option | Meaning |
| :--- | :--- |
| `--behavior-data` | The OmniGibson data directory: it contains `behavior-1k-assets/` and `omnigibson.key` |
| `--smplh` | `mano_v1_2.zip`, or the `SMPLH_male.pkl` extracted from it |
| `--accad` | The ACCAD `SMPL-X N` archive, or the folder it extracts to |
| `--output` | Destination for everything below |
| `--mdl-dir` | OmniGibson material folder; found automatically from the installed `omnigibson` package (`BEHAVIOR-1K/OmniGibson/omnigibson/materials`) |
| `--workers`, `--gpus` | Parallel Isaac Sim bake processes and the physical GPUs they use |

The script is idempotent: rerunning it skips finished work, and you can run the three
parts separately (pass only `--behavior-data`, only `--smplh` or only `--accad`; the
environment file keeps previous entries). It produces:

```text
~/interactiveworld-assets/
├── behavior-cache/
│   ├── usd/<category>/<model>.usd     6,831 decrypted objects with physics stripped
│   ├── spec/<category>/<model>.json   bounding box, mass, role and joints of each object
│   └── materials/*.mdl                OmniGibson MDL materials (without them surfaces render red)
├── human/
│   ├── SMPLH_male.pkl                 chumpy-free SMPL+H model (sha256 769f2655…)
│   └── ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz   (sha256 fbe0bb5e…)
└── env.sh                             exports every variable below
```

Before baking, the script checks the dataset version, the key and the encrypted source of
the 100 objects that TaskPacks bind by digest. Baked files embed the absolute path of
your dataset, so their bytes differ from ours; the simulator maps a bake of an identical
source back to the frozen digest (see `simulator/world_catalog/behavior_assets.json`).

Finally:

```bash
source ~/interactiveworld-assets/env.sh
./launch/simulator.sh doctor
```

`doctor` checks the Isaac Python, every external asset with its checksum, the bundled
label images and the bundled grasp atlas, and exits with status 1 if anything is missing.

## Environment variables

`env.sh` sets the required variables; the others are optional.

| Variable | Required | Meaning |
| :--- | :---: | :--- |
| `ISAAC_PYTHON` | ✓ | Isaac Sim 5.1 Python executable (absolute path) |
| `ISAAC_ENV_DIR` | | Environment root whose `lib/` is prepended to `LD_LIBRARY_PATH` |
| `INTERACTIVEWORLD_BEHAVIOR_DATA` | ✓ | BEHAVIOR-1K data directory (`behavior-1k-assets/`, `omnigibson.key`) |
| `INTERACTIVEWORLD_BEHAVIOR_CACHE` | ✓ | Baked cache (`usd/`, `spec/`, `materials/`) |
| `INTERACTIVEWORLD_SMPLH_MODEL` | ✓ | Converted `SMPLH_male.pkl` |
| `INTERACTIVEWORLD_WALK_CLIP` | ✓ | `B3_-_walk1_stageii.npz` |
| `INTERACTIVEWORLD_MDL_PATH` | | Override for the MDL material folder (default `<cache>/materials`) |
| `INTERACTIVEWORLD_MANO_PYTHON` | | A [MANO](https://github.com/otaheri/MANO) checkout to use instead of `smplx`; both give identical results |
| `INTERACTIVEWORLD_HUMAN_ASSET_ROOT` | | Override for the bundled grasp atlas |
| `INTERACTIVEWORLD_OUTPUT_DIR` | | Runtime outputs of the low-level runners (default `/tmp/interactiveworld-output`) |
| `INTERACTIVEWORLD_RUN_ROOT` | | Parent of benchmark run directories when `--output-dir` is not given |

`simulator/external_dependencies.json` is the machine-readable form of this table.

## Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `doctor` lists a path under `missing` | Rerun `prepare_assets.sh` for that part, then `source env.sh` again |
| `BEHAVIOR-1K 3.9.0 is required, found …` | Reinstall the dataset from a `v3.9.0` checkout of BEHAVIOR-1K |
| Surfaces render flat red | `<cache>/materials` is missing; rerun `prepare_assets.sh --behavior-data …` or set `INTERACTIVEWORLD_MDL_PATH` |
| `exact scene instance … does not resolve uniquely` | The cache was baked from a different dataset; rebake from BEHAVIOR-1K 3.9.0 |
| `Install smplx in the Isaac Sim Python` | Run `pip install -r requirements.txt` inside the `behavior` environment |
| `GLIBCXX_3.4.30 not found` | `export ISAAC_ENV_DIR=$CONDA_PREFIX` so the environment's `libstdc++` is used |
| `release integrity preflight failed` | Files inside the repository changed. Keep your own files outside the repository and run `git status` |
| Warnings about deprecated extensions or `GLIBC_2.35` at Isaac start-up | Harmless: they come from optional robot extensions that InteractiveWorld does not use |
| Converted `SMPLH_male.pkl` has the wrong checksum | Use the *Models & Code* archive (not the separate SMPL+H download) and run the script with the Isaac Sim Python (Python 3.11, numpy 1.26) |

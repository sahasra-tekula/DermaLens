# Data pipeline

## Dataset path and files

The local machine-specific paths live in the ignored `dataset_config.json`. Copy
`dataset_config.example.json` and edit its paths on another machine. The current
configuration reads `HAM10000_metadata.csv` and the two raw JPEG folders from
`D:/Datasets/DermaLens/data`. It writes resized image files to
`D:/Datasets/DermaLens/cache_224`, outside the project, OneDrive, and the source
dataset.

The raw audit found 10,015 metadata rows and 10,015 readable 600×450 JPEGs, with
no missing or unreferenced image files. All expected diagnosis labels are
present. There are 7,470 unique lesion IDs; 57 `age` values are missing. No
other metadata fields are missing. `age`, `sex`, `localization`, and `dx_type`
are not model inputs. The model input is the dermatoscopic image pixels; `dx`
is the target label.

## Exact duplicate handling

Images are compared by SHA-256 over the original file bytes. For each exact
duplicate group, the lexicographically smallest `image_id` is kept and the
other copy is removed from the manifest before splitting or caching. The
source images are left unchanged. The two excluded copies are:

| Kept image | Excluded image | Lesion ID | Label |
|---|---|---|---|
| `ISIC_0024366` | `ISIC_0029861` | `HAM_0002300` | `nv` |
| `ISIC_0024777` | `ISIC_0029938` | `HAM_0001895` | `nv` |

The excluded rows and hashes are also recorded in
`reports/exact_duplicate_images.csv`. The resulting manifest contains 10,013
images.

## Resize cache and data loaders

Each retained source image is converted to RGB, resized once to 224×224 using
bilinear interpolation, and stored as an individual JPEG (quality 95) in the
external cache. Data loaders read these files from disk, so the whole image
dataset is never loaded into RAM. Training-time flips, 15-degree rotation, and
mild colour jitter happen after loading; validation and test images get only
ImageNet normalization. `dx` is converted using the fixed class order
`akiec, bcc, bkl, df, mel, nv, vasc`.

The image cache stores resized pixels, not CNN feature vectors. Feature caching
for the first training stage will not support image augmentation; the later
layer4 fine-tuning stage will load images and apply augmentation.

## Split method

Splits use `StratifiedGroupKFold` with seed 42: a 7-fold grouped holdout first
sets aside test data, then a 6-fold grouped split makes validation data from
the remaining rows. The result is approximately 70/15/15. Each `lesion_id` is
contained in exactly one split; a test checks for overlap. The split is
**lesion-level, not patient-level**, because HAM10000 provides no patient ID.
No patient-level separation can be claimed.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `akiec` | 233 | 47 | 47 | 327 |
| `bcc` | 367 | 74 | 73 | 514 |
| `bkl` | 785 | 157 | 157 | 1,099 |
| `df` | 82 | 16 | 17 | 115 |
| `mel` | 795 | 159 | 159 | 1,113 |
| `nv` | 4,788 | 958 | 957 | 6,703 |
| `vasc` | 101 | 20 | 21 | 142 |
| **Images** | **7,151** | **1,431** | **1,431** | **10,013** |

The split CSVs contain only `image_id`, `lesion_id` (group bookkeeping), and
`dx` (the target). They contain no demographic or localization features.
Per-split class counts are printed when preparing data and saved in
`reports/split_class_counts.csv`.

## Rebuild the manifest, splits, and cache

After setting local paths in `dataset_config.json`, run from the project root:

```powershell
python -m src.data_pipeline --config dataset_config.json
```

The command writes `splits/train.csv`, `splits/val.csv`, and `splits/test.csv`,
small count and summary reports under `reports/`, and the resized image cache
under the configured external cache path. The sample-image montage is also
stored in the external cache folder so source image content is not committed.
The command skips valid cached images on subsequent runs. The committed split
CSVs were generated from the current dataset with seed 42.

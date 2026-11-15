# Training and evaluation

Training is CPU-only on this machine. The official torchvision
`ResNet18_Weights.IMAGENET1K_V1` weights are cached under the external
checkpoint directory configured in `dataset_config.json`.

## Two-stage training

1. **Stage A:** keep the pretrained ResNet18 backbone and all BatchNorm layers
   in eval mode. Extract train and validation features once into memory-mapped
   `.npy` files under `cache_root/resnet18_features/`. Train only the 7-class
   linear head using those cached features.
2. **Stage B:** start from the best Stage A head, unfreeze `layer4` and the
   head, and train on cached 224×224 images with horizontal/vertical flips,
   15-degree rotation, and mild colour jitter. Earlier backbone blocks and
   their BatchNorm layers remain frozen and in eval mode.

Class weights are computed only from `splits/train.csv`. Both stages use
AdamW, a ReduceLROnPlateau scheduler, and early stopping on validation
macro-F1. The defaults are 30 Stage A epochs (patience 5) and 6 Stage B epochs
(patience 3). Every completed epoch writes `last.pt`; validation improvements
also write `best.pt`. Checkpoints are stored outside the repo at
`checkpoint_root/stage_a/` and `checkpoint_root/stage_b/` and include model,
optimizer, scheduler, epoch, best validation macro-F1, seed values, and RNG
states. Stage A checkpoints store the frozen backbone state and trained head;
Stage B checkpoints store the complete fine-tuned ResNet18 state.

Run Stage A:

```powershell
python -m src.train --stage a
```

Run one Stage B epoch to measure its actual duration before continuing:

```powershell
python -m src.train --stage b --max-new-epochs 1
```

Resume Stage B from its last completed epoch:

```powershell
python -m src.train --stage b --resume --max-new-epochs 1
```

Stage histories and curves are written under `artifacts/metrics/`.

## Model and threshold selection, then test

Compare the best Stage A and Stage B checkpoints using validation macro-F1;
validation melanoma recall breaks an exact tie. For the selected model, choose
the melanoma probability threshold that maximizes melanoma F1 on validation.
Freeze both decisions before running the test command:

```powershell
python -m src.evaluate --mode select
python -m src.evaluate --mode test
```

The test command refuses to run unless `model_selection.json` exists and it
refuses a second test evaluation after `test_evaluation_started.json` is
written. It saves the report, curves, confusion matrix, and missed-melanoma
case list under `artifacts/metrics/`. A missed-case contact sheet is saved in
the external D: cache for visual error review.

The 7-class metrics use argmax predictions. Melanoma precision/recall are also
reported with the validation-selected melanoma threshold; that threshold is an
educational operating point, not a clinical recommendation.

## Measured results

On the validation split, Stage A's best checkpoint was epoch 17
(macro-F1 `0.4328581480172276`, melanoma recall `0.46540880503144655`). Stage
B's best was epoch 5 (macro-F1 `0.49398133582719367`, melanoma recall
`0.5786163522012578`), so Stage B was selected by the predeclared validation
macro-F1 rule. Stage A stopped after epoch 22 without a new validation best;
Stage B reached its six-epoch cap.

The validation-only melanoma F1 threshold was `0.6005103588104248`. On
validation, this gave melanoma precision `0.52` and recall
`0.4088050314465409`, compared with argmax precision `0.36220472440944884`
and recall `0.5786163522012578`.

The frozen Stage B model was evaluated on the 1,431-image test split once:

| Metric | Result |
|---|---:|
| Accuracy | 0.6813417190775681 |
| Macro precision | 0.4787962011813805 |
| Macro recall | 0.6590088572767029 |
| Macro-F1 | 0.5357564583204374 |
| Macro-F1 bootstrap 95% CI | [0.4912987243588317, 0.5756394786290895] |
| Macro one-vs-rest ROC-AUC | 0.9117022440564034 |
| Melanoma one-vs-rest ROC-AUC | 0.7810658201811637 |

Always predicting `nv` gives test accuracy `0.6687631027253669` and macro-F1
`0.11450107681263459`. The model's accuracy is only slightly above that
majority-class baseline, while macro-F1 shows the additional class coverage.
Full per-class metrics, curves, counts, and the confusion matrix are in
`artifacts/metrics/metrics.json` and adjacent files.

The validation threshold traded recall for precision on the test split:
thresholded melanoma precision was `0.4251968503937008` and recall was
`0.33962264150943394`; unthresholded argmax precision was
`0.31620553359683795` and recall was `0.5031446540880503`. The threshold was
not changed after seeing test results. Visual review of the 105 melanomas
missed at the selected threshold found varied appearances: some were faint,
small, or diffuse; others had mixed pigmentation, hairs, or vignetting. Some
dark, irregular-looking examples were also missed. There was no single
consistent visual pattern that explained the errors. This is an image-level
error review, not a medical interpretation.

## Grad-CAM visualization

Grad-CAM uses the output of the final block in `layer4` from the selected
Stage B `best.pt`. Its heatmap is resized to the cached 224×224 input and
overlaid on the image. The example panels explain the class chosen by the
frozen melanoma threshold decision. They are drawn from the validation split;
the test split is not used again for visualization. All full-resolution panels
and the contact sheet are stored outside the repository at
`D:\Datasets\DermaLens\gradcam\`.

Grad-CAM shows model attention, not medical correctness. The visuals are
educational illustrations and do not validate a diagnosis. The two compact
README examples are validation images from HAM10000 and are attributed in the
README.

Generate the panels and the two compact README examples with:

```powershell
python -m src.gradcam --include-readme-examples
```

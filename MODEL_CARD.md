# AgeVision model card

## Model summary

AgeVision estimates apparent age from a single, front-facing face image. The deployed
model is an ImageNet-pretrained EfficientNet-B0 fine-tuned as a regression model. YuNet
detects and crops exactly one face before inference.

This is a portfolio and educational model. It is not an identity-verification system and
does not determine a person's verified chronological age.

## Training data

The model was trained on UTKFace. Age labels are parsed from filenames, and the repository
creates reproducible train, validation, and test manifests stratified by age group where
possible. UTKFace contains aligned face images and labels for age, binary gender, and a
limited set of ethnicity categories.

UTKFace's labels and categories may contain errors, are not necessarily self-identified,
and do not represent every population or image condition. The dataset is not distributed
with this repository. Anyone reproducing the project is responsible for reviewing and
following the dataset's terms.

## Training procedure

The deployed checkpoint uses:

- 224 x 224 input images
- Huber regression loss with moderate inverse-square-root age-group weighting
- Horizontal flips, color jitter, small affine transformations, and occasional grayscale
- AdamW with an initial learning rate of `3e-4` and weight decay of `1e-4`
- `ReduceLROnPlateau` with factor `0.3` and patience `2`
- Early-stopping patience of `4`
- Random seed `42`

The selected checkpoint was epoch 12 of a 15-epoch run. Training took approximately
45 minutes on an NVIDIA GeForce GTX 1080 Ti.

## Evaluation

Evaluation uses the untouched UTKFace test manifest and applies the same YuNet crop used
by the web application. YuNet detected one face in 3,549 of 3,556 images (99.8%). The seven
detection failures are excluded from prediction metrics.

| Metric | Result |
| --- | ---: |
| Mean absolute error | 4.664 years |
| Root mean squared error | 6.634 years |
| Median absolute error | 3.3 years |
| 90th-percentile absolute error | 10.82 years |

| Age group | MAE | Samples |
| --- | ---: | ---: |
| 0-12 | 1.736 | 512 |
| 13-19 | 3.644 | 177 |
| 20-29 | 3.647 | 1,101 |
| 30-39 | 5.055 | 678 |
| 40-49 | 6.342 | 336 |
| 50-59 | 6.545 | 344 |
| 60+ | 7.967 | 401 |

The complete gender and ethnicity slices are in
`reports/efficientnet_b0_improved_deployed_pipeline.json`.

## Intended use

- Educational demonstrations of computer-vision model development
- Approximate apparent-age estimation for non-consequential entertainment
- Reproducible comparison of age-estimation experiments

## Limitations and prohibited uses

Prediction error generally increases with age, and performance can vary with lighting,
pose, camera quality, occlusion, and demographic representation. The displayed benchmark
errors summarize the test dataset; they are not calibrated uncertainty for a particular
person or image.

Do not use AgeVision for medical, legal, employment, insurance, identity, eligibility,
age-gating, surveillance, or other consequential decisions. Do not treat its output as a
verified chronological age.

Uploaded images are processed for the prediction request and are not intentionally retained
by the application.

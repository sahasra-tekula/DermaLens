# Melanoma false-negative review

The frozen Stage B checkpoint and melanoma probability threshold were selected
using validation only. The held-out test split was then evaluated once. The
validation-selected threshold was `0.6005103588104248`.

At that threshold, 54 of 159 test melanomas were predicted as melanoma and 105
were missed (precision `0.4251968503937008`, recall `0.33962264150943394`). By
comparison, 7-class argmax predictions had melanoma precision
`0.31620553359683795` and recall `0.5031446540880503`. The threshold improved
precision but reduced recall on this test split. It was not retuned on test.

Visual review of the saved contact sheets showed heterogeneous errors. Some
missed images contain faint, small, or diffuse lesions; others have mixed
pigmentation, hairs, or circular vignetting. Several dark and visibly
irregular examples were also missed. There was no single image-level pattern
that accounted for the false negatives. These observations describe model
errors only and do not establish medical meaning.

The case IDs and probabilities are in `missed_melanoma_cases.csv`. Contact
sheets are stored outside the repository under
`D:/Datasets/DermaLens/cache_224/test_error_review/` to avoid copying dataset
images into the repo.

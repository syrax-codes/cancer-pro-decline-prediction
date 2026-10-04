Target: `drop_outcome` (GPH drops by 10+ points on chemotherapy), test n=360, positives 20.3%. * = final model (chosen by CV AUROC).

| Model                 |   CV AUROC (train) |   Test AUROC |   PR-AUC |   Accuracy |   Precision |   Recall |    F1 |   Threshold |
|:----------------------|-------------------:|-------------:|---------:|-----------:|------------:|---------:|------:|------------:|
| Logistic Regression * |              0.753 |        0.764 |    0.513 |      0.725 |       0.398 |    0.699 | 0.507 |       0.210 |
| KNN                   |              0.719 |        0.705 |    0.406 |      0.664 |       0.338 |    0.685 | 0.452 |       0.180 |
| SVM                   |              0.733 |        0.772 |    0.488 |      0.719 |       0.387 |    0.658 | 0.487 |       0.220 |
| Random Forest         |              0.741 |        0.750 |    0.457 |      0.714 |       0.375 |    0.616 | 0.466 |       0.230 |
| Gradient Boosting     |              0.738 |        0.739 |    0.438 |      0.622 |       0.309 |    0.699 | 0.429 |       0.180 |
| Voting Ensemble       |              0.752 |        0.761 |    0.485 |      0.683 |       0.361 |    0.726 | 0.482 |       0.200 |
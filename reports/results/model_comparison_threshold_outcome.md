Target: `threshold_outcome` (On-chemo GPH score is 40 or below), test n=360, positives 51.9%. * = final model (chosen by CV AUROC).

| Model                 |   CV AUROC (train) |   Test AUROC |   PR-AUC |   Accuracy |   Precision |   Recall |    F1 |   Threshold |
|:----------------------|-------------------:|-------------:|---------:|-----------:|------------:|---------:|------:|------------:|
| Logistic Regression * |              0.912 |        0.913 |    0.921 |      0.825 |       0.777 |    0.930 | 0.847 |       0.320 |
| KNN                   |              0.884 |        0.886 |    0.889 |      0.797 |       0.803 |    0.807 | 0.805 |       0.420 |
| SVM                   |              0.902 |        0.906 |    0.913 |      0.808 |       0.763 |    0.914 | 0.832 |       0.310 |
| Random Forest         |              0.903 |        0.911 |    0.913 |      0.825 |       0.795 |    0.893 | 0.841 |       0.440 |
| Gradient Boosting     |              0.905 |        0.910 |    0.914 |      0.822 |       0.797 |    0.882 | 0.838 |       0.410 |
| Voting Ensemble       |              0.910 |        0.914 |    0.920 |      0.825 |       0.790 |    0.904 | 0.843 |       0.380 |
# Cassava Disease Classification | ResNet18, PyTorch

Classification of cassava leaf photos into 5 classes (4 diseases + healthy) from the [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) competition. A pretrained ResNet18 is improved step by step, changing one thing at a time.

<a href="https://www.kaggle.com/code/ivanventyk/cassava-disease-classification-resnet18-pytorch"><img src="https://img.shields.io/badge/View_notebook_on-Kaggle-F8C840?labelColor=1F1F1F&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAE%2BUlEQVR42u2bWYgcVRSGv7pVMzi45kHREAiiPhmCJCgKUVFx3xANuAzuiAriAoI7LqgRNIK7EEQRI4oLjlGEiA7og6D4oBjFgDD4IKKIYtA409Xlwz0HD9Izdaumum73TF8omO7pqr7nv%2Bee%2F5z%2F3E52f34KPUYCFMAq4HtgQl4n%2F%2FtcB8iAR4Db5O8OQzRcyf%2FngC5LeJQBkLDEh2OZjxEAIwBGAIwAGAEwAmD%2BUSx3AMaWupe4kgxwP6kDlmxWmA3JPNOaC9Atq2WyITE%2BX8T9yUKxbJABSGSL5sC5cnUDY9IcsA%2FwNfAYMDsfCFnAJGIbfxPweI1n7ADuFeNrb4FZmUTacmDuyvduBm4O2cuywjmwB%2FAKcDlenKm1BfSG3fKQlN6KUL%2F2%2BwTwErDRfH9WYnxXaPtR4FbjRbWCoKI2bla%2F38arnLYSeAM4hv8kt7JIn8g8b5HtkgZ6TakH%2FCFesFefPUCNXwe8DhwSaLxuz3%2BAS%2BXeTN4vQr%2B4bD%2B2tfJnAa8K2HnA3BSgX4DzgU%2BoIcrGTHPVbTvAtcCUMT4NNP47YENd42MC4IwLbwKeFZctKhj%2FEXAcXravLcfHSIQ00o8DLwCXmEifBNBcBrwMXGnu6yx2JdqsPXLgQGC7MT4LML6Qz22SgJebZGkoiiF10zXAW8BhFWhOg%2FGNwBNVaK4tAIpA408DtgIrKtLcX8Ak8DYNt9%2Ba2gJj87iwM3v0KmCbGB9KcynwE3CSGD9Gw71H19D9JxgvyORKTU7%2FALDFpKahkf4b4HjgM3k917YkFlq4rJMV7pgrBw4H3gPuMvvVBRq%2FHTgW2LnYSN%2FvGKB1wxZx1Y8lodkAnCHVWUhyowVNBrwIXG3uy%2BnTaAoAHRfJ1SuQlRmvSdCD4jFJEzTXNg3aIiQxQbDMeAXxOuC5JmmubQCqCicK1i5JiqZo%2BZRJTE2wMKu8EfhAaG6uzUnErAY1NmwW48fbNj4mAJrX%2F45XbR2RDlfFAkCD21ciaECkw1ix%2B35%2FCwNEa7vFBsCZHGBZAaArvobIzdeYklgOHARcbFLgZbUFtJB6GDhUKDBbTgCoy%2B8PfFhBIRpoALqmHO5W8ILVwKfA0W2D4Bo23hlBxAUWNBoPDsBL3ecQJpQOFACFPGsHvrvzJvCDvBdS0mr1NwG8I1VhhxY6U66hlU%2BAJ4G1EtUvEDXoQmAmUNFxpkB6RnSBvN%2BxyjXk9jP4gwwqdqb4puprwJF4MVTL3KIkMCbyuTvwDZAxwnTEaAAg1VzXGJmLIdq8PBvf%2Fsrkc2Ug6HMmqaYkR4sBv%2FXYq4XZxw64HrjTrGRZcFQQTgamgYP7wRBNAZAusKpdU%2F4%2BJMrPbGBwVBDWCk0e1TQITbJA2f914lvxHaKfA4Oj9hNXCk2e2SQIbWeCOvFpvOb%2FLWEaoNLknsC7wDWEdZQHMhVWEHbi%2B%2FvTgSA4s52eB%2B5rgiZj1QK6er8Cp%2BKPtYXQpCZGOXAPvoGSLoYmY4uiTgLiJL7vn5lkaCGa1NhxGfA%2BsC81zzPGVoTsEbfbgRsIPN%2FXgyZX18kVBuG3APboy1NSDP1ZkSaPEJpcX5UhBunHEDrxbcCJwI8VaXIVvjF7ehUQBu3XIDrxL4QmvzQGhtDk3kKTV5gsdKgAsAwxgz8cMWUMDKHJBH%2F67G7zeqgAwET0XcB5QpMuEASlyfuBp%2FEtt3nt%2FBfcU2ClwFaupAAAAABJRU5ErkJggg%3D%3D&style=flat" alt="View notebook on Kaggle" height="28"></a>

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white&style=for-the-badge) ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white&style=for-the-badge) ![matplotlib](https://img.shields.io/badge/matplotlib-11557C?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNTYgMjU2Ij48Y2lyY2xlIGN4PSIxMjgiIGN5PSIxMjgiIHI9IjExMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjE0Ii8%2BPGc%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0ibTEwNy44IDM2IC4xLjcgMjAuOCA5MS4xLTEuMy43LTU5LjktNzEuOC0uNS0uNi42LS41QTk0IDk0IDAgMCAxIDEwNyAzNi4xeiIvPjxwYXRoIGZpbGw9IiNmZmYiIGQ9Im01NCA5MS41LjYuMyA3My43IDM1LjUtLjIgMS40LTgwLjUgMTQuNi0uNy4yLS4xLS44YTgzIDgzIDAgMCAxIDYuOC01MC41eiIvPjxwYXRoIGZpbGw9IiNmZmYiIGQ9Im0xMjcuNyAxMjcuMy43IDEuM0w5MCAxNTUuMmwtLjYuNS0uNC0uNmE0OCA0OCAwIDAgMS0zLjgtNi41bC0uMy0uNy43LS4zeiIvPjxwYXRoIGZpbGw9IiNmZmYiIGQ9Im0xMjguNyAxMjcuOSA5LjggNTcuNi4yLjctLjguMmE1OSA1OSAwIDAgMS0yMy0uN2wtLjgtLjIuMi0uNyAxMy01N3oiLz48cGF0aCBmaWxsPSIjZmZmIiBkPSJtMTI4LjUgMTI3LjQgNzMuNyA1Ny41LjYuNC0uNS42YTk0IDk0IDAgMCAxLTE1LjUgMTUuOGwtLjYuNS0uNS0uNi01OC4zLTczLjF6Ii8%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0ibTE1MC43IDExOS44LjIuN2EyNCAyNCAwIDAgMSAxLjIgNy41di44aC0yMy42bC4xLTEuOCAyMS40LTd6Ii8%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0ibTE1Mi4yIDYxLjQuNy4yYTcxIDcxIDAgMCAxIDE5LjMgMTFsLjYuNS0uNS41LTQzLjcgNTQuOS0xLjMtLjggMjQuNi02NS42eiIvPjwvZz48Y2lyY2xlIGN4PSIxMjgiIGN5PSIxMjgiIHI9IjYiIGZpbGw9IiNmZmYiLz48L3N2Zz4%3D&style=for-the-badge)

## 📊 Results

Validation set: 4,280 images (stratified 20% split). Single run per experiment.

| Version | Change vs previous | Accuracy | Macro-F1 |
|---|---|---|---|
| Baseline | Always predict the largest class (CMD) | 0.615 | 0.152 |
| v1 | ResNet18, 224 px, lr 0.001, 3 epochs, horizontal flip | 0.759 | 0.514 |
| v2 | lr 0.0003, 8 epochs, cosine schedule | 0.831 | 0.692 |
| v3 | Class weights in the loss | 0.827 | 0.712 |
| v4 | Stronger augmentation | 0.811 | 0.714 |
| v5 | 15 epochs | 0.825 | 0.723 |
| v6 | Class weights removed | 0.850 | 0.736 |
| **v7** | **Resolution 384 px** | **0.874** | **0.772** |

**Bold** = best result (the final model, v7).

Per-class F1 of v7: CMD 0.955, CBSD 0.794, CGM 0.774, Healthy 0.720, CBB 0.617.

- The largest gains came from training settings (v1 → v2) and higher resolution (v6 → v7). At 384 px, F1 improved for all 5 classes, and diseased leaves predicted as healthy dropped from 213 to 170.
- Class weights and stronger augmentation removed overfitting (train/val gap 15.8 → 5.3 points) and raised recall of rare classes, but changed the overall metrics by only about 0.02, close to run-to-run noise. Removing the weights again (v6) raised accuracy at the cost of rare-class recall.
- In the final model, validation loss stops improving after about epoch 9 while training loss keeps falling, so more epochs alone would not help.
- CBB, the smallest class (5% of data), remains the weakest.
- Some confidently "wrong" predictions look like mislabeled images, but only 83 such cases (<2% of validation) exist, so label noise explains only a small part of the errors.

<!-- >>> IMAGE: put v7_curves.png (from the Kaggle notebook's Output tab) into the images/ folder <<< -->
![Training curves of the final model](images/v7_curves.png)

*Final model (v7, 384 px): loss (left), accuracy and macro-F1 (right) per epoch. Validation loss stops improving after about epoch 9.*

<!-- >>> IMAGE: put v7_confusion_matrix.png (from the Kaggle notebook's Output tab) into the images/ folder <<< -->
![Confusion matrix of the final model](images/v7_confusion_matrix.png)

*Row-normalized confusion matrix of v7: each row shows how the images of one true class were predicted.*

<!-- >>> IMAGE: put v7_examples.png (from the Kaggle notebook's Output tab) into the images/ folder <<< -->
![Correct and wrong predictions of the final model](images/v7_examples.png)

*Two correct (green) and two wrong (red) v7 predictions per class.*

## 🛠️ Approach

1. **Data.** 21,397 images, all 800×600. Strong imbalance: CMD 61.5%, CBB 5.1%. Stratified 80/20 split (17,117 train / 4,280 validation).
2. **Preprocessing.** Images stretched to a square (no cropping, so lesions near the edge are kept), ImageNet normalization. Augmentation only on training data.
3. **Model.** ResNet18 pretrained on ImageNet with a new 5-class output layer, AdamW optimizer.
4. **Experiments.** Each version changes one thing (except v2), with a prediction written before the run and the outcome after it.
5. **Error analysis.** Confusion matrix, per-class report and a review of the most confident wrong predictions.

## 📏 Metric

Because one class covers 61.5% of the data, accuracy alone is misleading: always predicting that class already gives 0.615. Macro-F1 is reported alongside it. It averages F1 over the classes with equal weight, so weak performance on rare classes is visible.

## ⚠️ Limitations

- One run per experiment; differences around 0.02 may be noise. GPU training is not fully deterministic, so a re-run gives slightly different numbers. The notes inside the notebook were written after an earlier run; this README uses the numbers of the published run.
- All versions were compared on the same validation split, with no separate test set and no Kaggle leaderboard submission, so the final score is slightly optimistic.
- v2 changed three settings at once, so their individual effects are unknown.
- The dataset has known label noise; judging which labels are wrong was done by eye, not by an expert.
- Only ResNet18 was tried, and only 224 and 384 px resolutions.

## ▶️ How to run

Open the notebook on Kaggle with the competition data attached, enable GPU and Internet (needed to download pretrained weights), and Run All (about 3 hours, because 7 models are trained).
Outputs in `/kaggle/working`: `results.csv` (per-class metrics of v5–v7), `v7_history.csv`, the three figures above, and model weights `v6_resnet18.pth`, `v7_resnet18_384.pth`.

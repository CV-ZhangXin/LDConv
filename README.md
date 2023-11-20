
# AKConv: Convolutional Kernel with Arbitrary Sampled Shapes and Arbitrary Number of Parameters ([preprint](https://doi.org/10.48550/arXiv.2304.03198))
This repository is a PyTorch implementation of our paper: AKConv: Convolutional Kernel with Arbitrary Sampled Shapes and Arbitrary Number of Parameters.
The relevant interpolation codes and resampling codes are referenced at https://github.com/dontLoveBugs/Deformable_ConvNet_pytorch.

| Models    | AKConv | AP50 | AP75 | AP   | APS  | APM  | APL  | GFLOPS | Params (M) |
|-----------|--------|------|------|------|------|------|------|--------|------------|
| YOLOv5n   | -      | 45.6 | 28.9 | 27.5 | 13.5 | 31.5 | 35.9 | 4.5    | 1.87       |
|           | 3      | 47.8 | 31   | 29.8 | 14.5 | 33.2 | 41   | 3.8    | 1.51       |
|           | 5      | 48.8 | 32.6 | 31   | 14.6 | 34.1 | 43.2 | 4.1    | 1.65       |
|           | 9      | 50.5 | 33.9 | 32.3 | 14.9 | 36.1 | 44.1 | 4.8    | 1.94       |
|           | 13     | 51.2 | 34.5 | 33   | 15.7 | 36.3 | 45.6 | 5.5    | 2.23       |
| YOLOv5s   | -      | 57   | 39.9 | 37.1 | 20.9 | 42.4 | 47.8 | 16.4   | 7.23       |
|           | 4      | 58.2 | 41.9 | 39.2 | 21.4 | 43.2 | 53.4 | 14.1   | 6.01       |
|           | 6      | 59.2 | 42.6 | 39.9 | 21.5 | 44.2 | 54.7 | 15.3   | 6.55       |
|           | 7      | 59.4 | 43.2 | 40.4 | 21.5 | 44.6 | 55.1 | 15.9   | 6.82       |

| Models  | AKConv | Precision | Recall | mAP50 | mAP  | FLOPS | Params (M) |
|---------|--------|-----------|--------|-------|------|-------|------------|
| YOLOv5n | -      | 38.5      | 28     | 26.4  | 13.4 | 4.2   | 1.77       |
|         | 3      | 37.9      | 27.4   | 25.9  | 13.2 | 3.5   | 1.41       |
|         | 5      | 40        | 28     | 26.9  | 13.7 | 3.8   | 1.56       |
|         | 6      | 38.1      | 28.1   | 26.8  | 13.6 | 4     | 1.63       |
|         | 7      | 39.8      | 28.2   | 27.5  | 14.2 | 4.2   | 1.7        |
|         | 9      | 39.7      | 28.9   | 27.7  | 14.3 | 4.5   | 1.84       |
|         | 11     | 40.4      | 28.8   | 27.7  | 14.2 | 4.8   | 1.99       |
|         | 14     | 40        | 28.8   | 27.9  | 14.3 | 5.3   | 2.2        |


# AKConv: Convolutional Kernel with Arbitrary Sampled Shapes and Arbitrary Number of Parameters ([preprint](https://doi.org/10.48550/arXiv.2304.03198))
This repository is a PyTorch implementation of our paper: AKConv: Convolutional Kernel with Arbitrary Sampled Shapes and Arbitrary Number of Parameters.
The relevant interpolation codes and resampling codes are referenced at https://github.com/dontLoveBugs/Deformable_ConvNet_pytorch.


# Object detection based on COCO2017 and YOLOv5
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


# Object detection based on VisDrone-DET2021 and YOLOv5

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


# Comparison experiments
| Models                        | AP50 | AP75 | AP   | APS  | APM  | APL  | GFLOPS | Params (M) |
|-------------------------------|------|------|------|------|------|------|--------|------------|
| YOLOv5s                       | 54.8 | 37.5 | 35   | 19.2 | 40   | 45.2 | 16.4   | 7.23       |
| YOLOv5s (DSConv =5)           | 43.2 | 23.5 | 23.9 | 13.0 | 27.6 | 30.5 | 14.8   | 6.45       |
| YOLOv5s (AKConv=5)            | 56.6 | 40.7 | 38   | 20.8 | 41.8 | 52   | 14.8   | 6.54       |
| YOLOv5s (AKConv=9)            | 57.8 | 41.4 | 38.7 | 20.8 | 42.8 | 52.3 | 17.1   | 7.37       |
| YOLOv5s (AKConv=9, padding)   | 58.3 | 41.9 | 39.2 | 21.6 | 43.2 | 53.5 | 17.1   | 7.37       |
| YOLOv5s (Deformable Conv = 3) | 58.5 | 41.8 | 39.1 | 20.8 | 43.4 | 53.6 | 17.1   | 7.37       |
| YOLOv5s (AKConv=11)           | 58.5 | 42.1 | 39.3 | 21.9 | 43.3 | 53.8 | 18.3   | 7.91       |
| YOLOv5s (AKConv=11, padding)  | 58.6 | 42.1 | 39.5 | 21.3 | 43.7 | 53.2 | 18.3   | 7.91       |

# Comparison experiments
| Models             | Precision | Recall | mAP50 | mAP  | GFLOPS | Params (M) |
|--------------------|-----------|--------|-------|------|--------|------------|
| YOLOv5n            | 73.8      | 62.2   | 68.1  | 41.5 | 4.2    | 1.77       |
| YOLOv5n (DSConv=4) | 63        | 50.4   | 54.2  | 26.1 | 3.7    | 1.55       |
| YOLOv5n (AKConv=4) | 76.5      | 63.6   | 70.8  | 46.5 | 3.7    | 1.55       |
| YOLOv5n (DSConv=9) | 60.6      | 50.8   | 53.4  | 25.3 | 4.8    | 1.9        |
| YOLOv5n (AKConv=9) | 76.7      | 65.2   | 71.8  | 48.4 | 4.8    | 1.9        |

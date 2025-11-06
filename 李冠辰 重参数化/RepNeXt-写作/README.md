# 1 Experiments on ImageNet Classification

|      模型       |  image size   | #param. | FLOPs | 推理速度 | top1 acc | top5 acc |
| :-------------: | :-----------: | :-----: | :---: | :------: | :------: | :------: |
|    resnet50     | 3 × 224 × 224 |         |       |          |          |          |
|   effnet系列    | 3 × 224 × 224 |         |       |          |          |          |
|    ViTs系列     | 3 × 224 × 224 |         |       |          |          |          |
|   SwinT_tiny    | 3 × 224 × 224 |         |       |          |   81.3   |          |
|  ConvNeXt_tiny  | 3 × 224 × 224 |         |       |          |   82.1   |          |
| ConvNeXtER_tiny | 3 × 224 × 224 |         |       |          |   81.6   |          |
| RepNeXtV1_tiny  | 3 × 224 × 224 |         |       |          |   81.7   |          |
| RepNeXtV2_tiny  | 3 × 224 × 224 |         |       |          |   81.8   |          |
| RepNeXtV3_tiny  | 3 × 224 × 224 |         |       |          |   81.9   |          |

# 2 Experiments on Fine-grained and Low Resolution Image Classification

## ImageWoof

| 模型                       | image size    | #param.         | FLOPs          | 推理速度(it/s, in RTX2060Mini) | top1 acc%       | top5 acc% |
| -------------------------- | ------------- | --------------- | -------------- | ------------------------------ | --------------- | --------- |
| ResNet50                   | 3 × 224 × 224 | 25.6M           | 4.1B           | 73.7                           | 87.7            | 98.7      |
| ConvNeXt_tiny              | 3 × 224 × 224 | 28.6M           | 4.5B           | 78.7                           | 82.3——88.0（3） | 98.0      |
| RepNeXt_tiny(ours)         | 3 × 224 × 224 | 28.4M --> 51.6M | 4.4B  --> 8.2B | 65.4 --> **101.6**             | **90.5**        | **99.0**  |
| RepNeXtV2_tiny             | 3 × 224 × 224 | 28.4M --> 51.6M | x.xB  --> 8.2B | x.x --> **101.6**              | 90.7            |           |
| RepNeXtV3_tiny             |               |                 |                |                                | 90.8            |           |
| RepLKNet_tiny(without rep) | 3 × 224 × 224 | 44.0M           |                |                                | 84.9            |           |
| RepLKNet_tiny(with 3rep)   | 3 × 224 × 224 | 44.0M           |                |                                | 85.6            |           |

## cifar100

| 模型             | image size  | #param.           | FLOPs             | 推理速度(it/s, in RTX2060Mini) | top1 acc   | top5 acc   |
| ---------------- | ----------- | ----------------- | ----------------- | ------------------------------ | ---------- | ---------- |
| resnet50         | 3 × 96 × 96 | 25.56M            | 756M              | 93.23                          | 80.51%     | 95.03%     |
| efficientnetv2_s | 3 × 96 × 96 | 22.10M            | 537M              | 34.98                          | 80.40%     | 94.64%     |
| ConvNeXt_tiny    | 3 × 96 × 96 | 28.58M            | 822M              | 94.98                          | 81.59%     | 94.98%     |
| RepNeXt-tiny     | 3 × 96 × 96 | 28.44M --> 51.55M | 814.15M --> 1.51B | 78.11 --> **126.82**           | **82.98%** | **96.03%** |

# 3 Experiments on Downstream Tasks

## COCO object detection and segmentation using Mask-RCNN and Cascade Mask-RCNN

| 模型baseline  | FLOPS | FPS  | AP(box) | AP(box50) | AP(box75) | AP(mask) | AP(mask50) | AP(mask75) |
| ------------- | ----- | ---- | ------- | --------- | --------- | -------- | ---------- | ---------- |
| resnet系列    |       |      |         |           |           |          |            |            |
| mobilenet系列 |       |      |         |           |           |          |            |            |
| effnet系列    |       |      |         |           |           |          |            |            |
| ViTs系列      |       |      |         |           |           |          |            |            |
| Swin系列      |       |      |         |           |           |          |            |            |
| ours          |       |      |         |           |           |          |            |            |

## Semantic segmentation on ADE20K

| 模型baseline | FLOPS | #param. | input crop. | mIoU |
| ------------ | ----- | ------- | ----------- | ---- |
| resnet系列   |       |         |             |      |
| Swin系列     |       |         |             |      |
| ours         |       |         |             |      |

# 3 Pruning experiment




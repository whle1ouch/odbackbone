## Models

models including top such as:

* vgg： classic deep convolution with scale down and channel increase

* resnet v1, v2：Pioneeringly, residual connections were introduced in neural networks

* resnext：introducing group convolution in resnet

* mobilenet v1, v2:   split a conv to separable conv and pointwise conv to reduce parameters, named depthwise conv.

* mobilenet v3:  come up with some new features to mobilenet such Sequeeze-Excitation、hard-swish activation，dropout, etc

* inception v1, v2:  group convolution with different struct, and finaly concatenate the results in channels axis to build feature map.

* inception v3:  optimization of inception, keep 3 kind block structure of group conv

* inception v4(resnet-inception v1):  residual connection introduced in inception

* resnet-inception v2: further optimization of inception v4.

* xception: introducing separable convolution which in depthwise convolution of mobilenet.

* efficientnet: a group of network proposed by google using its network architecture search(NAS) strategy

* densenet：propose the dense shortcut connection used in CNN instead of residual connection

  

and models only having backbones such as:

* fpn: used in Fast-RCNN, a simple feature pyramid extractor with deep cnn
* resnet-fpn: used in Faster-RCNN, introducing residual connection in fpn
* darknet53: used in yolo v3,  with 5 residual block and 5 times downscale
* cspdarknet53: used in yolo v4 and yolo v5, introducing some features like csp block, focus block, mish activation.

## Parameters

including tops:

| Model               | Input Size    | Total parameters | Trainable parameters |
| ------------------- | ------------- | ---------------- | -------------------- |
| vgg16               | 224 * 224 * 3 | 138,357,544      | 138,357,544          |
| vgg19               | 224 * 224 * 3 | 143,667,240      | 143,667,240          |
| resnet50            | 224 * 224 * 3 | 25,636,712       | 25,583,592           |
| resnet50v2          | 224 * 224 * 3 | 25,613,800       | 25,568,360           |
| resnet101           | 224 * 224 * 3 | 44,707,176       | 44,601,832           |
| resnet101v2         | 224 * 224 * 3 | 44,675,560       | 44,577,896           |
| resnet152           | 224 * 224 * 3 | 60,419,944       | 60,268,520           |
| resnet152 v2        | 224 * 224 * 3 | 60,380,648       | 60,236,904           |
| resnext50           | 224 * 224 * 3 | 26,970,024       | 25,965,352           |
| mobilenet           | 224 * 224 * 3 | 4,253,864        | 4,231,976            |
| mobilenet v2        | 224 * 224 * 3 | 3,538,984        | 3,504,872            |
| mobilenet v3 small  | 224 * 224 * 3 | 2,554,968        | 2,542,856            |
| mobilenet v3 Large  | 224 * 224 * 3 | 5,507,432        | 5,483,032            |
| Inception           | 224 * 224 * 3 | 7,000,476        | 7,000,476            |
| inception v2        | 224 * 224 * 3 | 11,152,840       | 11,133,064           |
| inception v3        | 299 * 299 * 3 | 23,851,784       | 23,817,352           |
| inception-resnet    | 299 * 299 * 3 | 42,711,400       | 42,648,232           |
| inception-resnet v2 | 299 * 299 * 3 | 55,873,736       | 55,813,192           |
| densenet121         | 224 * 224 * 3 | 8,062,504        | 7,978,856            |
| densenet169         | 224 * 224 * 3 | 14,307,880       | 14,149,480           |
| densenet201         | 224 * 224 * 3 | 20,242,984       | 20,013,928           |
| xception            | 299 * 299 * 3 | 22,910,480       | 22,855,952           |

only backbone:

| Backbone           | Input size    | Total parameters | Trainable parameters |
| ------------------ | ------------- | ---------------- | -------------------- |
| fpn                | 208 * 208 * 3 | 14,306,048       | 14,306,048           |
| resnet-fpn         | 600 * 600 * 3 | 23,587,712       | 23,534,592           |
| darknet53          | 416 * 416 * 3 | 40,620,640       | 40,584,928           |
| cspdarknet53       | 416 * 416 * 3 | 26,652,512       | 26,617,184           |
| cspdarknet53-focus | 608 * 608 * 3 | 26,633,792       | 26,599,872           |
|                    |               |                  |                      |


# Train MobileNetV2 and ShuffleNetV2 on CIFAR10 with [PyTorch](http://pytorch.org/)

Credits to [kuangliu](https://github.com/kuangliu/pytorch-cifar)

## Prerequisites
- Python 3.7.10+
- PyTorch 1.10.0+

## Training
```
# Start training with: 
python main.py --model=<shufflenet_v2|mobilenet_v2> --lr=0.01 --epochs=50 

# Resume the training with: 
python main.py --resume --model=<shufflenet_v2|mobilenet_v2> --lr=0.01 --epochs=50 
```

## Accuracy
| Model             | Validation Acc.        |
| ----------------- | ---------------------- |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 85.18%      |
| [ShuffleNetV2](https://arxiv.org/abs/1807.11164)      | 81.92%      |


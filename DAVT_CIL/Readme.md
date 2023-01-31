## Environments
- Python: 3.7.0
- PyTorch: 1.12.0
- einops
- torchvision
- numpy
- pandas
- tqdm

## Training
### Training for CIFAR100
Please set CIFAR100 dataset root in 'class_incremental_cosine_cifar100.py' lines 126-131.
```
python ./cifar100/class_incremental_cosine_cifar100.py --resume

```
### Training for ImageNet-100 & ImageNet
Please set your dataset root in 'class_incremental_cosine_imagenet.py' line 43.
```
python ./imagenet/class_incremental_cosine_imagenet.py --resume

```
### Training for tiny ImageNet
Please set your dataset root in 'tini_imagenet.py' line 45.

```
python ./tinyimagenet/tini_imagenet.py --resume
```

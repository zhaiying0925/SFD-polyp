
# Learning Diverse and Realistic Polyp Data via Style-Fused Diffusion Models

## Prerequisites

- `Python 3.10.0`
- `Pytorch 2.1.0`

This code has been tested using `Pytorch` on a V100-32GB GPU.

## Dataset

Our experiments are conducted on five public polyp datasets: Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, and ETIS-LaribPolypDB. To maintain consistency with the experimental setup, we follow the data partitioning scheme of the PraNet benchmark: 900 images from Kvasir-SEG and 550 images from CVC-ClinicDB are used as the training set, while the remaining samples along with the other three complete datasets (CVC-ColonDB, CVC-300, and ETIS-LaribPolypDB) serve as the test set.

## Training and Testing
```
# Training
python train.py

# Testing
python process.py
python inference.py

```


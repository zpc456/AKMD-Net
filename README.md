AKMD-Net
=====
Codes for AAAI 2025 submitted paper, "AKMD-Net: A Uniform Network for Real-World RS Image Reconstruction".

Getting Started
----
**Requirements:**
* PyTorch >= 1.7
* Python >= 3.7
* NVIDIA GPU + CUDA
* Linux
  
Prepare for Dataset
-----
```
cd models/data/
jupyter notebook --dataset_processing.ipynb
```

Networks
-----
python models/networkd/AKMDNet_arch.py

To Train
-----
cd models
python train.py

To Test
-----
cd models
python test.py


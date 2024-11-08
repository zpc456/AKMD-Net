AKMD-Net
=====
Codes for paper, "Jointly RS Image Deblurring and Super-Resolution
with Adjustable-Kernel and Multi-Domain Attention".

![image](https://github.com/user-attachments/assets/26b24658-94b5-4769-baf5-de35e978f792)


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
lynx test_three_dataset.html
```

Networks
-----
```
python models/networks/AKMD_arch4x.py
```

To Train
-----
```
cd models
python models/train.py -opt configs/train_AKMFNet_RSSCN7x2.yml
```
![image](https://github.com/user-attachments/assets/7bf4c420-8633-41ed-bed5-934e301b14ac)

To Test
-----
```
cd models
python models/test.py -opt configs/test_AKMFNet_RSSCN7x2.yml
```


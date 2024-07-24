AKMD-Net
=====
Codes for AAAI 2025 submitted paper, "AKMD-Net: A Uniform Network for Real-World RS Image Reconstruction".

![image](https://github.com/user-attachments/assets/c81d94ba-ee79-4567-b5ec-9e0c996d63fe)

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
python train.py
![image](https://github.com/user-attachments/assets/7678f258-5968-44f3-b207-2438cf5234f5)


```

To Test
-----
```
cd models
python test.py
```


# DLbased-IDPC

This repository is the implementation of the U-net model for DL-based Isotropic Quantitative Differential Phase Contrast microscopy (DL-based IDPC) by An-Cin Li from the Institute of Medical Device and Imaging, college of medicine, NTU.

## Patch-based U-net Model for Isotropic Quantitative Differential Phase Contrast Imaging
### Flowchart
![DL-based IDPC flowchart](https://github.com/ricky85917/DLbased-IDPC/blob/main/ReadmeImg/Flowchart.PNG)

### Model Architecture
![model architecture](https://github.com/ricky85917/DLbased-IDPC/blob/main/ReadmeImg/Model%20Architecture.PNG)
>Quantitative differential phase-contrast (qDPC) imaging is a label-free phase retrieval method for weak phase objects using asymmetric illumination. However, qDPC imaging with fewer intensity measurements leads to anisotropic phase distribution in reconstructed images. In order to obtain isotropic phase transfer function, multiple measurements are required; thus, it is a time-consuming process. Here, we propose the feasibility of using deep learning (DL) method for isotropic qDPC microscopy from the least number of measurements. We utilize a commonly used convolutional neural network namely U-net architecture, trained to generate 12-axis isotropic reconstructed cell images (i.e. output) from 1-axis anisotropic cell images (i.e. input). To further extend the number of images for training, the U-net model is trained with a patch-wise approach. In this work, seven different types of living cell images were used for training, validation, and testing datasets. The results obtained from testing datasets show that our proposed DL-based method generates 1-axis qDPC images of similar accuracy to 12-axis measurements. The quantitative phase value in the region of interest is recovered from 66% up to 97%, compared to ground-truth values, providing solid evidence for improved phase uniformity, as well as retrieved missing spatial frequencies in 1-axis reconstructed images. In addition, results from our model are compared with paired and unpaired CycleGANs. Higher PSNR and SSIM values show the advantage of using the U-net model for isotropic qDPC microscopy. The proposed DL-based method may help in performing high-resolution quantitative studies for cell biology.


## Usage
### Dependencies
The program requires Python 3.x and the following packages:

Tensorflow-GPU v1.4.1

Keras v2.1.6

Scipy v1.0.0

Numpy v1.16.6

### Load and Test
  1.Prepare the folder with following files and folder  
      -DLbasedIDPC_predict.py  
      -DLbasedIDPC_weights.hdf5  
      -patches.py  
      -test_input/  
      -test_groundtruth/  
  2.Make sure each folder contains 24 .mat files for predicting and comparing the testing results  
  3.Run the following code to load the pretrained weights and predict the testing results  
  ```
  python DLbasedIDPC_predict.py
  ```
  4.Results from model prediction will be saved in the new *result_img* folder with 3 .jpg and 3 .mat files for visulization and further analysis

## Result
Visual comparisons for input 1-axis qDPC, 12-axis isotropic qDPC, and DL-based isotropic qDPC reconstructions
![DL-based IDPC flowchart](https://github.com/ricky85917/DLbased-IDPC/blob/main/ReadmeImg/Result1.PNG)

3D visualization 
![DL-based IDPC flowchart](https://github.com/ricky85917/DLbased-IDPC/blob/main/ReadmeImg/Result3D.PNG)

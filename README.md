# CodesignOAT
This repository provides code for Learning Optimal System Designs for Cost-Efficient and High-Performance Optoacoustic Imaging (Codesign OAT). The framework supports both 2D circular array and 3D hemispherical array configurations, with training and inference pipelines, pretrained models, and simulated/experimental datasets.

## 📁 Project Structure

##### CodesignOAT/ 

  ├── Data/ Data download instructions  
  ├── Models/ Pretrained model download instructions    
  ├── Output/ Sample output download instructions    
  ├── Src/ Core model definitions and helper functions    
  ├── Train/ Training entry point and configuration    
  ├── Test/ Testing scripts for evaluation and visualization    
  ├── environment.yml Python dependencies

## 🚀 One-Click Quick Start (CPU, No Environment Setup)
We provide a one-click executable for quick testing on 2D circular array (mouse abdomen) data. This package includes the test dataset, pretrained weights, and the required deep learning environment, so no additional installation is needed.

Note: The executable runs on CPU only, so inference may be slower.
1. Download the packaged executable [here](https://drive.google.com/file/d/1YbaCuTYaSIhNv8hoO3YQR1H5OYW3qs9W/view?usp=sharing).  
2. Extract the archive to obtain a QuickTest folder.
3. Run 'CodesignOA_2D_GUI.exe' inside the folder.
4. In the GUI, select the number of transducer elements. Once ready, the Run button will turn green. Click it to start inference and visualization.



## ⚡ Test with GPU (Full Deployment)
### 1. Install Dependencies
Make sure you have Python ≥ 3.8 + Pytorch ≥ 1.12.0 and install required packages:  
conda env create -f environment.yml

### 2. Download Datasets and Pretrained Models
- Data:
Due to dataset size limitations, only the experimental dataset based on the 2D circular array (mouse abdomen) is included in this repository for quick testing. Additional datasets for both 2D and 3D configurations can be downloaded from the following links:  
Test and training datasets for both 2D and 3D configurations are available [here](https://drive.google.com/drive/folders/1RqE8x5nnz4RmY9ixdjyOgK-ioEiB9I2l?usp=drive_link).    
Some 3D experimental data (>50GB) can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1f3kyXv5aas6rG763KdQJ-Q) (Code: 0000).    
- Models:
Pretrained models for both configurations are available [here](https://drive.google.com/drive/folders/1RqE8x5nnz4RmY9ixdjyOgK-ioEiB9I2l?usp=drive_link).  
Place the downloaded files into their respective folders as indicated in the training and test scripts.

### 3. Test Pretrained Models  
The following scripts evaluate pretrained models using both experimental and simulated data:  
- Run Test/Test_Exp2D_CircularArray.py – test on experimental mouse abdomen data using 2D Circular Array  
- Run Test/Test_Sim2D_CircularArray.py – test on simulated 2D Circular Array data
- Run Test/Test_Exp3D_HemisphericalArray.py – test on experimental human finger data using 3D Hemispherical Array  
*If you want to visualize the 3D stitched result, please run the MATLAB script at `Src/Stitch3D/Main.m`*  
- Run Test/Test_Sim3D_HemisphericalArray.py – test on simulated 3D Hemispherical Array data
This will evaluate the pretrained models on the test datasets and generate reconstructed images and metrics under Output/.

### 4. Train from Scratch
Use the following scripts to retrain the model for each array configuration:  
- Run Train/Train2D_CircularArray.py – retrain the model for 2D Circular Array  
- Run Train/Train3D_HemisphericalArray.py – retrain the model for 3D Hemispherical Array  
Make sure your data is correctly placed under Data/ and follow the expected folder structure.

## 🖼 Sample Results
You can find the sample results from [here](https://drive.google.com/drive/folders/1RqE8x5nnz4RmY9ixdjyOgK-ioEiB9I2l?usp=drive_link).

## 📬 Contact
For questions, please contact:
xuyang_chang@163.com
Or open an issue on this GitHub repository.

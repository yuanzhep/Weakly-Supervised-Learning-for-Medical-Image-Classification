Project: 
WSI Classification based on tree-WPD and pre-trained CNN

1. code path on GitHub: 
https://github.com/CaixdLab/WSI-Classification-using-DWT

2. data path on Cailabgpu1 server: 
/home/yuanzhe/Storage/tcga_save/

3. data intro: 
cd /home/yuanzhe/Storage/tcga_save/
tcga_save
|-- 0_wsi (WSI)
|   |-- luad
|   |-- lusc
|-- 1_wsi (unprocessed WSI)
|-- wavelet features_0408_2023 (big tiles' wavelet features after 7-level tree-WPD, Each csv file represents a bag (WSI), the rows of csv represent different instances (tiles), and the columns represent frequency band features.)
|   |-- tcga_wavelet_0408_2023
|   |   |--TCGA-22-5480-01Z-00-DX1.csv
|   |   |--TCGA-44-6778-01Z-00-DX1.csv
|   |   |--TCGA-44-A479-01Z-00-DX1.csv
|   |   |--...
|   |-- label.csv (bag label)
|-- crop_224_June14_inprogress (small tiles in progress)

4. code intro:
gdc_manifest.2023-05-10_LUSC.txt: LUSC metadata for gdc download
gdc_manifest.2023-05-10_LUAD.txt; LUAD metadata for gdc download
crop_2048_256.py: Crop 2048*2048 tiles to 256*256 tiles
crop_224_nonoverlap.py: Crop 2048*2048 tiles to nonoverlap 224*224 tiles
crop_224_overlap.py: Crop 2048*2048 tiles to overlap 224*224 tiles
Sizedistribution_2022_1125.pdf: Size distribution of WSI (updating by TCGA)
label.csv: TCGA luad/lusc bag label
crop_single20.py: generates tiles
wpd_0225.py: wavelet packet transform
wsi_wpd.py: wavelet packet transform
abmil.py: baseline abmil
clam.py: baseline CLAM
MIL_ElasticNet_v3_5.py: multiple instance learning with elasticnet

5. Required Environment
    - cycler==0.11.0
    - fonttools==4.38.0
    - idna==3.4
    - imageio==2.27.0
    - joblib==1.2.0
    - kiwisolver==1.4.4
    - matplotlib==3.5.3
    - numpy==1.21.6
    - packaging==23.0
    - pandas==1.3.5
    - pillow==9.5.0
    - plotly==5.14.1
    - pyparsing==3.0.9
    - python-dateutil==2.8.2
    - pytz==2023.3
    - pywavelets==1.3.0
    - requests==2.28.2
    - scikit-image==0.19.3
    - scikit-learn==1.0.2
    - scipy==1.7.3
    - seaborn==0.12.2
    - six==1.16.0
    - tenacity==8.2.2
    - torch==1.13.1
    - torchvision==0.14.1
    - tqdm==4.65.0
    - urllib3==1.26.15
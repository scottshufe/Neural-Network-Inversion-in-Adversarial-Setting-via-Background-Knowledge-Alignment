# Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment
The PyTorch implementation of (2019 CCS) Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment
 
## Data
Both the target classifier and the inversion model are trained on the MNIST dataset. The overall data is firstly splitted into Train (50%) / Test (50%) set for the target classifier, and then 50% of Test data are used to train the inversion model.

The data are resized to 32 × 32 as the authors did in the article.

## Run
Train the target classifier:
```
python train_target.py
```

Train the inversion model:
```
python train_inversion.py
```

## Acknowledgement
The official implementation on FaceScrub and CelebA datasets: [GitHub: adversarial-model-inversion](https://github.com/yziqi/adversarial-model-inversion)

The PyTorch implementation of the paper "(2023 TDSC) Boosting Model Inversion Attacks with Adversarial Examples": [GitHub: Adversarial_Augmentation](https://github.com/ncepuzs/Adversarial_Augmentation)

## Citation
```
@inproceedings{Yang:2019:NNI:3319535.3354261,
 author = {Yang, Ziqi and Zhang, Jiyi and Chang, Ee-Chien and Liang, Zhenkai},
 title = {Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment},
 booktitle = {Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security},
 series = {CCS '19},
 year = {2019},
 isbn = {978-1-4503-6747-9},
 location = {London, United Kingdom},
 pages = {225--240},
 numpages = {16},
 url = {http://doi.acm.org/10.1145/3319535.3354261},
 doi = {10.1145/3319535.3354261},
 acmid = {3354261},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {deep learning, model inversion, neural networks, privacy, security},
}
```
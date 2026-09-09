# Code for the CVPR 2022 paper "Person Re-identification Method Based on Color Attack and Joint Defence".

Prerequisites
·Python 3.6
·GPU Memory >= 6G
·Numpy
·Pytorch 0.3+ (http://pytorch.org/)
·Torchvision from the source

You can run pip install -r requirements.txt to install required packages or conda env create -f environment.yml to create a new environment with the required packages installed.

As we use Market1501 and DukeMTMC-reid datasets for our experiments, you must download them beforehand.

--------------------------------------------------------------------------Getting started-----------------------------------------------------------------------------
Part 1: Training

Part 1.1: Prepare Data Folder (python prepare.py)

You may notice that the downloaded folder is organized as:
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* We do not use it 
│   ├── gt_query/                   /* We do not use it 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt

Open and edit the script prepare.py in the editor. Change the fifth line in prepare.py to your download path, such as \home\Download\Market. Run this script in the terminal:

python prepare.py

We create a subfolder called pytorch under the download folder：
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* We do not use it 
│   ├── gt_query/                   /* We do not use it
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
│   ├── pytorch/
│       ├── train_all/               /* train     
│           ├── 0002
│           ├── 0007
│           ...
│       ├── val/                     /* val
│       ├── query/                   /* query files  
│       ├── gallery/                 /* gallery files  
In every subdir, such as pytorch/train/0001, images with the same ID are arranged in the folder. Now we have successfully prepared the data for torchvision to read the data.
-----------------

Part 1.2: Training (python train.py)

We can train a normally trained model by:

python train.py --gpu_ids 0 --name Normally_Trained --data_dir your_data_path --epoch 60

--gpu_ids: which gpu to run.
--name: the name of the model.
--data_dir: the path of the training data.
--epoch: the training epoch

The default used is Resnet50, you may apply '--use_dense' to use DenseNet.
The trained model will be saved in . /model

If you want to train a DL defense model, you can do as follows :
Change the 65-th line in train.py to apply the code 'Fuse_LFusePR(G=0.05, G_rgb=0.01, S_rgb=0.01,Aug = 0.05,F = 0.1)’.

python train.py --gpu_ids 0 --name DL --data_dir your_data_path --epoch 120
---------------------------------------------------------------------------------------------------------------------------------------------------
Part 2: Test

Part 2.1: Extracting feature (python test.py)

In this part, we load the network weight (we just trained) to extract the visual feature of every image.
python test.py --gpu_ids 0 --name Normally_Trained --test_dir your_data_path 

--name: the dir name of the trained model.

-----------------
Part 2.2: test with re-ranking.
python evaluate_gpu.py

Before using it，you must first run the 'python test.py'.It may take more than 10G memory to run. So run it on a powerful machine if possible.

---------------------------------------------------------------------------------------------------------------------------------------------------
Part 3: White-Box Attack

python aa_LTA.py --gpu_ids 0 --name (such as: DL) --test_dir your_data_path
--name: the name of the folder where the model you want to attack

The adversarial examples will be saved in . /adv_data, which is the adversarial version of the query set. Use it to replace the original query set and run 'test.py' and 'evaluate_gpu.py' to test the effect of the attack.
---------------------------------------------------------------------------------------------------------------------------------------------------
Part 4: Joint Adversarial Defense(JAD)

Before that, you need to train a DL model and perform a white-box attack on it to get adversarial examples.

Change the 78th-80th line in test.py to apply the code:
######## JAD
        transforms.Resize((110,50), interpolation=3),
        transforms.Resize((220,100), interpolation=3),
        transforms.Resize((110,50), interpolation=3),

This is our passive defense Circuitous Scaling. To run 'test.py' and 'evaluate_gpu.py' to test the effect of the JAD defence.

Considering the limitations of DL, it is better to train several DL models, test them separately for attacks and defenses, or use another dataset to test the cross-domain performance of DL models. In cross-domain tests, models that perform better than the normally trained models will have better defensive capabilities.


If you encounter any issues with reproducing adversarial defense, please refer to the earlier open-source version of the code: https://github.com/finger-monkey/ReID_Adversarial_Defense/

if you use our code, please  cite the following paper:

```
@inproceedings{colorAttack2022,
  title={Person re-identification method based on color attack and joint defence},
  author={Gong, Yunpeng and Huang, Liqing and Chen, Lifei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},# Person Re-identification Method Based on Color Attack and Joint Defence

> Official PyTorch implementation for the CVPR 2022 paper.  
> **Title**: Person Re-identification Method Based on Color Attack and Joint Defence  
> **Authors**: Yunpeng Gong, Liqing Huang, Lifei Chen  
> **Conference**: CVPR 2022

---

## Prerequisites

- Python 3.6
- GPU Memory ≥ 6G
- NumPy
- PyTorch 0.3+ (http://pytorch.org/)
- Torchvision (from source)

You can install all dependencies via:

```bash
pip install -r requirements.txt
```

or create a new Conda environment:

```bash
conda env create -f environment.yml
```

### Datasets

We use **Market1501** and **DukeMTMC-reid** in our experiments. Please download them beforehand.

---

## Getting Started

### Part 1: Training

#### 1.1 Prepare Data Folder

Run `prepare.py` to reorganise the dataset into a format compatible with `torchvision.datasets.ImageFolder`.

1. Edit the **5th line** of `prepare.py` to set your dataset download path, e.g.:
   ```python
   data_path = '/home/Download/Market'   # change to your path
   ```

2. Run:
   ```bash
   python prepare.py
   ```

This creates a `pytorch/` subfolder under the dataset root with the following structure:

```
Market/
├── bounding_box_test/
├── bounding_box_train/
├── gt_bbox/
├── gt_query/
├── query/
├── readme.txt
└── pytorch/
    ├── train_all/          # training images, each ID in its own subfolder (e.g., 0002, 0007, ...)
    ├── val/                # validation (if used)
    ├── query/              # query images
    └── gallery/            # gallery images
```

---

#### 1.2 Training

Train a normally trained model (ResNet‑50 by default):

```bash
python train.py --gpu_ids 0 --name Normally_Trained --data_dir your_data_path --epoch 60
```

**Arguments**:
- `--gpu_ids` : GPU device ID(s)
- `--name`    : name for the saved model
- `--data_dir`: path to the dataset root (e.g., `/path/to/Market/pytorch`)
- `--epoch`   : number of training epochs

Optional:
- `--use_dense` : use DenseNet instead of ResNet‑50

Trained models are saved under `./model/`.

---

##### Train a DL (Defence) Model

Edit line 65 in `train.py` to enable the defence module:

```python
Fuse_LFusePR(G=0.05, G_rgb=0.01, S_rgb=0.01, Aug=0.05, F=0.1)
```

Then run:

```bash
python train.py --gpu_ids 0 --name DL --data_dir your_data_path --epoch 120
```

---

### Part 2: Test

#### 2.1 Extract Features

Load the trained weights and extract visual features for all images:

```bash
python test.py --gpu_ids 0 --name Normally_Trained --test_dir your_data_path
```

- `--name` : directory name of the trained model (under `./model/`)

#### 2.2 Evaluate with Re‑ranking

```bash
python evaluate_gpu.py
```

> **Note**: This step requires >10GB memory. Run it on a powerful machine if possible.  
> **Important**: You must run `test.py` **before** running `evaluate_gpu.py`.

---

### Part 3: White‑Box Attack

Generate adversarial examples (on the query set) using our LTA attack:

```bash
python aa_LTA.py --gpu_ids 0 --name <model_folder_name> --test_dir your_data_path
```

- `--name` : folder name of the model you want to attack (e.g., `DL`)

The generated adversarial images are saved under `./adv_data/`.  
To test the attack effect, replace the original query set with these adversarial examples and re‑run `test.py` and `evaluate_gpu.py`.

---

### Part 4: Joint Adversarial Defence (JAD)

First, train a DL model and perform a white‑box attack on it to generate adversarial examples.

Then, edit lines 78‑80 in `test.py` to apply our passive defence – **Circuitous Scaling**:

```python
######## JAD
transforms.Resize((110,50), interpolation=3),
transforms.Resize((220,100), interpolation=3),
transforms.Resize((110,50), interpolation=3),
```

After enabling this, run `test.py` and `evaluate_gpu.py` to evaluate the defence performance.

---

> **Tip**: For more robust results, train several DL models, test attacks/defences separately, or evaluate cross‑domain performance on other datasets. In cross‑domain settings, DL‑trained models typically outperform normally trained ones in terms of defence.

---

## Troubleshooting & Earlier Code

If you encounter issues with reproducing the adversarial defence, please refer to our earlier open‑source version:  
👉 [https://github.com/finger-monkey/ReID_Adversarial_Defense/](https://github.com/finger-monkey/ReID_Adversarial_Defense/)

---

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{colorAttack2022,
  title={Person re-identification method based on color attack and joint defence},
  author={Gong, Yunpeng and Huang, Liqing and Chen, Lifei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4313--4322},
  year={2022}
}
```

---

## Contact

Email: [fmonkey625@gmail.com](mailto:fmonkey625@gmail.com)
  pages={4313--4322},
  year={2022}
}
```

## Contact Me

Email: fmonkey625@gmail.com

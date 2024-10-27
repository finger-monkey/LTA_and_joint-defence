# This repository contains the reference source code for the CVPR2022 paper titled 'Person Re-identification Method Based on Color Attack and Joint Defence'.


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
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4313--4322},
  year={2022}
}
```

## Contact Me

Email: fmonkey625@gmail.com

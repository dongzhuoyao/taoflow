option1:
conda create -n chaiyujin python=3.6
conda install  pytorch=0.4.1  torchvision=0.2.1  -c pytorch
pip install docopt tqdm scipy matplotlib tensorboardx wandb opencv-python docopt

option2:

conda create -n chaiyujin10 python=3.6
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install docopt tqdm scipy matplotlib tensorboardx wandb opencv-python docopt tabulate termcolor sklearn fbpca

option3(only available for das6):

conda activate stylegan3
pip install torch-fidelity



CUDA_VISIBLE_DEVICES=0 python train.py ./hparams/celeba_das6.json celeba /home/thu/dataset/celeba_torchvision/celeba/

CUDA_VISIBLE_DEVICES=0 python train.py ./hparams/clever_das6.json xxxx /home/thu/dataset/celeba_torchvision/xxxxxx/
CUDA_VISIBLE_DEVICES=0 python train.py ./hparams/celebamask256_ivi.json xxxx /home/thu/dataset/celeba_torchvision/xxxxxx/
python train.py ./hparams/celeba_ivi1_debug.json celeba /home/thu/dataset/celeba_torchvision/celeba/

CUDA_VISIBLE_DEVICES=0,1,2 python train.py ./hparams/celebamask256_das6_3g.json celeba /home/thu/dataset/celeba_hd256/data256x256

CUDA_VISIBLE_DEVICES=0,1,2 python train.py ./hparams/celebamask256_das6_3g_8k.json celeba /home/thu/dataset/celeba_hd256/data256x256



python train.py ./hparams/celeba_ivi4.json celeba /home/thu/dataset/celeba_torchvision/celeba/


python train.py ./hparams/celeba_ivi1_debug.json celeba /home/thu/dataset/celeba_torchvision/celeba/

python train.py ./hparams/celeba_ivi1.json celeba /home/thu/dataset/celeba_torchvision/celeba/

python train.py ./hparams/celeba_das6_k96l1.json celeba /home/thu/dataset/celeba_torchvision/celeba/



python infer_celeba.py ./hparams/celeba_inf.json /home/thu/dataset/celeba_torchvision/celeba/ ./celeba_z

python train.py ./hparams/celeba.json celeba /home/thu/dataset/celeba_torchvision/celeba/img_align_celeba



LU_decompose, n_bit, coupling, fuck_prior, image_size, K-L, 

# Tricks for celebaHD256

- 5bit!!!!
- K=32, L=6, minbs=40, coupling=additive from Glow
- batch number 1000k x bs=40
- das6, batch size=21, warmup_steps changed from 4000 to 64000
get CelebA-HD dataset from: [https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download](https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download)


/home/thu/dataset/celeba_hd256/data256x256

# Celebamask

conda activate stylegan3
makedir ./mask
python preprocess_data_celebamask30k_515.py




# Glow
This is pytorch implementation of paper "Glow: Generative Flow with Invertible 1x1 Convolutions". Most modules are adapted from the offical TensorFlow version [openai/glow](https://github.com/openai/glow).

# TODO
- [x] Glow model. The model is coded as described in original paper, some functions are adapted from offical TF version. Most modules are tested.
- [x] Trainer, builder and hparams loaded from json.
- [x] Infer after training
- [ ] Test LU_decomposed 1x1 conv2d

# Scripts
- Train a model with
    ```
    train.py <hparams> <dataset> <dataset_root>
    ```
- Generate `z_delta` and manipulate attributes with
    ```
    infer_celeba.py <hparams> <dataset_root> <z_dir>
    ```

# Training result
Currently, I trained model for 45,000 batches with `hparams/celeba.json` using CelebA dataset. In short, I trained with follwing parameters

|      HParam      |            Value            |
| ---------------- | --------------------------- |
| image_shape      | (64, 64, 3)                 |
| hidden_channels  | 512                         |
| K                | 32                          |
| L                | 3                           |
| flow_permutation | invertible 1x1 conv         |
| flow_coupling    | affine                      |
| batch_size       | 12 on each GPU, with 4 GPUs |
| learn_top        | false                       |
| y_condition      | false                       |

- Download pre-trained model from [Dropbox](https://www.dropbox.com/s/3wx7vmsurjzfelm/trained.pkg?dl=0)





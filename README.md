# SRFlow
#### Official SRFlow training code: Super-Resolution using Normalizing Flow in PyTorch <br><br>
#### [[Paper] ECCV 2020 Spotlight](https://bit.ly/2DkwQcg)

<br>

**News:** Unified Image Super-Resolution and Rescaling [[code](https://bit.ly/2VOKHBb)]
<br>
<br>

[![SRFlow](https://user-images.githubusercontent.com/11280511/98149322-7ed5c580-1ecd-11eb-8279-f02de9f0df12.gif)](https://bit.ly/3jWFRcr)
<br>
<br>
<br>

# Setup: Data, Environment, PyTorch Demo

<br>

```bash
git clone https://github.com/andreas128/SRFlow.git && cd SRFlow && ./setup.sh
```

<br>

This oneliner will:
- Clone SRFlow
- Setup a python3 virtual env
- Install the packages from `requirements.txt`
- Download the pretrained models
- Download the DIV2K validation data
- Run the Demo Jupyter Notebook

If you want to install it manually, read the `setup.sh` file. (Links to data/models, pip packages)

<br>
<br>

# Demo: Try Normalizing Flow in PyTorch

```bash
./run_jupyter.sh
```

This notebook lets you:
- Load the pretrained models.
- Super-resolve images.
- Measure PSNR/SSIM/LPIPS.
- Infer the Normalizing Flow latent space.

<br><br>

# Testing: Apply the included pretrained models

```bash
source myenv/bin/activate                      # Use the env you created using setup.sh
cd code
CUDA_VISIBLE_DEVICES=-1 python test.py ./confs/SRFlow_DF2K_4X.yml      # Diverse Images 4X (Dataset Included)
CUDA_VISIBLE_DEVICES=-1 python test.py ./confs/SRFlow_DF2K_8X.yml      # Diverse Images 8X (Dataset Included)
CUDA_VISIBLE_DEVICES=-1 python test.py ./confs/SRFlow_CelebA_8X.yml    # Faces 8X
```
For testing, we apply SRFlow to the full images on CPU.

<br><br>

# Training: Reproduce or train on your Data

The following commands train the Super-Resolution network using Normalizing Flow in PyTorch:

```bash
source myenv/bin/activate                      # Use the env you created using setup.sh
cd code
python train.py -opt ./confs/SRFlow_DF2K_4X.yml      # Diverse Images 4X (Dataset Included)
python train.py -opt ./confs/SRFlow_DF2K_8X.yml      # Diverse Images 8X (Dataset Included)
python train.py -opt ./confs/SRFlow_CelebA_8X.yml    # Faces 8X
```

- To reduce the GPU memory, reduce the batch size in the yml file.
- CelebA does not allow us to host the dataset. A script will follow.

### How to prepare CelebA?

**1. Get HD-CelebA-Cropper**

```git clone https://github.com/LynnHo/HD-CelebA-Cropper```

**2. Download the dataset**

`img_celeba.7z` and `annotations.zip` as desribed in the [Readme](https://github.com/LynnHo/HD-CelebA-Cropper).

**3. Run the crop align**

```python3 align.py --img_dir ./data/data --crop_size_h 640 --crop_size_w 640 --order 3 --face_factor 0.6 --n_worker 8```

**4. Downsample for GT**

 Use the [matlablike kernel](https://github.com/fatheral/matlab_imresize) to downscale to 160x160 for the GT images.

**5. Downsample for LR**

Downscale the GT using the Matlab kernel to the LR size (40x40 or 20x20)

**6. Train/Validation**

For training and validation, we use the corresponding sets defined by CelebA (Train: 000001-162770, Validation: 162771-182637)

**7. Pack to pickle for training**

`cd code && python prepare_data.py /path/to/img_dir`

<br><br>

# Dataset: How to train on your own data

The following command creates the pickel files that you can use in the yaml config file:

```bash
cd code
python prepare_data.py /path/to/img_dir
```

The precomputed DF2K dataset gets downloaded using `setup.sh`. You can reproduce it or prepare your own dataset.

<br><br>

# Our paper explains

- **How to train Conditional Normalizing Flow** <br>
  We designed an architecture that archives state-of-the-art super-resolution quality.
- **How to train Normalizing Flow on a single GPU**  <br>
  We based our network on GLOW, which uses up to 40 GPUs to train for image generation. SRFlow only needs a single GPU for training conditional image generation.
- **How to use Normalizing Flow for image manipulation**  <br>
  How to exploit the latent space for Normalizing Flow for controlled image manipulations
- **See many Visual Results**  <br>
  Compare GAN vs Normalizing Flow yourself. We've included a lot of visuals results in our [[Paper]](https://bit.ly/2D9cN0L).

<br><br>

# GAN vs Normalizing Flow - Blog

[![](https://user-images.githubusercontent.com/11280511/98148862-56e66200-1ecd-11eb-817e-87e99dcab6ca.gif)](https://bit.ly/2EdJzhy)

- **Sampling:** SRFlow outputs many different images for a single input.
- **Stable Training:** SRFlow has much fewer hyperparameters than GAN approaches, and we did not encounter training stability issues.
- **Convergence:** While GANs cannot converge, conditional Normalizing Flows converge monotonic and stable.
- **Higher Consistency:** When downsampling the super-resolution, one obtains almost the exact input.

Get a quick introduction to Normalizing Flow in our [[Blog]](https://bit.ly/320bAkH).
<br><br><br>

<br><br>

# Wanna help to improve the code?

If you found a bug or improved the code, please do the following:

- Fork this repo.
- Push the changes to your repo.
- Create a pull request.

<br><br>

# Paper
[[Paper] ECCV 2020 Spotlight](https://bit.ly/2XcmSks)

```bibtex
@inproceedings{lugmayr2020srflow,
  title={SRFlow: Learning the Super-Resolution Space with Normalizing Flow},
  author={Lugmayr, Andreas and Danelljan, Martin and Van Gool, Luc and Timofte, Radu},
  booktitle={ECCV},
  year={2020}
}
```
<br><br>

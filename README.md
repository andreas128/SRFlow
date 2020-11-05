# SRFlow
## Learning the Super-Resolution Space with Normalizing Flow <br> <sub> [[Paper] ECCV 2020 Spotlight](https://bit.ly/2DkwQcg) </sub>
<br><br>

[![SRFlow](https://user-images.githubusercontent.com/11280511/98149322-7ed5c580-1ecd-11eb-8279-f02de9f0df12.gif)](https://bit.ly/3jWFRcr)
<br><br>

## Setup the Environment and start the Demo

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
- Download the validation data
- Run the Demo Jupyter Notebook

If you want to install it manually, read the `setup.sh` file. (Links to data/models, pip packages)

## Start the Demo once everything is setup

```bash
./run_jupyter.sh
```

## Reproduce the SRFlow Results

```
source myenv/bin/activate                      # Use the env you created using setup.sh
cd code
python test.py ./confs/SRFlow_DF2K_4X.yml      # Diverse Images 4X (Dataset Included)
python test.py ./confs/SRFlow_DF2K_8X.yml      # Diverse Images 8X (Dataset Included)
python test.py ./confs/SRFlow_CelebA_8X.yml    # Faces 8X
```
For testing, we apply SRFlow to the full images on CPU.

# Our paper explains

- **How to train Conditional Normalizing Flow** <br>
  We designed an architecture that archives state-of-the-art super-resolution quality.
- **How to train Normalizing Flow on a single GPU**  <br>
  We based our network on GLOW, which uses up to 40 GPUs to train for image generation. SRFlow only needs a single GPU for training conditional image generation.
- **How to use Normalizing Flow for image manipulation**  <br>
  How to exploit the latent space for Normalizing Flow for controlled image manipulations
- **See many Visual Results**  <br>
  Compare GAN vs Normalizing Flow yourself. We've included a lot of visuals results in our [[Paper]](https://bit.ly/2D9cN0L).

# Why I stopped using GAN - Blog

[![](https://user-images.githubusercontent.com/11280511/98148862-56e66200-1ecd-11eb-817e-87e99dcab6ca.gif)](https://bit.ly/2EdJzhy)

- **Sampling:** SRFlow outputs many different images for a single input.
- **Stable Training:** SRFlow has much fewer hyperparameters than GAN approaches, and we did not encounter training stability issues.
- **Convergence:** While GANs cannot converge, conditional Normalizing Flows converge monotonic and stable.
- **Higher Consistency:** When downsampling the super-resolution, one obtains almost the exact input.

Get a quick introduction to Normalizing Flow in our [[Blog]](https://bit.ly/320bAkH).
<br><br><br>

# Code

Due to legal hurdles, we are not yet able to release the training code. Hope we can do so soon!

We use many components of https://github.com/chaiyujin/glow-pytorch and https://github.com/xinntao/BasicSR.
Thanks a lot!

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

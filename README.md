# Summary

- We conducted a series of experiments on the MNIST dataset, including  ddpm and conditional ddpm, as well as cold diffusion and score based model.
- You can run the main.py with command("ddpm/ddpm_conditional/cold_median/cold_kernel/cold_resolution/score/sde") to see our results of experiments.

# DDPM and DDPM Condtional

We use U-NET to generate a new image based on the image after adding noise

# Cold Diffusion

We tried three deterministic operations, adding median blur, mean blur and masking part of the image.

# Score based model

This illustration shows how the reverse process recovers the distribution of the training data.

![](static/reverse.png)

## References

* The dino dataset comes from the [Datasaurus Dozen](https://www.autodesk.com/research/publications/same-stats-different-graphs) data.
* HuggingFace's [diffusers](https://github.com/huggingface/diffusers) library.
* lucidrains' [DDPM implementation in PyTorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
* Jonathan Ho's [implementation of DDPM](https://github.com/hojonathanho/diffusion).
* InFoCusp's [DDPM implementation in tf](https://github.com/InFoCusp/diffusion_models).


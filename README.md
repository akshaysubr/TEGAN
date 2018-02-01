## Generative Adversarial Networks for Turbulence Enrichment
Project category: GANs

### What and why
Turbulent flow is important in many engineering applications. However, simulating turbulence is very computationally expensive due to extremely high resolution requirements. Large Eddy Simulations (LES) that simulate only the large scales have become popular due much lower cost, but require explicit modeling of the small scales. Here, we propose to enrich LES data by populating it with small scales obtained using a Generative Adversarial Network (GAN).

Modeling turbulence accurately is extremely challenging especially in capturing high order statistics due to its intermittent nature. GANs have been shown to perform better than other data driven approaches like PCA in capturing high order moments [[1]](https://arxiv.org/pdf/1708.01810.pdf). In addition, generating physically realistic realizations are important for physical simulation data; a constraint not present when using generative models for images. Incorporating this constraint into the GAN framework would be crucial to its performance.

### Data
We plan on generating our own data set using an in-house simulation code (PadeOps). This requires running the code on a compute cluster using 256 processors for multiple days. We have access to compute clusters that would allow us to generate the required data.

### Methodology


### References

### Evaluation

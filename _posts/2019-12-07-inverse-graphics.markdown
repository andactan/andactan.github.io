---
layout: post
title:  "3D-Aware Scene Manipulation via Inverse Graphics"
date:   2019-12-07 16:18:21 +0100
categories: jekyll update
---
## Abstract

As humans, we renown at perceiving the world around us and simultaneously make deductions out of it. But, what is interesting and also magnificient part about it is, as we do perceive the world around, we are also able to simulate and imagine in what ways changes can appear unto our perceptions for future scenarios. For instance, we can, without showing much effort, 
detect and recognize cars on a road, and infer their attributes. At the same time, we have also a special power by which we could imagine how cars may move on the road, or they will ever rotate right or left, or even think about what might
be a more plausible color a car may have other than the driver's choice. 

Motivation arises from this very specific human ability. Can we enable a neural network architecture gain such ability?

## Introduction

To grant that ability to the machines, we should let them understand given scenes and encode given scenes into latent space representations. From those intermediate representations, they should also be able to generate plausible images back. And actually, deep generative models and specifically **Generative Adversarial Networks (GANs)** [[Goodfellow et al., 2014]](https://arxiv.org/abs/1406.2661){:target="_blank"} excels at this duty with their terrific and simplistic encoder-decoder architecture. In a GAN setup, two differentiable functions, represented by neural networks, are locked in a game. The two players (the generator and the discriminator) have different roles in this framework. The generator tries to produce data that come from some probability distribution (namely the latent-space representation) and the discriminator acts like a judge. It gets to decide if the input comes from the generator or from the true training set. However, deep generative models have their flaws in which latent-space representations are often limited to a single object, not easy to interpret, and missing the 3D structure behind 2D projection. As aresult, deep generative models are not perfectly fit for scene manipulation tasks such as moving objects around. Additionally, as it is a scene manipulation task at hand, it is needed to have human-interpretable and intuitive representations so that any human user or a graphics engine is enabled to use.

{% include image.html url="/assets/figures/gans.png" description="Generative Adversarial Network Structure" %}

In this paper, motivated by the aforementioned human abilities, authors propose an expressive, disentangled and interpretable scene representation method for machines. The proposed architecture elaborates an encoder-decoder structure for the main purpose and divides different tasks to three separate branches: one for **semantics**, one for **geometry** and one for **texture**. This separative and human-interpretable approach also overcomes the mentioned flaws of the deep generative models. By adapting this architecture, it is further possible to easily manipulate given images.

### Datasets

The proposed method has been tested and validated upon two different datasets: **Virtual KITTI** [[Gaidon et al., 2016]](https://arxiv.org/abs/1605.06457){:target="_blank"} and **Cityscapes** [[Cordts et al., 2016]](https://arxiv.org/abs/1604.01685){:target="blank"}. Both quantitative and qualitative experiments are demonstrated over those two datasets. Additionally, authors create an image editing benchmark on **Virtual KITTI** to elaborate the effectiveness of the proposed method and also to compare editing power against the 2D baseline models.

## Related Work

Inspiration for the proposed method comes from three different sets of state-of-art work. Authors cancel out the flaws of prior works by combining the best features of all mentioned methods below:

### Interpretable Image Representation

The main idea of the proposed method comes from obtaining interpretable visual representations with neural networks.

#### Deep Convolutional Inverse Graphics Network [[Kulkarni et al., 2015]](https://arxiv.org/abs/1503.03167)
In this method, to obtain representations from the given image authors freezes a subset of latent space elements while feeding images that move along a specific direction on the image manifold. While this is one of the first succesfull attempts of scene understanding and representation creation, the proposed method focuses on only a single object.

#### Neural Scene De-rendering [[Wu et al., 2017a]](http://nsd.csail.mit.edu/papers/nsd_cvpr.pdf)
This is the method which the proposed disentangled and differentiable method most resembles. In the paper, authors likewise propose an encoder-decoder architecture that uses a neural network for the encoder part and a graphics engine for the decoder. However, since backpropagating gradients from the graphics engine is not possible (because of discrete representations of features), it is not generalizable to a new environment.


### Deep Generative Models
Deep generative models have been used extensively for image synthesis nad learning internal representations. However, representations learned by these methods are not likely to be interpreted easily and often ignore the 3D characteristics of our world.

#### Synthesizing 3D Shapes via Modeling Multi-view Depth Maps and Silhouttes with Deep Generative Models [[Soltani et al., 2017]](#home)
--> it does something doing something on something by something


### Deep Image Manipulation
Learning-based methods have enabled their users in various tasks, such as image-to-image translation [[Isola et al., 2017](https://arxiv.org/abs/1611.07004), [Zhu et al., 2017a](https://arxiv.org/abs/1703.10593), [Liu et al., 2017](https://arxiv.org/abs/1703.00848)], style transfer [[Gatys et al., 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)], automatic colorization [[Zhang et al., 2016](https://arxiv.org/abs/1603.08511)], inpainting [[Pathak et al., 2016](https://arxiv.org/abs/1604.07379)], attribute editing [[Yan et al., 2016a](https://arxiv.org/abs/1512.00570)], interactive editing [[Zhu et al., 2016](https://arxiv.org/abs/1609.03552)], and denoising [[Gharbi et al., 2016](https://groups.csail.mit.edu/graphics/demosaicnet/data/demosaic.pdf)]. But the proposed method differs from those methods at, while prior work focuses on 2D setting, proposed architecture enables 3D-aware image manipulation. In addition to that, often those methods require a structured representation, proposed method can learn internal representations by itself.

## Method
In the paper, authors propose **3D scene de-rendering network (3D-SDN)** in an encoder-decoder framework. As it is said in the introduction chapter, network first de-renders (encodes) an image into disentangled representation for three information sets: **semantic**, **geometric** and **textural**. Then, using those representations, it tries to render (recontrucr, decode) the image into a plausible and similar copy.

{% include image.html url="/assets/figures/network_architecture.png" description="Overview of the General Architecture" %}

### Branches and Pipeline
Mentioned pipeline branches first decouple into two parts: **de-renderer** and **renderer**. While semantic branch doesn't have any rendering part, other two parts of the pipeline (namely, geometric and textural) first de-render the image into various representations and for last textural branch combines different outputs from different branches (including its outputs) and tries to generate a plausible image back. Let us see what each parts do individually.

#### Semantic Branch

Semantic branch does not employ any rendering process, at all. One and only duty of the semantic branch is learning to produces semantic segmentation of the given image to infer what parts of the image are foreground objects (cars and vans) and what parts are background (trees, sky, road). For the purpose, it adopts Dilated Residual Network (DRN) [[Yu et al., 2017]](https://arxiv.org/abs/1705.09914).
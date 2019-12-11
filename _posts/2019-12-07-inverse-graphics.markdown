---
layout: post
title:  "3D-Aware Scene Manipulation via Inverse Graphics"
date:   2019-12-07 16:18:21 +0100
categories: jekyll update
---
## **Abstract**

As humans, we renown at perceiving the world around us and simultaneously make deductions out of it. But, what is interesting and also magnificient part about it is, as we do perceive the world around, we are also able to simulate and imagine in what ways changes can appear unto our perceptions for future scenarios. For instance, we can, without showing much effort, 
detect and recognize cars on a road, and infer their attributes. At the same time, we have also a special power by which we could imagine how cars may move on the road, or they will ever rotate right or left, or even think about what might
be a more plausible color a car may have other than the driver's choice. 

Motivation arises from this very specific human ability. Can we enable a neural network architecture gain such ability?

## **Introduction**

To grant that ability to the machines, we should let them understand given scenes and encode given scenes into latent space representations. From those intermediate representations, they should also be able to generate plausible images back. And actually, deep generative models and specifically **Generative Adversarial Networks (GANs)** [[Goodfellow et al., 2014]](https://arxiv.org/abs/1406.2661){:target="_blank"} excels at this duty with their terrific and simplistic encoder-decoder architecture. In a GAN setup, two differentiable functions, represented by neural networks, are locked in a game. The two players (the generator and the discriminator) have different roles in this framework. The generator tries to produce data that come from some probability distribution (namely the latent-space representation) and the discriminator acts like a judge. It gets to decide if the input comes from the generator or from the true training set. However, deep generative models have their flaws in which latent-space representations are often limited to a single object, not easy to interpret, and missing the 3D structure behind 2D projection. As aresult, deep generative models are not perfectly fit for scene manipulation tasks such as moving objects around. Additionally, as it is a scene manipulation task at hand, it is needed to have human-interpretable and intuitive representations so that any human user or a graphics engine is enabled to use.

{% include image.html url="/assets/figures/gans.png" description="Generative Adversarial Network Structure" %}

In this paper, motivated by the aforementioned human abilities, authors propose an expressive, disentangled and interpretable scene representation method for machines. The proposed architecture elaborates an encoder-decoder structure for the main purpose and divides different tasks to three separate branches: one for **semantics**, one for **geometry** and one for **texture**. This separative and human-interpretable approach also overcomes the mentioned flaws of the deep generative models. By adapting this architecture, it is further possible to easily manipulate given images.

{% include image.html url="/assets/figures/proposal.png" description="Proposed Method" %}

### **Datasets**

The proposed method has been tested and validated upon two different datasets: **Virtual KITTI** [[Gaidon et al., 2016]](https://arxiv.org/abs/1605.06457){:target="_blank"} and **Cityscapes** [[Cordts et al., 2016]](https://arxiv.org/abs/1604.01685){:target="blank"}. Both quantitative and qualitative experiments are demonstrated over those two datasets. Additionally, authors create an image editing benchmark on **Virtual KITTI** to elaborate the effectiveness of the proposed method and also to compare editing power against the 2D baseline models.

## **Related Work**

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

## **Method**
In the paper, authors propose **3D scene de-rendering network (3D-SDN)** in an encoder-decoder framework. As it is said in the introduction chapter, network first de-renders (encodes) an image into disentangled representation for three information sets: **semantic**, **geometric** and **textural**. Then, using those representations, it tries to render (recontrucr, decode) the image into a plausible and similar copy.

{% include image.html url="/assets/figures/network_architecture.png" description="Overview of the General Architecture" %}

### **Branches and Pipeline**
Mentioned pipeline branches first decouple into two parts: **de-renderer** and **renderer**. While semantic branch doesn't have any rendering part, other two parts of the pipeline (namely, geometric and textural) first de-render the image into various representations and for last textural branch combines different outputs from different branches (including its outputs) and tries to generate a plausible image back. Let us see what each parts do individually.

#### **Semantic Branch**

Semantic branch does not employ any rendering process, at all. One and only duty of the semantic branch is learning to produces semantic segmentation of the given image to infer what parts of the image are foreground objects (cars and vans) and what parts are background (trees, sky, road). For the purpose, it adopts Dilated Residual Network (DRN) [[Yu et al., 2017]](https://arxiv.org/abs/1705.09914).


#### **Geometric Branch**

Before anything feed into the inference module of the geometric branch, object instances are segmented by using Mask R-CNN [[He et al., 2017]](#home).Mask R-CNN generates an image patch and bounding box for each detected object in given image. And afterwards, geometric branch tries to infer 3D mesh model and attributes of each object segmented from the masked image patch and bounding box. So how does it do that?

{% include image.html url="/assets/figures/geo_branch.png" description="Geometric Branch Pipeline" %}

An object present in the given image are described via its 3D mesh model $$ \textbf{M} $$, scale $$ \textbf{s} \in R^{3}$$, rotation quaternion $$ \textbf{q} \in R^{4} $$ and translation from the camera center $$ \textbf{t} \in R^{3} $$. For the simplicity, authors assume that for the most real-world problems, objects are assumed lie on the ground and therefore have only one rotational degree of freedom. Therefore, they do not employ the full quaternion and convert it to a one-value vector, $$ \textbf{q} \in R $$. After defining the attributes of geometric branch, let us describe how they formulate the training objective for the network.


##### **3D Attribute Prediction Loss**

The de-renderer part directly predicts the values of $$ \textbf{s} $$ and $$ \textbf{q} $$. But, for the translation inference, it is not straightforward compared to others. Instead, they separate the translation vector $$ \textbf{t} $$ into two parts as one being the object's distance to camera plane, $$ t $$ and the other one is the 2D image coordinates of the object's center in 3D world, $$ [x_{3D}, y_{3D}] $$. Combining those two and given intrinsic matrix of the camera, one can implement the $$ \textbf{t} $$ of the mentioned object.

To predict $$ t $$, first they parametrize t in the log-space [[Eigen et al., 2014]](#home) and reparametrize it to a normalized distance $$ \tau = t\sqrt{wh} $$ where $$ [w, h] $$ is the width and height of the bounding box, respectively. Additionally, to infer $$ [x_{3D}, y_{3D}] $$, following a prior work [[Ren et al., 2015]](#home), they predict the offset distance from the estimated bounding box center $$ [x_{2D}, y_{2D}] $$, and thus the estimated offset becomes $$ e = [(x_{3D}-x_{2D})/w, (y_{3D}-y_{2D})/h] $$.

Combining all those aforementioned attributes in a loss function yields one of the training objectives of geometric de-renderer part as 

$$
    \mathcal{L}_{pred} = 
    \big \| \log{\tilde{\textbf{s}}} - \log{\textbf{s}} \big \|_{2}^{2} +
    \big (1 - (\tilde{\textbf{q}} - \textbf{q})^2 \big) +
    \big \| \tilde{\textbf{e}} - \textbf{e} \big \|^{2}_{2} + 
    (\log{\tilde{\tau}} - \log{\tau})^{2}
$$

where $$ \tilde{.} $$ denotes the respective predicted attribute.

##### **Reprojection Consistency Loss**

Another training objective of the geometric de-renderer part is ensuring the 2D rendering of the predicted 3D mesh model $$ \textbf{M} $$ matches its silhoutte $$ \textbf{S} $$ so that 3D mesh model with the Free-Form Deformation (FFD) [[Sederberg and Party, 1986]](#home) parameters best fits the detected object. This is the **reprojection loss**. It should also be noted that  since they have no ground truth value of the objects in images, reprojection loss is the only training signal for mesh selection and silhoutte calibration (namely finding the deformation parameters).

To render the 2D silhoutte of 3D mesh model, they use a differential renderer [[Kato et al., 2018]](#home), according to the FFD parameters $$ \phi $$ and predicted 3D attributes of the given image $$ \tilde{\pi} = \{\tilde{\textbf{s}}, \tilde{\textbf{q}}, \tilde{\textbf{t}}\} $$. By convention, the 2D silhoutte of the given 3D mesh model $$ M $$ is a function of $$\phi$$ and $$  \tilde{\pi} $$: $$ \tilde{\textbf{S}} = RenderSilhoutte(\text{FFD}_{\phi}(\textbf{M}), \tilde{\pi}) $$.

So, another training objective comes into the action:

$$
    \mathcal{L}_{reproj} = \big \| \tilde{\textbf{S}} - \textbf{S} \big \|
$$

{% include image.html url="/assets/figures/reproj.png" description="Object silhouttes rendered with and without reprojection loss" %}

All above, we defined how they achieve a training objective for consistency when they have tried to find the suitable mesh model and deformation parameters. This procedure can directly infer the deformation parameters, but mesh model selection is non-differentiable and just doing backpropagation will not take it anywhere, at all. Therefore, they reformulate the whole problem as reinforcement learning problem. They adopt a multi-sample REINFORCE algorithm [[Williams, 1992]](#home) to choose a suitable mesh from a set of eight candidate meshes using the negative $$  \mathcal{L}_{reproj} $$ as the reward.

{% include image.html url="/assets/figures/reinforce.png" description="Using REINFORCE and allowing FFD enables precise reconstruction" %}

After de-rendering process, geometric de-renderer combines all information gathered from different process lines and outputs an instance map, an object map and a pose map for each given image.

#### **Textural Branch**

Now we have got the 3D geometric attributes of the objects and also the semantic segmentation of the given scene, we can infer textural features of each detected object.

First of all, textural branch combines the semantic map from the semantic semantic branch and the instance map from the geometric branch to generate an instance-wise semantic label map $$ \textbf{L} $$. Resultant label map encodes which pixels in the image are the objects pixels and whether instance class of the each object pixel belongs to a foreground object or a background object. And also, during combination any conflict is resolved in favor of the instance map [[Kirillov et al., 2018]](#home). Using an extended version of the models used in multimodal image-to-image translation [[Zhu et al., 2017b](#home), [Wang et al., 2018](#home)], a **textural de-renderer** encodes the texture information into a low-dimensional embedding and later a **textural renderer** tries to reconstruct the original image from that representation. Let us break into pieces how the whole branch de-renders and renders given image:

Given an image $$ \textbf{I} $$ and its instance-wise label map $$ \textbf{L} $$, the objective is to obtain a low-dimensional embedding $$ \textbf{z} $$ such that from $$ (\textsf{L}, \textbf{z}) $$, it is possible the reconstruct a plausible copy of the original image. The whole idea is formulated as a conditional adversarial learning framework with three networks $$ (G, D, E) $$:

1. Textural de-renderer $$ E: (\textbf{L}, \textbf{I}) \xrightarrow{} \textbf{z} $$
2. Textural renderer $$ G: (\textbf{L}, \textbf{z}) \xrightarrow{} \tilde{\textbf{I}} $$
3. Discriminator $$ D: (\textbf{I}, \tilde{\textbf{I}}) \xrightarrow{} [0, 1] $$

where $$ \tilde{\textbf{I}} $$ is the reconstructed image.

##### **Photorealism Loss**
To increase the photorealism of the reconstructed images, authors employ the standard GAN loss as

$$
    \mathcal{L}_{GAN}(\textit{G}, \textit{D}, \textit{E}) = \mathbb{E}_{\textbf{L}, \textbf{I}} \big [
    \log (\textit{D}(\textbf{L}, \textbf{I})) + \log (1 - \textit{D}(\textbf{L}, \tilde{\textbf{I}})) \big ]
$$

##### **Pixel-wise Reconstruction Loss**

Also, to cancel out the pixel-wise differences between the original image and the reconstructed image, they use pixel-wise reconstruction loss.

$$
    \mathcal{L}_{Recon}(\textit{G}, \textit{E}) = \mathbb{E}_{\textbf{L}, \textbf{I}} \big [
    \big \| \textbf{I} - \tilde{\textbf{I}} \big \|_{1} ]
$$

##### **Stabilizing GAN Training**
GAN models can suffer severely from the non-convergence problem. The generator tries to find the best image to fool the discriminator, while discriminator tries to counterattack this proposal by labeling it as not-a-plausible reconstruction. The "best" image keeps changing while both networks are counteracting each other. However, this might turn out to be a never-ending cat-and-mouse game and model unfortunately never converges to a steady state. To overcome this and stabilize the training, authors follow the prior work in [[Wang et al., 2018]](#home) and use both discriminator matching loss [[Wang et al., 2018](#home), [Larsen et al., 2018]](#home) and perceptual loss [[Dosovitsky and Brox, 2016](#home), [Johnson et al., 2016](#home)] and both of which goal to minimize the statistical difference between the feature vectors of real image and the reconstructed image. For the perceptual loss, feature vectors are generated from the intermediate layers of the VGG network [[Simonyan and Zesserman, 2015](#home)] and for the discriminator feature matching loss, as the name suggests, they are generated using the layers of discriminator network. And the overall objective becomes as

$$
    \mathcal{L}_{FM}(\textit{G}, \textit{D}, \textit{E}) = \mathbb{E}_{\textbf{L}, \textbf{I}} \Big [
    \sum_{i=1}^{T_{F}}\frac{1}{N_{i}} \big \| \textit{F}^{(i)}(\textbf{I}) - \textit{F}^{(i)}(\tilde{\textbf{I}}) \big \|_{1} + 
    \sum_{i=1}^{T_{D}}\frac{1}{M_{i}} \big \| \textit{D}^{(i)}(\textbf{I}) - \textit{D}^{(i)}(\tilde{\textbf{I}}) \big \|_{1}
    \Big ]
$$

where $$ \textit{F}^{(i)} $$ denotes the $$ i $$-th layer of a pre-trained VGG network with $$ N_{i} $$ elements. Likewise, $$  \textit{D}^{(i)} $$ denotes the $$ i $$-th layer of the discriminator network with $$ M_{i} $$ elements.

These all objectives above, are combined into a minimax game between $$ D $$ and $$ (G, E) $$:

$$
    G^{*}, E^{*} = \arg \min_{G, E} \Bigg ( 
    \max_{D} (\mathcal{L}_{GAN}(G, D, E)) + 
    \lambda_{FM}\mathcal{L}_{FM}(G, D, E) + 
    \lambda_{Recon}\mathcal{L}_{Recon}(G, E)
    \Bigg )
$$

where $$ \lambda_{FM} $$ and $$ \lambda_{Recon} $$ are the relative importance of each objective respectively.
---
layout: post
title:  "3D-Aware Scene Manipulation via Inverse Graphics"
date:   2019-12-20 16:18:21 +0100
categories: jekyll update
permalink: /inverse-graphics/
---
## **Abstract**

As humans, we are renown at perceiving the world around us and simultaneously make deductions out of it. But, what is interesting and also magnificient is that as we do perceive the world around, we are also able to simulate and imagine in what ways changes can appear unto our perceptions for future scenarios. For instance, we can, without showing much effort, detect and recognize cars on a road, and infer their attributes. At the same time, we have also a special power by which we could imagine how cars may move on the road, or they will ever rotate right or left, or we can even wonder about what might
be a more plausible color the car may have other than the driver's choice. 

Motivation arises from this very specific human ability. Can we enable a neural network architecture gain such ability?

## **Introduction**

To grant that ability to the machines, we should let them understand given scenes and encode given scenes into latent space representations. From those intermediate representations, they should also be able to generate plausible images back. And actually, deep generative models and specifically **Generative Adversarial Networks (GANs)** [[Goodfellow et al., 2014]](https://arxiv.org/abs/1406.2661){:target="_blank"} excels at this duty with their terrific and simplistic encoder-decoder architecture. In a GAN setup, two differentiable functions, represented by neural networks, are locked in a game. The two players (the generator and the discriminator) have different roles in this framework. The generator tries to produce data that come from some probability distribution (namely the latent-space representation) and the discriminator acts like a judge. It gets to decide if the input comes from the generator or from the true training set. However, deep generative models have their flaws in which latent-space representations are often limited to a single object, not easy to interpret, and missing the 3D structure behind 2D projection. As aresult, deep generative models are not perfectly fit for scene manipulation tasks such as moving objects around. Additionally, as it is a scene manipulation task at hand, it is needed to have human-interpretable and intuitive representations so that any human user or a graphics engine is enabled to use.

{% include image.html url="/assets/figures/gans.png" description="Generative Adversarial Network Structure" %}

In this paper, motivated by the aforementioned human abilities, authors propose an expressive, disentangled and interpretable scene representation method for machines. The proposed architecture elaborates an encoder-decoder structure for the main purpose and divides different tasks to three separate branches: one for **semantics**, one for **geometry** and one for **texture**. This separative and human-interpretable approach also overcomes the mentioned flaws of the deep generative models. By adapting this architecture, it is further possible to easily manipulate given images.

{% include image.html url="/assets/figures/proposal.png" description="Proposed Method" %}

## **Related Work**

As said before, thw whole proposed method is consisted of three different pipelines: first an interpretable image representation is obtained for the given image, second another pipeline takes those interpretable representations of the given image and tries to synthesize an realistic image back, third but not least enables its user to edit images in anyway imagined since disentangled and human-interpretable representations is at hand. Therefore, inspiration for the proposed method comes from three different sets of state-of-art work:
1.  Interpretable image representation
2.  Deep generative models
3.  Deep image manipulation

Authors cancel out the flaws of prior works by combining the best features of all mentioned methods below.

### Interpretable Image Representation
The main idea behind inverse graphics and obtaining image representations is to reverse-engineer an given image such that end-product can be used to understand the physical world that produced the image or edit the image in various way since the world behind is viable to such manipulations after getting to know it.

This work is inspired by prior work on obtaining interpretable image representations by [[Kulkarni et al., 2015]()] and [[Chen et al., 2016]()]. [[Kulkarni et al., 2015]()] proposes and deep neural network which is composed of several convolution and de-convolution layers. It is called **Deep Convolution Inverse Graphics Network (DC-IGN)** and the model learns interpretable and disentangled representations for pose transformations and lighting conditions. Given a single image, the proposed model can produce different images of the same object with different poses and lighting variations.

Another line of inspiration comes from the **InfoGAN**[[Chen et al., 2016](https://arxiv.org/abs/1606.03657)]. InfoGAN proposes to feed traditional, vanilla GAN [[Goodfellow et al., 2014]()] with two disentangled vectors, namely 'noise vector' and 'latent code', rather than just simply submitting an single noise vector. Thus, as having two different 'turn knobs' for the images, it is easy to obtain disentangled and easy-to-manipulate representations.

But the problem with those two powerful implementations is that they are limited to single objects, whereas overall aim of the proposed method is to have an scene understanding model which captures the whole complexity in images with multiple objects. Therefore, this paper most resembles the work by [[Wu et al., 2017](http://nsd.csail.mit.edu/papers/nsd_cvpr.pdf)] in which it is proposed to 'de-render' the given image with an encoder-decoder framework. While encoder part uses a classical neural network structure, the decoder part is simply a 'graphics engine' for which encoder produces human-readable and interpretable representations. Although graphics engines in their nature require structured and interpretable representations, it is not possible to back-propagate gradients end-to-end because of their discrete and non-differentiable characteristics. Rather, they use an black-box optimization via REINFORCE algorithm [[reinforce et al.,]()]. Compared to this work, the proposed method in this paper uses differentiable models for both encoder and decoder parts.

### Deep Generative Models
Deep generative model is a powerful way to learning rich, internal representations in a unsupervised manner and using those representations to synthesize realistic images back. But, those rich and internal latent representations are hard to interpret for humans and more of than not ignore 3D characteristics of our 3D world. Many work have investigated ways of 3D reconstruction from a single color image[[Choy et al., 2016](https://arxiv.org/abs/1604.00449), [Kar et al., 2015](https://arxiv.org/abs/1411.6069), [Tatarchenko et al., 2016](https://arxiv.org/abs/1511.06702), [Tulsiani et al., 2017](https://arxiv.org/abs/1704.06254), [Wu et al., 2017b](https://arxiv.org/abs/1711.03129)], depth map or silhoutte[[Soltani et al., 2017](https://www.jiajunwu.com/papers/mv3d_cvpr.pdf)]. This works is built upon those proposals and extended them. It does not only reconstructs the image from internal representations via 2D differentiable renderer, but also does it provide an 3D-aware scene manipulation option. 


### Deep Image Manipulation
Learning-based methods have enabled their users in various tasks, such as image-to-image translation [[Isola et al., 2017](https://arxiv.org/abs/1611.07004), [Zhu et al., 2017a](https://arxiv.org/abs/1703.10593), [Liu et al., 2017](https://arxiv.org/abs/1703.00848)], style transfer [[Gatys et al., 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)], automatic colorization [[Zhang et al., 2016](https://arxiv.org/abs/1603.08511)], inpainting [[Pathak et al., 2016](https://arxiv.org/abs/1604.07379)], attribute editing [[Yan et al., 2016a](https://arxiv.org/abs/1512.00570)], interactive editing [[Zhu et al., 2016](https://arxiv.org/abs/1609.03552)], and denoising [[Gharbi et al., 2016](https://groups.csail.mit.edu/graphics/demosaicnet/data/demosaic.pdf)]. But the proposed method differs from those methods. While prior work focuses on 2D setting, proposed architecture enables 3D-aware image manipulation. In addition to that, often those methods require a structured representation, proposed method can learn internal representations by itself.

## **Method**
In the paper, authors propose a **3D scene de-rendering network (3D-SDN)** in an encoder-decoder framework. As it is said in the introduction chapter, network first de-renders (encodes) an image into disentangled representations for three information sets: **semantic**, **geometric** and **textural**. Then, using those representations, it tries to render (reconstruct, decode) the image into a plausible and similar copy.

{% include image.html url="/assets/figures/network_architecture.png" description="Overview of the General Architecture" %}

### **Branches and Pipeline**
Mentioned pipeline branches are decoupled into two parts: **de-renderer** and **renderer**. While semantic branch doesn't have any rendering part, other two parts of the pipeline (namely, geometric and textural) first de-render the image into various representations and for last textural branch combines different outputs from different branches (including its outputs) and tries to generate a plausible image back. Let us see what each parts do individually.

#### **Semantic Branch**

Semantic branch does not employ any rendering process, at all. One and only duty of the semantic branch is learning to produce semantic segmentation of the given image to infer what parts of the image are foreground objects (cars and vans) and what parts are background (trees, sky, road). For the purpose, it adopts Dilated Residual Network (DRN) [[Yu et al., 2017]](https://arxiv.org/abs/1705.09914).


#### **Geometric Branch**

Before anything fed into the inference module of the geometric branch, object instances are segmented by using Mask R-CNN [[He et al., 2017]](https://arxiv.org/pdf/1703.06870.pdf). Mask R-CNN generates an image patch and bounding box for each detected object in given image. And afterwards, geometric branch tries to infer 3D mesh model and attributes of each object segmented, from the masked image patch and bounding box. So how does it do that?

{% include image.html url="/assets/figures/geo_branch.png" description="Geometric Branch Pipeline" %}

An object present in the given image is described via its 3D mesh model $$ \textbf{M} $$, scale $$ \textbf{s} \in R^{3}$$, rotation quaternion $$ \textbf{q} \in R^{4} $$ and translation from the camera center $$ \textbf{t} \in R^{3} $$. For the simplicity, it is assumed that objects lie on the ground and therefore have only one rotational degree of freedom for the most of the real-world problems. Therefore, they do not employ the full quaternion and convert it to a one-value vector, $$ \textbf{q} \in R $$. After defining the attributes of geometric branch, let us describe how they formulate the training objective for the network.


##### **3D Attribute Prediction Loss**

The de-renderer part directly predicts the values of $$ \textbf{s} $$ and $$ \textbf{q} $$. But, for the translation inference, it is not straightforward compared to others. Instead, they separate the translation vector $$ \textbf{t} $$ into two parts as one being the object's distance to camera plane, $$ t $$ and the other one is the 2D image coordinates of the object's center in 3D world, $$ [x_{3D}, y_{3D}] $$. Combining those two and given intrinsic matrix of the camera, one can implement the $$ \textbf{t} $$ of the mentioned object.

To predict $$ t $$, first they parametrize $$ t $$ in the **log-space** [[Eigen et al., 2014]](https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf) and reparametrize it to a normalized distance $$ \tau = t\sqrt{wh} $$ where $$ [w, h] $$ is the width and height of the bounding box, respectively. Additionally, to infer $$ [x_{3D}, y_{3D}] $$, following the prior work in **Faster R-CNN** [[Ren et al., 2015]]([#home](https://arxiv.org/pdf/1506.01497.pdf)), they predict the offset distance from the estimated bounding box center $$ [x_{2D}, y_{2D}] $$, and thus the estimated offset becomes $$ e = [(x_{3D}-x_{2D})/w, (y_{3D}-y_{2D})/h] $$.

Combining all those aforementioned attributes in a loss function yields one of the training objectives of geometric de-renderer part as,

$$
    \mathcal{L}_{pred} = 
    \big \| \log{\tilde{\textbf{s}}} - \log{\textbf{s}} \big \|_{2}^{2} +
    \big (1 - (\tilde{\textbf{q}} - \textbf{q})^2 \big) +
    \big \| \tilde{\textbf{e}} - \textbf{e} \big \|^{2}_{2} + 
    (\log{\tilde{\tau}} - \log{\tau})^{2}
$$

where $$ \tilde{.} $$ denotes the respective predicted attribute.

##### **Reprojection Consistency Loss**

Another training objective of the geometric de-renderer part is ensuring the 2D rendering of the predicted 3D mesh model $$ \textbf{M} $$ matches its silhoutte $$ \textbf{S} $$ so that 3D mesh model with the Free-Form Deformation (FFD) [[Sederberg and Party, 1986]](http://faculty.cs.tamu.edu/schaefer/teaching/689_Fall2006/p151-sederberg.pdf) parameters best fit the detected object's 2D silhoutte. This is the **reprojection loss**. It should also be noted that since they have no ground truth 3D shapes of the objects in images, reprojection loss is the only training signal for mesh selection and silhoutte calibration (namely finding the deformation parameters).

To render the 2D silhoutte of 3D mesh model, they use a differential renderer [[Kato et al., 2018]](https://arxiv.org/abs/1711.07566), according to the FFD parameters $$ \phi $$ and predicted 3D attributes of the given image $$ \tilde{\pi} = \{\tilde{\textbf{s}}, \tilde{\textbf{q}}, \tilde{\textbf{t}}\} $$. By convention, the 2D silhoutte of the given 3D mesh model $$ M $$ is a function of $$\phi$$ and $$  \tilde{\pi} $$: $$ \tilde{\textbf{S}} = RenderSilhoutte(\text{FFD}_{\phi}(\textbf{M}), \tilde{\pi}) $$.

So, another training objective comes into the action:

$$
    \mathcal{L}_{reproj} = \big \| \tilde{\textbf{S}} - \textbf{S} \big \|
$$

{% include image.html url="/assets/figures/reproj.png" description="Object silhouttes rendered with and without reprojection loss" %}

All above, we defined how they achieve a training objective for consistency when they have tried to find the suitable mesh model and deformation parameters. This procedure can directly infer the deformation parameters, but mesh model selection is non-differentiable and just doing backpropagation will not take it anywhere, at all. Therefore, they reformulate the whole problem as reinforcement learning problem. They adopt a multi-sample REINFORCE algorithm [[Williams, 1992]](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf) to choose a suitable mesh from a set of eight candidate meshes using the negative $$  \mathcal{L}_{reproj} $$ as the reward.

{% include image.html url="/assets/figures/reinforce.png" description="Using REINFORCE and allowing FFD enables precise reconstruction" %}

After de-rendering process, geometric de-renderer combines all information gathered from different process lines and outputs an instance map, an object map and a pose map for each given image.

#### **Textural Branch**

Now we have got the 3D geometric attributes of the objects and also the semantic segmentation of the given scene, we can infer textural features of each detected object.

First of all, textural branch combines the semantic map from the semantic semantic branch and the instance map from the geometric branch to generate an instance-wise semantic label map $$ \textbf{L} $$. Resultant label map encodes which pixels in the image are the objects pixels and whether instance class of the each object pixel belongs to a foreground object or a background object. And also, during combination, any conflict is resolved in favor of the instance map [[Kirillov et al., 2018]](https://arxiv.org/abs/1801.00868). Using an extended version of the models used in multimodal image-to-image translation [[Zhu et al., 2017b](https://arxiv.org/abs/1711.11586), [Wang et al., 2018](https://arxiv.org/abs/1711.11585)], a **textural de-renderer** encodes the texture information into a low-dimensional embedding and later a **textural renderer** tries to reconstruct the original image from that representation. Let us break into pieces how the whole branch de-renders and renders given image:

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
GAN models can suffer severely from the non-convergence problem. The generator tries to find the best image to fool the discriminator, while discriminator tries to counterattack this proposal by labeling it as not-a-plausible reconstruction. The "best" image keeps changing while both networks are counteracting each other. However, this might turn out to be a never-ending cat-and-mouse game and model unfortunately never converges to a steady state. To overcome this and stabilize the training, authors follow the prior work in [[Wang et al., 2018](https://arxiv.org/abs/1711.11585)] and use both discriminator matching loss [[Wang et al., 2018](https://arxiv.org/abs/1711.11585), [Larsen et al., 2018]](https://arxiv.org/abs/1512.09300) and perceptual loss [[Dosovitsky and Brox, 2016](https://arxiv.org/abs/1602.02644), [Johnson et al., 2016](https://arxiv.org/abs/1603.08155)] and both of which goal to minimize the statistical difference between the feature vectors of real image and the reconstructed image. For the perceptual loss, feature vectors are generated from the intermediate layers of the VGG network [[Simonyan and Zesserman, 2015](https://arxiv.org/abs/1409.1556)] and for the discriminator feature matching loss, as the name suggests, they are generated using the layers of discriminator network. And the overall objective becomes as

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

#### **Implementation Details and Configurations**
In this section, authors detail the implementation and trainig configuration for each branch.

##### **Semantic Branch**
Semantic branch adopts Dilated Residual Networks [[Yu et al., 2017](https://arxiv.org/abs/1705.09914)] for semantic segmentation. Network is trained for 25 epochs.

##### **Geometric Branch**
They use Mask-RCNN [[He et al., 2018](https://arxiv.org/pdf/1703.06870.pdf)] for object proposal. For object meshes, they choose eight different CAD models from ShapeNet [[Chang et al., 2015](https://arxiv.org/abs/1512.03012)] as candidates. They set $$ \lambda_{reproj} = 0.1 $$. They first train the network alone with $$ \mathcal{L}_{pred} $$ using Adam optimizer [[Kingma et al., 2015](https://arxiv.org/abs/1412.6980)] by setting learning rate to $$10^{-3}$$ for $$ 256 $$ epochs and then fine-tune it with 
$$ \mathcal{L}_{pred} + \lambda_{reproj}\mathcal{L}_{reproj} $$ 
and REINFORCE with a learning rate of $$ 10^{-4} $$ for another 64 epochs.

##### **Textural Branch**
They use same architecture as in [Wang et al., 2018](https://arxiv.org/abs/1711.11585). They use two different discriminators of different scales and one generator. They set $$ \lambda_{FM} = 5 $$ and $$ \lambda_{Recon} = 10 $$ and train the textural branch for $$ 60 $$ epoch on Virtual KITTI and $$ 100 $$ epoch on Cityscapes.

### **Results and Comparisons**
Results in the paper are reported two-fold:
1.  Image editing capabilities of proposed method
2.  Analysis of design choices and accuracy of representations

The proposed method has been tested and validated upon two different datasets: **Virtual KITTI** [[Gaidon et al., 2016]](https://arxiv.org/abs/1605.06457){:target="_blank"} and **Cityscapes** [[Cordts et al., 2016]](https://arxiv.org/abs/1604.01685){:target="blank"}. Both quantitative and qualitative experiments are demonstrated over those two datasets. Additionally, authors create an image editing benchmark on **Virtual KITTI** to elaborate the effectiveness of the proposed method and also to compare editing power against the 2D baseline models.

#### **Datasets**
##### **Virtual KITTI**
The dataset is consisted of five different worlds rendered under ten different conditions, leading to sum of 21,260 images with instance and semantic segmentations. Each object has its own 3D ground truth attributes encoded. 

##### **Cityscapes**
The dataset contains 5,000 images with fine annotations and 20,000 with coarse annotations obtained in several conditions (seasons, daylight conditions, weather, etc.) in 30 cities with 30 classes of complexity. With fine annotations, they mean deep pixel-wise annotations for each object's semantic class. But the problem with this dataset is that it lacks 3D annotations for objects. Therefore, for each given Cityscapes images they first predict 3D attibutes with the geometric branch pre-trained on Virtual KITTI dataset, then try to infer 3D and deformation parameters.

#### **Image Editing Capabilities**
The disentanglement of attributes of an object provides an expressive 3D manipulation capability. Its 3D attributes can be changed. For example, it might scaled up and down or we can rotate it however we would like to, while keeping visual appearance at still. Likewise, we can change the appearance of the object or even the background in any way imaginable.

The proposed method is compared two baselines:
1.  **2D:** Given the source and target positions, only apply 2D translation and scaling, without rotation the object at all.
2.  **2D+:** Given the source and target positions, apply 2D translation and scaling, and rotate the 2D silhoutte of the object (instead of 3D shape) along the $$y-$$axis.

To allow to evaluate image editing capabilities, they have built the **Virtual KITTI Image Editing Benchmark** wherein it is consisted of 92 pairs of images, with the one being the original image and the other being the edited one in each pair.

{% include image.html url="/assets/figures/kitti.png" description="Example user editing results on Virtual KITTI" %}
{% include image.html url="/assets/figures/city_scapes.png" description="Example user editing results on Cityscapes" %}

For the comparison metric, they employ Learned Perceptual Image Patch Similarity (LPIPS) [[Zhang et al., 2018](https://arxiv.org/abs/1801.03924)], instead of adopting the L1 or L2 distance, since, while two images may differ slightly in perception, their L1/L2 distance may have a large value. They compare proposed model and the baselines and apply LPIPS in three different configurations:
1.  **The full image**: Evaluate the perceptual similarity on the whole image (whole)
2.  **All edited regions**: Evaluate the perceptual similarity on all the edited regions (all)
3.  **Largest edited region**: Evaluate the perceptual similarity only on the largest edited region (largest)

Addition to the quantitative evaluation, they also conduct an human study in which participants report their preferences on 3D-SDN over two other baselines according to which edited result looks closer to the target. They ask 120 human subjects on [Amazon Mechanical Turk](https://www.mturk.com/) and ask them whether which image is more realistic than the other in two settings: 3D-SDN vs. 2D and 3D-SDN vs 2D+.

{% include image.html url="/assets/figures/editing_results.png" description="Evaluations on Editing Benchmark" %}

In LPIPS metric, scores ranges from 0 to 1, 0 meaning that two images are the same. Therefore, lower scores are better. As you can see from the perception similarity scores, the proposed architecture overwhelms both baselines in every experiment setting. Likewise, in human study, users often prefer 3D-SDN over other two baselines.

#### **Analysis of Design Choices and Accuracy of Representations**
To understand contributions of each component proposed, they experiment on four diffent settings in which a different component excluded from the full proposed model ,,and compare them to the original, full model. The experiment configurations are:
1. **w/o $$\mathcal{L}_{reproj}$$**: Use only $$\mathcal{L}_{pred}$$
2. **w/o quaternion constraint**: Use full quaternion vector, instead of limiting to $$R$$
3. **w/o normalized distance $$\tau$$**: Predict the original distance in log-space rather than the normalized distance
4. **w/o MultiCAD and FFD**: Use single CAD model without free-form deformation
   
They also add a 3D box estimation model [[Mousavian et al., 2017]](https://arxiv.org/abs/1612.00496), which first infers the object's 2D bounding box and searches for its 3D bounding box, to the comparison list.

{% include image.html url="/assets/figures/design_results.png" description="Evaluation of Different Variants on Virtual KITTI" %}

In the above figure, different quantities are evaluated with different metrics:
*   For rotation, orientation similarity $$ (1 + \cos{\theta})/2 $$, where $$ \theta $$ is the geodesic distance between the predicted and the ground truth
*   For distance, absolute log-error $$ \| log t - log \tilde{t} \|$$
*   For scale, Euclidean distance $$ \| \textbf{s} - \tilde{\textbf{s}} \|_{2} $$
*   For reprojection error, compute per-pixel reprojection error between 2D silhouttes and ground truth segmentations


As you can see from the above figure, the smallest reprojection error and the highest orientation similarity have been met by the full model. It ultimately shows that all proposed components contribute to the final performance of full 3D-SDN model.

### Conclusion and Discussion
In this work, authors propose a novel encoder-decoder architecture that disentangles semantic, geometric and textural attributes of the detected objects into expressive and rich representations. With these representations, users are enabled easily manipulate the given images in various ways. This proposed method also overcomes the single-object limitations in its prior work and cancels out the occlusion problem. Although the main focus here is 3D-aware scene manipulation, learned distentangled and expressive representations can also be used in various tasks such as image captioning.

To my discussion, even though the proposed method eliminates couple of limitations in its prior state-of-art works, it is still limited to just one specific set of objects: vehicles. Without introducing new meshes, it is not applicable to more complex and diverse sceneries, such as indoor images. For any regular indoor image, we face lots of different types of objects. And as the number of object classes and the number of mesh types in each class increases, unfortunately overall complexity increases dramatically. Also, the model might not have the capability to handle much deformable shapes like human bodies without introducing more deformation parameters.

For the slides presented in the lecture, please follow the [link](https://www.dropbox.com/s/kt64dvwme3xhi3o/seminar_presentation.pdf?dl=0).
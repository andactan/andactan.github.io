---
layout: post
title:  "3D-Aware Scene Manipulation via Inverse Graphics"
date:   2019-12-07 16:18:21 +0100
categories: jekyll update
---
# Abstract

As humans, we renown at perceiving the world around us and simultaneously make deductions out of it. But, what is interesting and also magnificient part about it is, as we do perceive the world around, we are also able to simulate and imagine in what ways changes can appear unto the our perceptions for future scenarios. For instance, we can, without showing much effort, 
detect and recognize cars on a road, and infer their attributes. At the same time, we have also a special power by which we could imagine how cars may move on the road, or they will ever rotate right or left, or even think about what might
be a more plausible color a car may have other than the driver's choice. 

Motivation arises from this very specific human ability. Can we enable a neural network architecture gain such ability?

# Introduction

To grant that ability to the machines, we should let them understand given scenes and encode them into latent space representations. From those intermediate representations, they should also be able to generate plausible images back. And actually, deep generative models and specifically **Generative Adversarial Networks (GANs)** [[Goodfellow et, al, 2014]](https://arxiv.org/abs/1406.2661)excels at this duty with their terrific and simplistic encoder-decoder architecture. However, deep generative models have their flaws in which intermediaterepresentations are often limited to a single object, not easy to interpret, and missing the 3D structure behind 2D projection. As aresult, deep generative models are not perfectly fit for scene manipulation tasks such as moving objects around. Additioally, as itis a scene manipulation task at hand, it is needed to have human-interpretable and intuitive representations so that any human useror a graphics engine is enabled to use.


In this paper, motivated by the aforementioned human abilities, authors propose an expressive, disentangled and interpretable scene representation method for machines.To overcome those above problems, 


$$ x = A^{3} $$
{: style="text-align:center;"}

This is a test paragraph.

Another paragraph.
---
title: A bit-by-bit guide to the equations governing differentiable neural computers
author: Juan Antonio Pérez-Ortiz
layout: default
redirect_from: "/index.html"
use_math: true
---

# {{page.title}}

Written by **Juan Antonio Pérez-Ortiz**, researcher at Universidad de Alicante, Spain.

**Last update:** October 20 2017.

**Comments and suggestions:** feel free to send any comments or suggestions to enhance this guide through the [comments section](#disqus_thread) at the end of this page.

<div id="index" markdown="1">
**Table of contents:** &nbsp;
<a class="label label-blue1" href="#architecture">Architecture</a>
<a class="label label-blue2" href="#memory">Memory</a>
<a class="label label-blue3" href="#read-operation">Read operation</a>
<a class="label label-blue1" href="#write-operation">Write operation</a>
<a class="label label-blue2" href="#content-based-addressing">Content-based addressing</a>
<a class="label label-blue3" href="#temporal-memory-linkage">Temporal memory linkage</a>
<a class="label label-blue1" href="#computation-of-the-temporal-link-matrix">Computation of the temporal link matrix</a>
<a class="label label-blue2" href="#dynamic-memory-allocation">Dynamic memory allocation</a>
<a class="label label-blue3" href="#computation-of-the-allocation-weighting-vector">Computation of the allocation weighting vector</a>
<a class="label label-blue1" href="#cite">How to cite this work</a>
<a class="label label-blue2" href="#about">About the author</a>
<a class="label label-blue3" href="#sources">Sources</a>
<a class="label label-blue1" href="#license">License</a>
<a class="label label-blue2" href="#disqus_thread">Comments</a>
</div>

<div style="display:none">
$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\b}[1]{\boldsymbol{#1}}
\newcommand{\r}[1]{\mathrm{#1}}
\newcommand{\c}[1]{\mathcal{#1}}
$$
</div>

{% comment %}
- Some comments.
{% endcomment %}

This guide attempts to gradually present the various pieces involved in the mathematical formulation of the differentiable neural computer (DNC) model, a memory-augmented neural network first introduced in the paper by Alex Graves, Greg Wayne et al., researchers at [DeepMind](https://deepmind.com/), [published](http://www.nature.com/nature/journal/v538/n7626/abs/nature20101.html) in the Nature journal in October 2016. The paper in question, along with certain others published before it, is relevant in that it sows the seeds for an even brighter future of neural networks, potentially leading to their application in algorithmic tasks that were formerly inaccessible. The DNC architecture is also a good example of how to move concepts that are not in principle differentiable (e.g., the management of a read-write memory) to the realm of differentiable models that can be trained by means of gradient descent.

The guide provides a step-by-step analysis of the equations that make up the model at a slower pace than that originally followed in the methods section of the paper cited above, in the hope that doing so will facilitate the understanding of the model for those who require extra details in order to fully comprehend the whole story. I, personally, needed hours to feel comfortable with all of the equations used to describe the model in the aforementioned paper in spite of my expertise as regards neural networks; I therefore decided that it might be a good idea to prepare this document so as to help those (maybe you!) in a situation similar to mine.

The primary source of information regarding the DNC model is obviously the [original paper](http://www.nature.com/nature/journal/v538/n7626/abs/nature20101.html) and its contents prevail over any statement made in this guide. Deeply understanding how DNCs are designed could also make it possible to suggest modifications to the architecture that could lead to alternative models better suited to specific tasks.

This guide assumes that the reader understands the basics of machine learning and neural networks (weights and connections, activation functions, the gradient descent optimization algorithm, recurrent networks, LSTM cells, etc.) along with the fundamentals of matrix algebra, but, apart from that, no additional advanced knowledge is, in principle, required to follow the discussion. Many good introductions to the topic of neural networks can be found on the web. Another additional resources as regards DNCs are:

- the [webpage](https://deepmind.com/research/dnc/) and the [blog entry](https://deepmind.com/blog/differentiable-neural-computers/) in the DeepMind website,
- the [source code](https://github.com/deepmind/dnc) (written in Python with the TensorFlow and Sonnet libraries) released by DeepMind six months after the publication of the paper,
- the brief [description](http://distill.pub/2016/augmented-rnns/#neural-turing-machines) of the Neural Turing Machine (a predecessor of DNCs sharing some features with them) in one of the papers in the Distill journal,
- the video of the [talk given by Alex Graves](https://youtu.be/steioHoiEms) at the NIPS 2016 conference,
- the [video review](https://youtu.be/r5XKzjTFCZQ) of the architecture and the source code by Siraj Raval,
- the [presentation of the paper](http://www.nature.com/nature/journal/v538/n7626/full/nature19477.html) in the Nature journal,
- the [review work](https://web.stanford.edu/class/cs224n/reports/2753780.pdf) by the Stanford student Carol Hsin,
- and the analysis of the analogy between DNCs and the [human memory](https://greydanus.github.io/2017/02/27/differentiable-memory-and-the-brain/) by Sam Greydanus.

Throughout this guide, I will often use what I call <span class="term" markdown="1">dualistic simplification</span> in order to introduce discussions regarding many of the components of a DNC in a educational manner. The idea behind this is to start by assuming that the model could behave like a binary (dualistic) system in the same sense that, for example, a conventional memory location is always either written or not written, but cannot be *half* written. This behavior is what a programmer expects in most programming languages, in which an assignment instruction writes to the memory location of a particular variable and leaves the remaining variables untouched; for instance, <code>x=3</code> means that the variable <code>x</code> will be *fully* written, and its previous value completely removed, while the remaining variables that are visible from the current scope will not be written at all. Remarkably, however, *real life* in the DNC world is slightly different (half serious, half joking, we could say that things are slightly *differentiable*; see later) from this approximation: all the values streaming through the DNC are continuous real values, signifying that a switch is rarely completely on or off, or that a memory pointer hardly ever refers to a unique location in the memory. In some cases, it may occur that the switch is *almost* on or off, or that the memory pointer is *almost* exclusively referring to a single memory location, but in some other cases, the switch may have an intermediate degree of activation or the pointer may be simultaneously focusing on more than one memory location at the same time (paying a different degree of attention to each of them). If you have some knowledge of how a regular computer operates, this non-binary pluralistic behavior of DNCs is at the other end of many strong preconceptions you may have about how a computer and its memory work. Throughout this guide, you are expected to open your mind to fuzzier mechanisms and embrace a view of DNCs as a pluralistic approach to computing. In this document, the beginning of those fragments of text that assume the dualistic simplification are marked with a yin-yang symbol <span class="dual"></span>. I hope that the dualistic simplification will prove useful to the reader in order to attain a rough idea of how a particular operation works in the DNC model before learning it in its entire realization.

<div class="disg" markdown="1">
#### Threshold activation functions

The first neural networks proposed by Warren McCulloch and Walter Pitts back in the 1940s used linear threshold units, a form of neuron whose binary output is computed by means of a [Heaviside step activation function](http://mathworld.wolfram.com/HeavisideStepFunction.html) (activation is exactly 0 or 1). This kind of activation function (or extensions of it) may, in principle, appear to be suitable as regards obtaining dualistic references to memory locations in DNCs. Why, then, are they not used to embrace the dualistic view? Because a step-like function is not adequate for the gradient descent training algorithm since its derivative is $0$ at every point (except at the step point) and, therefore, provides no useful information for common learning algorithms. Other forms of activation functions will consequently be used and the references to memory locations will be pluralistic rather than dualistic.

</div>

I shall begin by describing the basic architecture of a DNC, after which I shall go on to discuss how the explicit memory is represented in a DNC and how data is read from and written to this memory.

## Architecture

Seen from afar, a DNC is similar to many other neural networks: it reads an external input $\b{x}_t$ and produces an output $\b{y}_t$ which is intended to represent things such as the estimation of a class, the prediction of the next element in a sequence, etc. The mapping from inputs to outputs is computed as a set of differentiable operations which depend on a set of weights that are automatically learned in order to minimize a loss function that compares the outputs emitted by the model with the desirable outputs contained in a training set. Gradient descent can then be used to estimate those weights for a particular task.

A more careful look will, however, reveal an explicit memory that is used to store vectors and later retrieve them. A DNC is composed of a <span class="term" markdown="1">controller</span>, a <span class="term" markdown="1">memory</span> and an <span class="term" markdown="1">output module</span>. The controller is responsible for emitting a <span class="term" markdown="1">controller output vector</span> $\b{\nu}_t$ given the current external input $\b{x}_t$, but also interacts with the memory by emitting a vector of <span class="term" markdown="1">interface parameters</span> $\b{\xi}_t$ that are used to regulate the memory operation at each time step. These interface parameters include values that directly or indirectly determine which memory locations will be read or written at the current time step. The memory has a set of $R$ <span class="term" markdown="1">read heads</span> and one <span class="term" markdown="1">write head</span>; the values read by the $i$-th read head at time step $t$ are denoted as $\b{r}^{i}_t$. The controller can, in principle, be any differentiable system (e.g., a feedforward neural network or a recurrent neural network) that computes a function $\c{N}$ in order to emit both vectors (the controller output and the interface parameters) given the external input and the previously read vectors:

$$
[\b{\nu}_t,\b{\xi}_t] = \c{N}\left(\b{x}_t,\b{r}^{1}_{t-1},\ldots,\b{r}^{R}_{t-1}\right)
$$

In the paper published in Nature, Alex Graves, Greg Wayne et al. use an LSTM-based recurrent neural network for the controller function $\c{N}$ in all the experiments, as a result of which these DNC controllers also use the state information (a much simpler form of memory) corresponding to the past input vectors $\b{x}\_{t-1},\b{x}\_{t-2},\ldots$

The values read from the memory at time $t$ ($\b{r}^{i}\_t$, with $i=1,\ldots,R$) are used not only to feed the controller at time $t+1$ but also to compute the global output $\b{y}\_t$ of the DNC by means of another differentiable system that computes a function $\c{N}'$:

$$
\b{y}_t = \c{N}'\left(\b{\nu}_t,\b{r}^{1}_{t},\ldots,\b{r}^{R}_{t}\right)
$$

In the paper, a simple single-layer feedforward neural network is used for the global output function $\c{N}'$. Note that the values of $\b{r}^{i}\_t$ at time $t$ cannot be involved in the computation of the interface parameters $\b{\xi}\_t$ because these parameters are precisely required to obtain each $\b{r}^{i}_{t}$. Note also that both neural networks $\c{N}$ and $\c{N}'$ are trained together (end-to-end) by simply using a training set made of inputs $\b{x}_t$ and the corresponding desired values for the global outputs $\b{y}_t$.


## Memory

Differentiable neural computers (DNCs) are an example of *memory augmented neural networks*. The memory $\b{M}_t$ stores a collection of $N$ real-valued vectors in $\R^W$, such as $[0.2, 0.6, 1.2] \in \R^3$. The *word size* $W$ of the memory locations is, in principle, set as constant for a particular DNC, and all vectors stored in memory  consequently have the same length. The $N$ vectors are deployed in an $N \times W$ matrix; the following $4 \times 3$ matrix, for example, represents a memory with $N=4$ rows or <span class="term" markdown="1">locations</span> in which the previous example vector is stored at the memory location $2$ (in accordance with the original article, the indexes start at one):

$$
\b{M}_t =
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
\co{0.2} & \co{0.6} & \co{1.2} \\
0 & 0 & 0 \\
-0.1 & -0.05 & 0
\end{bmatrix}
\label{memo}
$$

Rows arbitrarily represent locations, but locations could also have been represented using columns. The number of memory locations $N$ (i.e., the <span class="term" markdown="1">capacity</span> of the memory; in this case $N=4$) is also usually kept constant but it could, in principle, be dynamically increased to enable the storage of extra vectors once all the positions have been allocated and new vectors need to be stored (the concept of memory allocation will be discussed later). In this case, all the other vectors and matrices in the DNC defined in terms of $N$ would obviously also need to be resized.

The DNC memory is time-varying (like regular computer memories), repeatedly being read and written. This is why we use the subindex $t$: the matrix $\b{M}_t$ represents the memory contents at time step $t$. As will be explored later, at every time step $t$, the memory may be optionally read and optionally written.

A series of read heads and a write head interact with the contents of the memory. These are discussed below.

## Read operation

Each DNC integrates at least one read head which can be used to attain access to the content of the memory. Read heads use a vector called <span class="term" markdown="1">weighting</span> to convey the particular memory locations to be read at each time. <span class="dual"></span> Let us first follow the dualistic simplification and assume that this weighting can refer to a single memory location by using the *one-hot encoding* which, in the case of a memory with a capacity $N=4$, implies that, rather than representing the different locations as $1, 2, 3, 4$, they will be represented as the one-hot vectors $[1;0;0;0]$, $[0;1;0;0]$, $[0;0;1;0]$, $[0;0;0;1]$, respectively. This representation seems particularly relevant to our interests, since retrieving the content of a particular location would be expressed as a well-known matrix product between the memory matrix and the vector representing the location; for example, it would be possible to retrieve the content of location $2$ (represented as the weighting $[0;1;0;0]$) in the memory configuration of equation \eqref{memo} with:

$$
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
\co{0.2} & \co{0.6} & \co{1.2} \\
0 & 0 & 0 \\
-0.1 & -0.05 & 0
\end{bmatrix}^\top
\begin{bmatrix}
0 \\
1 \\
0 \\
0
\end{bmatrix}
=
\begin{bmatrix}
-0.5 & \co{0.2} & 0 & -0.1 \\
0.01 & \co{0.6} & 0 & -0.05 \\
3.1 & \co{1.2} & 0 & 0
\end{bmatrix}
\begin{bmatrix}
0 \\
1 \\
0 \\
0
\end{bmatrix}
=
\begin{bmatrix}
\co{0.2} \\
\co{0.6} \\
\co{1.2}
\end{bmatrix}
$$

where $\top$ denotes the transpose of a matrix. The vector representing the location to be read at time $t$ is called the <span class="term" markdown="1">read weighting</span> for reasons that will become clear later, and is represented as $\b{w}^\r{r}_t$ (in the running example, $\b{w}^\r{r}_t = [0;1;0;0]$). A position in memory can, therefore, be read with the operation $\b{M}^\top_t \b{w}^\r{r}_t$. There is a companion notebook that allows you to check how [these values]({{ "dnc-notebook.html#Example-of-a-dualistic-read-operation" | prepend: site.baseurl }}) are obtained and apply the formulae to your own inputs.

<div class="disg" markdown="1">
#### Column vectors

Unless otherwise stated, all vectors used in the equations governing DNCs are column vectors. In order to simplify their in-line writing, they will be represented in text as $[0;1;0;0]$, that is, as rows in which element vectors are separated using semicolons rather than commas.

</div>

The formulation shown above belongs to the dualistic world, a world in which, as mentioned in the introduction, a memory location is fully read or is not read at all; when working with DNCs, however, the functional unit (i.e., the <span class="term" markdown="1">read head</span>) that emits the read weighting will not generate this one-hot *clean* output, but rather a distribution (or, in other words, a <span class="term" markdown="1">weighting</span>) over all the memory locations; some of the reasons behind this were presented in the introduction when I discussed why threshold activation functions are not a good choice. As a result of this, $\b{w}^\r{r}_t[i]$, that is, the $i$-th element of $\b{w}^\r{r}_t$, will be interpreted as the degree to which memory location $i \in [1,N]$ is involved in the reading operation. For example, a weighting $\b{w}^\r{r}_t= [0;0.8;0.1;0.1]$ indicates that the second memory location gets eight times more attention than the third or fourth locations, and the first location gets no attention at all; using this weighting to retrieve the contents of the memory configuration \eqref{memo} would give:

$$
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
0.2 & 0.6 & 1.2 \\
0 & 0 & 0 \\
-0.1 & -0.05 & 0
\end{bmatrix}^\top
\begin{bmatrix}
0 \\
0.8 \\
0.1 \\
0.1
\end{bmatrix}
=
\begin{bmatrix}
-0.5 & 0.2 & 0 & -0.1 \\
0.01 & 0.6 & 0 & -0.05 \\
3.1 & 1.2 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
0 \\
0.8 \\
0.1 \\
0.1
\end{bmatrix}
=
\begin{bmatrix}
0.15 \\
0.475 \\
0.96
\end{bmatrix}
$$

Note that the resulting vector is still *close* to the content of the second memory location, but that the residual attention paid to the third and fourth locations prevents it from exactly matching the second location. The companion notebook allows you to check how [these values](/funicular/dnc-notebook.html#Example-of-a-non-dualistic-read-operation) are obtained. More specifically (as will be seen later), the operations performed by the read head produce weightings $\b{w}^\r{r}_t \in \Delta_N$, with $\Delta_N$ defined as:

$$
\Delta_N = \left\{ \b{\alpha} \in \R^N, \alpha[i] \in [0,1], \sum_{i=1}^N \alpha[i] \leq 1 \right\}
$$

The DNC allows for more than one read head in the hope that multiple simultaneous readings may simplify the dynamics that would otherwise be needed to perform a series of consecutive readings at steps $t$, $t+1$, etc. A set of $R$ read weightings $\b{w}^{\r{r},i}_t$ with $i \in {1,2,\ldots,R}$ are consequently generated for the $R$ read heads. The vector read by the $i$-th read head at time $t$ is:

$$
\b{r}^i_t = \b{M}^\top_t \b{w}^{\r{r},i}_t
$$

Note that this mechanism employed to retrieve the contents of a particular location is differentiable. If no location has to be read at the current time step by the $i$-th read head, then the read weighting vector $\b{w}^{\r{r},i}_t$ will be $\b{0}$, as will $\b{r}^i_t$.

<div class="disg" markdown="1">
#### Weightings

As we will be seeing, DNCs have different weightings that are used to selectively focus on different memory locations. These weightings are vectors with components in the range of $[0,1]$ that may add up to exactly $1$ or to a value in $[0,1]$, depending on the purpose of the weighting. Most of the weightings belong to the second group, but a few of them belong to the first. For example, as will be seen later, at every time step, DNCs compute a similarity measure between a given lookup key vector and the content of each memory location; the result is a weighting whose elements indicate the proportion of the total similarity corresponding to each location. In this case, the weighting vector will be an element of $\c{S}_N$, a set defined as:

$$
\begin{eqnarray}
\c{S}_N & = & \left\{ \b{\alpha} \in \R^N, \alpha[i] \in [0,1], \sum_{i=1}^N \alpha[i] = 1 \right\} \\
    & = & \left\{ \b{\alpha} \in [0,1]^N, \sum_{i=1}^N \alpha[i] = 1 \right\}
\end{eqnarray}
$$

$\c{S}_N$ is known in geometry as the unit  $(N-1)$-[simplex](https://en.wikipedia.org/wiki/Simplex#The_standard_simplex) (or standard $N-1$ simplex). For example, in the case of $N=2$, the set $\c{S}_N$ will comprise the points of the line segment (a 1-simplex) which can be seen in this [graph](http://www.wolframalpha.com/input/?i=plot+x%2By%3D1+and+x%3E0+and+y+%3E+0,+x%3D0..1,+y%3D0..1); for $N=3$, the corresponding $2$-simplex is the equilateral triangle (with vertices $[1,0,0]$, $[0,1,0]$ and $[0,0,1]$) that can be seen in this other [graph](https://upload.wikimedia.org/wikipedia/commons/3/38/2D-simplex.svg); for a weighting with $N=4$, the corresponding 3-simplex is a [tetrahedron](http://www.wolframalpha.com/input/?i=tetrahedron), etc.

The condition that the vector elements add up to $1$ may be relaxed in those scenarios in which attention is optional; for example, as it is not mandatory to write to a memory location at every time step, we should allow null weightings. In these cases, the components will add up to *at most* $1$, and the weighting vector will be an element of $\Delta_N$, a set which is defined as:

$$
\begin{eqnarray}
\Delta_N & = & \left\{ \b{\alpha} \in \R^N, \alpha[i] \in [0,1], \sum_{i=1}^N \alpha[i] \leq 1 \right\} \\
         & = & \left\{ \b{\alpha} \in [0,1]^N, \sum_{i=1}^N \alpha[i] \leq 1 \right\}
\end{eqnarray}
$$

$\Delta_N$ is known as the *corner of the cube* because it includes the points in the non-negative orthant of $\R^N$ that are *under* $\c{S}_N$, or, equivalently, all the points in the non-negative orthant with the $N-1$ unit simplex as a boundary. For example, in the case of weightings with $N=3$, the set $\Delta_N$ will be the tetrahedron (or triangular pyramid) under the [triangular](https://upload.wikimedia.org/wikipedia/commons/3/38/2D-simplex.svg) $2$-simplex shown above.

</div>

## Write operation

DNCs have only one write head, meaning that they can only store one new vector in the memory at each time step. As with the read heads, the write head generates a <span class="term" markdown="1">write weighting</span> $\b{w}^\r{w}_t \in \Delta_N$ in an analogous manner, and this contains the degree to which each memory location will be involved in the write operation. In the case of the read heads, the read weighting is the only thing needed to interact with the memory. The write head, however, also requires the vector to be written $\b{v}_t \in \R^W$ and an erase vector $\b{e}_t \in [0,1]^W$ which determines to what degree the elements of each of the locations involved in the write operation have to be erased before the update (note that this is analogous with the way in which LSTM cells use forget gates that have the ability to erase the content of the cell before the input gate opens). If $\b{e}_t[i] = 1$, the $i$-th element of the corresponding location in the memory will be completely erased; keep in mind that, as discussed for the read operation, the pluralistic nature of DNCs signifies that a location will rarely be completely erased.

The equation that determines the new content of the memory is:

$$
\b{M}_t = \b{M}_{t-1} \circ \left( \b{E} - \b{w}^\r{w}_t \b{e}^\top_t \right) + \b{w}^\r{w}_t \b{v}^\top_t
\label{newm}
$$

where $\circ$ denotes the *element-wise matrix product* and $\b{E}$ is a matrix of ones with the same size as the memory $\b{M}$. The [element-wise product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) (or Hadamard product) of two matrices is:

$$
\left( \b{A} \circ \b{B} \right)[i,j] = \b{A}[i,j] \cdot \b{B}[i,j]
$$

<span class="dual"></span> With the dualistic simplification, our previous vector stored at memory location $2$ (represented as the weighting $\b{w}^\r{w}_t = [0;1;0;0]$) could be completely replaced (by using $\b{e}_t = \b{1}$) with the new vector $[-1.5;-1.3;-1.1]$ by using these parameters:

$$
\b{w}^\r{w}_t =
\begin{bmatrix}
0 \\
1 \\
0 \\
0
\end{bmatrix}
\qquad
\b{v}_t =
\begin{bmatrix}
-1.5 \\
-1.3 \\
-1.1
\end{bmatrix}
\qquad
\b{e}_t =
\begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix}
$$

For these values, \eqref{newm} would give:

$$
\begin{eqnarray}
\b{M}_t
& = &
\b{M}_{t-1} \circ \left( \b{E} - \b{w}^\r{w}_t \b{e}^\top_t \right) + \b{w}^\r{w}_t \b{v}^\top_t \\[2ex]
& = &
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
\co{0.2} & \co{0.6} & \co{1.2} \\
0 & 0 & 0 \\
-0.1 & -0.05 & 0
\end{bmatrix} \circ
\left(\,
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
-
\begin{bmatrix}
0 \\
1 \\
0 \\
0
\end{bmatrix}
\begin{bmatrix}
1 & 1 & 1
\end{bmatrix}
\,\right) \\[2ex]
& &
\qquad +
\begin{bmatrix}
0 \\
1 \\
0 \\
0
\end{bmatrix}
\begin{bmatrix}
\co{-1.5} & \co{-1.3} & \co{-1.1}
\end{bmatrix}
\\[2ex]
& = &
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
\co{0.2} & \co{0.6} & \co{1.2} \\
0 & 0 & 0 \\
-0.1 & -0.05 & 0
\end{bmatrix} \circ
\begin{bmatrix}
1 & 1 & 1 \\
0 & 0 & 0 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 & 0 \\
\co{-1.5} & \co{-1.3} & \co{-1.1} \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
\\[2ex]
& = &
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
\co{0} & \co{0} & \co{0} \\
0 & 0 & 0 \\
-0.1 & -0.05 & 0
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 & 0 \\
\co{-1.5} & \co{-1.3} & \co{-1.1} \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
\\[2ex]
& = &
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
\co{-1.5} & \co{-1.3} & \co{-1.1} \\
0 & 0 & 0 \\
-0.1 & -0.05 & 0
\end{bmatrix}
\end{eqnarray}
$$

The companion notebook allows you to check how [these values](/funicular/dnc-notebook.html#Example-of-a-dualistic-write-operation) are obtained and apply the formulae to your own inputs. Note how the erasing and the updating have been wisely set out as a series of multiplications, additions and subtractions, all of which are differentiable. Obviously, $\b{v}_t \in \R^W$ like the other vectors stored in the memory. <span class="dual"></span> Although $\b{e}_t$ is a vector in $[0,1]^W$, we would expect it to always be a vector of ones, that is $\b{e}_t = \b{1}$, as it seems reasonable to completely erase the $W$ elements from a memory location before updating it. Again, this would be the case in a dualistic binary world, but it does not apply to actual DNCs for which a number of elements in the memory vector may be partially erased or not even erased at all. Add to this the fact that the write weighting could be focusing on different locations with different non-null degrees of attention and you will get a good idea of the pluralistic nature of DNCs. The learning algorithm will hopefully determine the best vectors for each time $t$ and they will consist of elements that are not as dualistic as in the running example.

Let us now consider the same memory matrix $\b{M}_t$ and vector to be written $\b{v}_t$, but let us change the write weighting and erase vector to non-dualistic alternatives:

$$
\b{w}^\r{w}_t =
\begin{bmatrix}
0 \\
0.8 \\
0.1 \\
0.1
\end{bmatrix}
\qquad
\b{e}_t =
\begin{bmatrix}
1 \\
0.5 \\
0
\end{bmatrix}
$$

With these pluralistic values, \eqref{newm} would give:

$$
\begin{eqnarray}
\b{M}_t
& = &
\b{M}_{t-1} \circ \left( \b{E} - \b{w}^\r{w}_t \b{e}^\top_t \right) + \b{w}^\r{w}_t \b{v}^\top_t \\[2ex]
& = &
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
0.2 & 0.6 & 1.2 \\
0 & 0 & 0 \\
-0.1 & -0.05 & 0
\end{bmatrix} \circ
\left(\,
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
-
\begin{bmatrix}
0 \\
0.8 \\
0.1 \\
0.1
\end{bmatrix}
\begin{bmatrix}
1 & 0.5 & 0
\end{bmatrix}
\,\right) \\[2ex]
& &
\qquad +
\begin{bmatrix}
0 \\
0.8 \\
0.1 \\
0.1
\end{bmatrix}
\begin{bmatrix}
-1.5 & -1.3 & -1.1
\end{bmatrix}
\\[2ex]
& = &
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
0.2 & 0.6 & 1.2 \\
0 & 0 & 0 \\
-0.1 & -0.05 & 0
\end{bmatrix} \circ
\begin{bmatrix}
1 & 1 & 1 \\
0.2 & 0.6 & 1 \\
0.9 & 0.95 & 1 \\
0.9 & 0.95 & 1
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 & 0 \\
-1.2 & -1.04 & -0.88 \\
-0.15 & -0.13 & -0.11 \\
-0.15 & -0.13 & -0.11
\end{bmatrix}
\\[2ex]
& = &
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
0.04 & 0.36 & 1.2 \\
0 & 0 & 0 \\
-0.09 & -0.0475 & 0
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 & 0 \\
-1.2 & -1.04 & -0.88 \\
-0.15 & -0.13 & -0.11 \\
-0.15 & -0.13 & -0.11
\end{bmatrix}
\\[2ex]
& = &
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
-1.16 & -0.68 & 0.32 \\
-0.15 & -0.13 & -0.11 \\
-0.24 & -0.1775 & -0.11
\end{bmatrix}
\end{eqnarray}
$$

Once again, the companion notebook allows you to check how [these values](/funicular/dnc-notebook.html#Example-of-a-non-dualistic-write-operation) are obtained. Equation \eqref{newm} is probably better understood if shown at the element level:

$$
\b{M}[i,j] = \b{M}[i,j] \left( 1 - \b{w}^\r{w}_t[i] \b{e}_t[j] \right) + \b{w}^\r{w}_t[i] \b{v}_t[j]
$$

where $[i,j]$ and $[i]$ refer to the corresponding element in a matrix or vector, respectively.

<div class="disg" markdown="1">
#### <span class="interface"></span> New interface parameters in this section

We have just introduced the first parameters emitted by the controller at time step $t$, namely, the write vector $\b{v}_t$ and the erase vector $\b{e}_t$. The controller actually emits two vectors, $\b{v}_t$ and $\hat{\b{e}}_t$, the first of which is used unaltered as the write vector. The [logistic sigmoid](https://www.wolframalpha.com/input/?i=plot+1%2F(1%2Be**-x)+from+-5+to+5) function $\sigma$ is, meanwhile, applied to $\hat{\b{e}}_t$ in order to constrain its elements to $[0,1]$ and obtain $\b{e}_t$:

$$
\b{e}_t = \sigma \left( \hat{\b{e}}_t \right)
$$

where

$$
\sigma(x) = \frac{1}{1+\r{exp}(-x)}
$$

</div>

I shall add a block like that above to each section in which new interface parameters are presented. As seen above, the beginning of each of these blocks is marked with the astronomical symbol for Uranus <span class="interface"></span>.


## Intermezzo

In the previous sections, we have explored the fundamentals of DNCs and their memory modus operandi. Basically, a matrix stores vectors, and weightings are used to decide which locations take part in the read or write operations. How these weightings are computed will be our main concern for the rest of the guide, so let's go to it!

## Content-based addressing

How do the read heads and the write head compute the corresponding read and write weightings? Various memory addressing schemes are used for this purpose, but for now, let me introduce the simplest: <span class="term" markdown="1">content-based addressing</span>. Content-based addressing allows us to refer (via the read or write weightings) to the memory locations whose vectors are more similar to a given <span class="term" markdown="1">lookup key</span> $\b{k} \in \R^W$. <span class="dual"></span> Recall that in a dualistic system, one would expect this kind of addressing mode to provide access to the most similar location only, but the pluralistic approach of DNC implies that all vectors will be affected to a greater or lesser extent.

Given a lookup key vector $\b{k} \in \R^W$, a vector-valued function $\c{C}(\cdot)$ is used to produce a weighting in $\c{S}_N$ (see the discussion concerning weightings above) with a weight for each memory location, such that $\c{C}(\cdot)[i] > \c{C}(\cdot)[j]$ indicates that the content of location $i$ is more similar to $\b{k}$ than the content of location $j$. As mentioned previously, the sum of all the elements of a vector in $\c{S}_N$ is $1$; it is, therefore, possible to interpret $\c{C}(\cdot)$ as a probability distribution over the memory locations.

Besides the memory matrix and the lookup key, the function $\c{C}$ also uses a scalar $\beta \in [1,\infty)$ that represents <span class="term" markdown="1">key strength</span> as discussed later. The $i$-th element of $\c{C}$ is obtained as follows:

$$
\c{C}(\b{M},\b{k},\beta)[i] = \frac{
  \r{exp}\left(\c{D}(\b{k},\b{M}[i,\cdot])\right)^\beta
}{
  \sum_{j=1}^N \r{exp}\left(\c{D}(\b{k},\b{M}[j,\cdot])\right)^\beta
}
$$

$\b{M}[i,\cdot]$ denotes the $i$-th row of matrix $\b{M}$ (represented as a column vector); in our case, this corresponds to the vector stored at memory location $i$. $\c{D}$ is the <span class="term" markdown="1">cosine similarity</span>, a scalar similarity measure between two vectors obtained as the cosine of the angle between them. If the two vectors have the same orientation (for example, $[0.5;0.5;0.5]$ and $[0.1;0.1;0.1]$), then this angle is $0^\circ$, and $\c{D}$ obtains its maximum value of $1$; if the vectors have exactly opposite orientations, then the angle is $180^\circ$, and $\c{D}$ obtains its minimum value of $-1$. Note that orientation and not magnitude is the relevant feature here: two vectors do not necessarily need to be  equal to have a cosine similarity of $1$. For any two vectors, the value of $\c{D}$ will range between $-1$ and $1$. The cosine similarity $\c{D}$ is defined as follows:

$$
\c{D}(\b{u},\b{v}) = \frac{\b{u} \cdot \b{v}}{\lVert\b{u}\rVert \, \lVert\b{v}\rVert}
$$

The numerator is the dot product of $\b{u}$ and $\b{v}$, computed as:

$$
\b{u} \cdot \b{v} = \sum_{i=1}^W u[i] v[i]
$$

where $W$ is the length of $\b{u}$ and $\b{v}$. The denominator is the product of the Euclidean norm (also known as the $2$-norm) of each vector:

$$
\lVert\b{u}\rVert = \sqrt{\sum_{i=1}^W u^2[i]}
$$

Let us assume that we have the following capacity 2 memory:

$$
\b{M} =
\begin{bmatrix}
-0.5 & 0.01 & 3.1 \\
\co{0.2} & \co{0.6} & \co{1.2} \\
\end{bmatrix}
$$

Consider that the lookup vector is $\b{k}= [0.3;0.5;1]$ (which, as will intuitively be noted, is *closer* to location $2$ than to location $1$). The cosine similarity between the lookup vector and each of the memory vectors is:

$$
\begin{eqnarray}
\c{D}(\b{k},\b{M}[1,\cdot]) & = & \c{D}(\b{k},[-0.5;0.01;3.1]) & = & 0.81295 \\
\c{D}(\b{k},\b{M}[2,\cdot]) & = & \c{D}(\b{k},[0.2;0.6;1.2]) & = & 0.99349
\end{eqnarray}
$$

This is a [graphical representation of the three vectors](https://www.wolframalpha.com/input/?i=vector%7B0.3,0.5,1%7D,+vector%7B-0.5,0.01,3.1%7D,+vector%7B0.2,0.6,1.2%7D). It is now possible to calculate the values of $\c{C}$ for two different values of $\beta$, namely $\beta \in \\{1,10\\}$. The companion notebook allows you to check how [all these values](/funicular/dnc-notebook.html#Content-based-addressing) are obtained.

$$
\begin{eqnarray}
\c{C}(\b{M},\b{k},\beta \equiv 1) & = &
\begin{bmatrix}
0.454987 \\
0.545012
\end{bmatrix} \\[1.5ex]
\c{C}(\b{M},\b{k},\beta \equiv 10) & = &
\begin{bmatrix}
0.141197 \\
0.858803
\end{bmatrix}
\end{eqnarray}
$$

Note the [effect](https://www.wolframalpha.com/input/?i=plot+e%5Ex,+e%5E(2*x),+e%5E(3*x)) of the exponent $\beta$ on the exponential function $\r{exp}$. Closer values are more separated by the function as the value of $\beta$ grows. In both cases, the second element of the resulting vector is greater than the first, thus illustrating the fact that the second memory location is closer to $\b{k}= [0.3;0.5;1]$ than the first; the second element is, however, considerably greater than the first when a larger value of $\beta=10$ is used. Recall that the two elements have to add up to $1$.

At each time $t$, the controller emits one lookup vector for each read head, namely $\b{k}^{\r{r},i}_{t}$ with $i=1,\ldots,R$, and a single lookup vector $\b{k}^{\r{w}}_t$ for the write head. It also emits $R$ read strengths $\beta^{\r{r},i}_t$ with $i=1,\ldots,R$, and a single write strength $\beta^{\r{w}}_t$. You may venture that the read and write heads compute the read and write weightings directly by using $\c{C}$, that is,

$$
\begin{eqnarray}
\clock \b{w}^{\r{r},i}_t & = & \c{C}\left(\b{M}_t,\b{k}^{\r{r},i}_t,\beta^{\r{r},i}_t\right) \quad i=1,\ldots,R \\
\clock \b{w}^\r{w}_t & = & \c{C}\left(\b{M}_t,\b{k}^{\r{w}}_t,\beta^{\r{w}}_t\right)
\end{eqnarray}
$$

The hourglass icon ⏳ is used throughout this guide to mark equations that are temporarily introduced for educational purposes but that are not part of the DNC model as presented in the Nature article. The previous two equations are an example of this, since, as will be seen later, the DNC computes the weightings in a more elaborate manner. Using only content-based addressing would considerably restrict the possibilities of the write and read heads to access the memory. For example, the task of copying an input vector sequence in such a way that the DNC repeats it completely once the input sequence has finished requires some kind of *incremental* addressing (i.e., reading at the next time step the vector that was written immediately after the last one read) that cannot be satisfied by means of content-based addressing. The heads consequently combine content-based addressing with more sophisticated schemes. In particular, content-based addressing is combined with <span class="term" markdown="1">dynamic memory allocation</span> when writing and with <span class="term" markdown="1">temporal memory linkage</span> when reading. The following section will focus on the latter.

<div class="disg" markdown="1">
#### <span class="interface"></span> New interface parameters in this section

The controller emits at time step $t$, among others, the following vectors and scalars corresponding to the discussion in this section:

$$
\b{k}^{\r{r},1}_{t},\ldots,\b{k}^{\r{r},R}_{t}, \hat{\beta}^{\r{r},1}_t,\ldots,\hat{\beta}^{\r{r},R}_t, \b{k}^{\r{w}}_t, \hat{\beta}^{\r{w}}_t
$$

The vectors $\b{k}$ are used unaltered as the lookup vectors for the read and write heads. The scalars $\hat{\beta}$ are passed through a [oneplus](https://www.wolframalpha.com/input/?i=plot+1+%2B+log+(1+%2B+e**x)+from+-5+to+5) function before obtaining the corresponding strengths in order to ensure that the final values of the strengths lie in the domain $[1,\infty)$:

$$
\begin{eqnarray}
\beta^{\r{r},i}_t &=& \text{oneplus} \left( \hat{\beta}^{\r{r},i}_t \right) \quad i=1,\ldots,R \\
\beta^{\r{w}}_t &=& \text{oneplus} \left( \hat{\beta}^{\r{w}}_t \right)
\end{eqnarray}
$$

with

$$
\text{oneplus}(x) = 1 + \log ( 1 + e^x)
$$

</div>


## Temporal memory linkage

This kind of memory reading scheme is mainly based on a <span class="term" markdown="1">temporal link matrix</span> $\b{L}_t$ that keeps track of the order in which locations have been written. This matrix acts like a chronicler, saving the history of memory writes in the mathematical equivalent of a chronicle as follows: *"In the beginning, memory location 2 was written. Then, location 4 was written after location 2. Then, location 1 was written after location 4."*  The temporal link matrix $\b{L}_t$ is an $N \times N$ matrix in which the element $\b{L}_t[i,j]$ indicates whether memory location $i$ was written after location $j$.

<span class="dual"></span> In our simplistic dualistic view, the chronicle above will be represented (assuming a memory with capacity $N=4$) as:

$$
\b{L}_t =
\begin{bmatrix}
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & \co{1} & 0 & 0
\end{bmatrix}
\label{lmatrix}
$$

The $1$ in the colored element $\b{L}_t[4,2]$ represents the sentence *"Then, location 4 was written after location 2"* in the narrative shown previously. Note that row $i$ holds *backward* information (namely, what was written *before* writing to location $i$; for example, the fourth row in the previous matrix lets us know that location $2$ was written before location $4$), whereas column $j$ contains *forward* information (namely, what was written after writing to location $j$; for example, the second column in the previous matrix lets us know that the write head moved to location $4$ after writing to location $2$). The fact that the second column has a non-zero element but the second row is made up of zeros indicates that memory location $2$ was the first to be written; the fact that both the third row and the third column are zero reveals that location $3$ has not yet been written.

Bearing this in mind, given a generic weighting $\b{w}_t$ it can easily be deduced how we can move backward or forward in time in order to shift attention to those locations written *before* or *after* those represented by $\b{w}_t$; the resulting locations will be respectively represented by the
<span class="term" markdown="1">backward weighting</span> $\b{b}_t$ and the <span class="term" markdown="1">forward weighting</span> $\b{f}_t$, which are computed as:

$$
\begin{eqnarray}
\clock \b{b}_t & = & \b{L}^\top_t \b{w}_t \\
\clock \b{f}_t & = & \b{L}_t \b{w}_t
\end{eqnarray}
$$

For example, given the previous temporal link matrix $\b{L}_t$ in \eqref{lmatrix} and a dualistic weighting $\b{w}_t = [0;0;0;1]$ representing location 4, the backward weighting would be:

$$
\b{b}_t =
\begin{bmatrix}
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}^\top \b{w}_t =
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
0 \\
0 \\
0 \\
1
\end{bmatrix} =
\begin{bmatrix}
0 \\
1 \\
0 \\
0
\end{bmatrix}
$$

which indicates that the location written before location 4 is location 2. The forward weighting would be analogously obtained as:

$$
\b{f}_t =
\begin{bmatrix}
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix} \, \b{w}_t =
\begin{bmatrix}
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
0 \\
0 \\
0 \\
1
\end{bmatrix} =
\begin{bmatrix}
1 \\
0 \\
0 \\
0
\end{bmatrix}

$$

which indicates that the location written after location 4 is location 1. Note again that although the previous example belongs to the simplistic dualistic world, DNCs deal with temporal link matrices and weightings that are pluralistic by nature. As a result of this, the element $\b{L}_t[i,j]$ in the temporal link matrix actually indicates *to what degree* memory location $i$ was written after location $j$; the backward and forward weightings also focus their attention on every location in a non-dualistic manner.

In the special case of attempting to obtain the backward weighting for the second memory location ($\b{w}_t = [0;1;0;0]$), which is the first location written according to $\b{L}_t$, we will obtain:

$$
\b{b}_t =
\begin{bmatrix}
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}^\top \b{w}_t =
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0
\end{bmatrix} \begin{bmatrix}
0 \\
1 \\
0 \\
0
\end{bmatrix} =
\begin{bmatrix}
0 \\
0 \\
0 \\
0
\end{bmatrix}
$$

As stated previously, temporal memory linkage is an addressing mode that is intended for reading, but is based on information regarding writes that is provided by the chronicler matrix $\b{L}\_t$ because the motivation here is to allow memory locations to be read in the same order as they were written (or in the reverse order, depending on the task). With this addressing scheme, and given $\b{L}\_t$ and the locations $\b{w}^{\r{r},i}\_{t-1}$ that were read at time $t-1$ by the $i$-th read head, the locations that the $i$-th read head should consequently pay attention to at time $t$ if the original write order is intended to be respected is represented by the forward weighting $\b{f}^i_t$, and the locations to be read by the $i$-th head at time $t$ if the reverse of the original write order needs to be followed is represented by the backward weighting $\b{b}^i_t$, both of which are computed as:

$$
\begin{eqnarray}
\b{b}^i_t & = & \b{L}^\top_t \b{w}^{\r{r},i}_{t-1} \\
\b{f}^i_t & = & \b{L}_t \b{w}^{\r{r},i}_{t-1}
\end{eqnarray}
$$

The temporal memory linkage addressing mode, therefore, provides two (forward and backward) means to determine the next location to be read by a read head. We have already studied a third mode, namely, <a href="content-based-addressing">content-based addressing</a>, which allows each read head to compute a <span class="term" markdown="1">content weighting</span> $\b{c}^{\r{r},i}_t \in \c{S}_N$ (the paper wrongly indicates $\Delta_N$ in the "Read weighting" section on page 478) as follows:

$$
\b{c}^{\r{r},i}_t = \c{C}(\b{M}_t,\b{k}^{\r{r},i}_t,\beta^{\r{r},i}_t) \quad i=1,\ldots,R
$$

where the read (lookup) keys $\b{k}^{\r{r},i}_t$ and the key strengths $\beta^{\r{r},i}_t$ are parameters emitted by the controller, as discussed previously.

How are these three addressing modes combined in order to determine the final weighting $\b{w}^{\r{r},i}_{t}$ to be used by the $i$-th read head? The three modes are interpolated using three scalar coefficients $\b{\pi}^i_t[1]$, $\b{\pi}^i_t[2]$ and $\b{\pi}^i_t[3]$ that are emitted by the controller:

$$
\b{w}^{\r{r},i}_{t} = \b{\pi}^i_t[1] \b{b}^i_t + \b{\pi}^i_t[2] \b{c}^{\r{r},i}_t + \b{\pi}^i_t[3] \b{f}^i_t
$$

A <span class="term" markdown="1">read mode vector</span> $\b{\pi}^i_t \in \c{S}_3$ is emitted by the controller for each read head. <span class="dual"></span> In a dualistic world, only one of the three coefficients would be one and the other two would be zero, but DNCs are pluralistic and much more flexible, and may actually combine different addressing schemes in a single read operation, the resulting weighting being a mixture of the contents of many different locations referred to via different addressing modes.

As with the other weightings, $\b{b}^i_t \in \Delta_N$ and $\b{f}^i_t \in \Delta_N$, proof of which is omitted in this guide. If, however, proof is required, it is necessary to know that every row and column in $\b{L}_t$ is a weighting that belongs to $\Delta_N$, that is, $\b{L}_t[i,\cdot] \in \Delta_N$ and $\b{L}_t[\cdot,j] \in \Delta_N$ for all $i$, $j$ and $t$. This will be clearer after reading the next section.

In the preceding discussion I have intentionally circumvented how the link matrix $\b{L}_t$ is computed. Recall that, although it is used to eventually compute the read weightings, this matrix contains information regarding writes. Details on how this is achieved are provided in the following section.

<div class="disg" markdown="1">
#### <span class="interface"></span> New interface parameters in this section

The controller emits a different vector $\hat{\b{\pi}}^i_t$ with $i=1,\ldots,R$ for each read head; in order to ensure that each of these vectors
belongs to $\c{S}_3$, the [softmax](https://en.wikipedia.org/wiki/Softmax_function) function is applied to obtain the $R$ read mode vectors $\b{\pi}^i_t$ with $i=1,\ldots,R$ in a such a way that the three elements of each vector add up to $1$ and are non-negative:

$$
\b{\pi}^i_t = \mathrm{softmax} \left( \hat{\b{\pi}}^i_t \right) \quad i=1,\ldots,R
$$

Each element in $\b{\pi}^i_t$ is, therefore, obtained as:

$$
\b{\pi}^i_t[j] = \frac{\r{exp}\left(\hat{\b{\pi}}^i_t[j]\right)}{\sum_{k=1}^3 \r{exp}\left(\displaystyle\, \hat{\b{\pi}}^i_t[k]\right)} \quad i=1,\ldots,R \quad j=1,2,3
$$

</div>



## Computation of the temporal link matrix

As stated in the previous section, $\b{L}_t \in [0,1]^{N \times N}$ is an $N \times N$ matrix in which the element $\b{L}_t[i,j]$ indicates to what degree memory location $i$ was written after location $j$ after time step $t$ and before time $t+1$. <span class="dual"></span> Let me begin by describing how things would work in the dualistic view when a single location is written at each time step, and let me also assume the (wrong) hypothesis that a write operation is always performed at each time step. With these simplifications $\b{L}_t$ would be a matrix of zeros and ones in which if $i$ is the last written location (at time $t$) and $j$ is the second-to-last written location (at time $t-1$), then $\b{L}_t[i,j]=1$ ; if neither $i$ has been written at time step $t$ nor $j$ has been written at time step $t-1$, then $\b{L}_t[i,j]$ remains unchanged with respect to its previous value at time step $t-1$; in the other cases (namely, either $i$ has been written at time $t$ or $j$ has been written at $t-1$ but both are not true), then $\b{L}_t[i,j]=0$ (recall that we are assuming that a write operation is always performed at each step), reflecting the fact that $i$ has not been written after writing to $j$:

$$
\clock
\b{L}_t[i,j] =
\begin{cases}
1 & \text{if } \b{w}^\r{w}_t[i]=1 \, \text{ and } \, \b{w}^\r{w}_{t-1}[j]=1 \\
0 & \text{if } \b{w}^\r{w}_t[i] \neq \b{w}^\r{w}_{t-1}[j] \\
\b{L}_{t-1}[i,j] & \text{otherwise}
\label{Lsimp}
\end{cases}
$$

Also, $\b{L}_0[i,j]=0$ for all $i$ and $j$.

Apart from the usual over-simplification that is intrinsic to the dualistic view, the previous equation ignores the fact that there may be an arbitrary number of time steps between one write and the next. In order to overcome these limitations, we need a more precise means to record the degree to which a memory location has been *recently* written. This objective is attained by first introducing a new weighting $\b{p}_t \in \Delta_N$, which is called the <span class="term" markdown="1">precedence weighting</span>. The element $\b{p}_t[i]$ denotes the degree to which $i$ was the last location written and is computed recursively; a high value of $\b{p}_t[i]$ may indicate that $i$ has been written at time step $t$ with *great dedication* (that is, the other locations have not been written at all at time step $t$ or they have been written to a much lesser extent), but it may also indicate that location $i$ was written with great dedication at $t'<t$ but that no significant writes have been carried out in the memory since $t'$; it may also indicate that writing attention has been partially paid to $i$ at different recent time steps in the past and the total cumulative attention degree is greater than the degrees of attention paid to the other locations. <span class="dual"></span> A preliminary dualistic formulation of the precedence weighting would be:

$$
\clock
\b{p}_t =
\begin{cases}
\b{w}^\r{w}_t & \text{if $\sum_{i=1}^N \b{w}^\r{w}_t[i]=1$} \\[1.5ex]
\b{p}_{t-1} & \text{otherwise}
\end{cases}
\label{preced}
$$

Note that $\b{p}_t$ is only updated when a write operation has been performed at time step $t$, and that it could consequently remain untouched for long read-only periods. To return to the pluralistic DNC reality, even a single read-only time step is notoriously difficult, as the write weighting $\b{w}^\r{w}_t$ will hardly be zero; we must, therefore, raise our sights and embrace this fact by introducing a continuous formulation of the precedence weighting as follows:

$$
\b{p}_t = \left( 1 - \sum_{i=1}^N \b{w}^\r{w}_t[i] \right) \b{p}_{t-1} + \b{w}^\r{w}_t
$$

Note that the previous equation would degenerate to \eqref{preced} in the limiting cases of a *full-write* or read-only time steps. In the first case, the summation of all the components of
$\b{w}^\r{w}_t$ would have its greatest value (i.e., $1$), and $\b{p}_t$ would be reduced to $\b{w}^\r{w}_t$; in the second case, the summation of all the components of $\b{w}^\r{w}_t$ would be $0$, $\b{w}^\r{w}_t$ would also be $\b{0}$, and $\b{p}_t$ would then be a copy of $\b{p}\_{t-1}$. Note that after a hypothetical full write, $\b{p}_t$ would reset and the past history (encoded in $\b{p}\_{t-1}$) completely forgotten. The precedence weighting is initialized to $\b{p}_0 = \b{0}$.

Bearing all this in mind, we are finally ready to set out a differentiable final formulation of $\b{L}_t[i,j]$ which surpasses the dualistic constraints of \eqref{Lsimp}. The equation must reflect the fact that if the memory location $i$ is significantly written at time step $t$ (i.e., $\b{w}^\r{w}_t[i]$ is high), then $\b{L}_t[i,j]$ must be given a value that is mostly proportional to the degree to which the memory location $j$ was the last location written before that (this information is represented by $\b{p}\_{t-1}[j]$); one possibility for this is the product $\b{w}^\r{w}_t[i] p\_{t-1}[j]$. If $\b{w}^\r{w}_t[i]$ is low (or, equivalently, $1-\b{w}^\r{w}_t[i]$ is high), then $\b{L}_t[i,j]$ will mostly maintain the value of $\b{L}\_{t-1}[i,j]$; there is, however, an exception to the latter rule: if location $j$ has been significantly written at time step $t$ (i.e., $\b{w}^\r{w}_t[j]$ is high), then this situation has to trigger a non-dualistic *reset* of $\b{L}_t[i,j]$ to a low value, as the memory location that will be written after $j$ will not be known until time step $t+1$. The resulting differentiable equation that integrates all these aspects is:

$$
\b{L}_t[i,j] = \left( 1 - \b{w}^\r{w}_t[i] - \b{w}^\r{w}_t[j] \right) \b{L}_{t-1}[i,j] + \b{w}^\r{w}_t[i] p_{t-1}[j]
$$

Note that as $\b{w}^\r{w}_t \in \Delta_N$, it is not possible for the subtractions in parenthesis to provide a negative value. In order to complete the specification of $\b{L}_t[i,j]$, the two following equations are required:

$$
\begin{eqnarray}
\b{L}_0[i,j] & = & 0 \\
\b{L}_t[i,i] & = & 0
\end{eqnarray}
$$

This concludes the description of how the read weightings are computed as a combination of content-based addressing and temporal memory linkage. In the paper published in Nature there is a subsection entitled *Sparse link matrix* on page 478 which contains additional details on how to efficiently store and compute the link matrix $\b{L}_t$, but as these are implementation notes they will not be commented on in this guide. Let us now explore how the write weightings are computed.

## Dynamic memory allocation

As already stated, content-based addressing is combined with dynamic memory allocation in order to obtain the write weighting $\b{w}^\r{w}_t$. Dynamic memory allocation in DNC represents the pluralistic equivalent of the memory allocation schemes in many programming languages.

The objective of the dynamic memory allocation is to make DNCs compute a new kind of weighting at every time step, the <span class="term" markdown="1">allocation weighting</span>, which is a vector $\b{a}_t \in \Delta_N$ that indicates *to what degree* each memory location is allocable (that is, not *write protected*). The fact that $\b{a}_t \in \Delta_N$ implies that the dualistic metaphor cannot be generally followed here. The allocation weighting corresponding to $N=4$ equally allocable memory locations would be, for example, $\b{a}_t = [0.1;0.1;0.1;0.1]$ rather than $[1;1;1;1]$; the fact that the second memory location is completely allocable but no other location can be reserved would be represented as $\b{a}_t = [0;1;0;0]$; the allocation weighting $\b{a}_t = [0.4;0.2;0;0]$ represents the situation in which the first location is more allocable than the second. If $\b{a}_t=\b{0}$, then the DNC has run out of free memory locations and no location can consequently be accessed for writing by means of dynamic memory allocation at time $t$; it is, however, important to note that it might still be possible to write to a location accessed by means of content-based addressing.

The details regarding the computation of the allocation weighting vector $\b{a}_t$ will be provided in the next section. We shall, however, first see how it is used in order to determine the write weighting.

In the same way that each read head computes a <span class="term" markdown="1">read content weighting</span> $\b{c}^{\r{r},i}_t \in \Delta_N$ (see <a href="#temporal-memory-linkage">above</a>), the write head computes a <span class="term" markdown="1">write content weighting</span> $\b{c}^\r{w}_t \in \Delta_N$ as follows:

$$
\b{c}^\r{w}_t = \c{C}(\b{M}_{t-1},\b{k}^\r{w}_t,\beta^\r{w}_t)
$$

where the write (lookup) key $\b{k}^\r{w}_t$ and the write key strength $\beta^\r{w}_t$ are parameters emitted by the controller. Observe the subindex of the memory matrix in the previous equation: $\b{c}^\r{w}\_t$ is used to obtain $\b{M}\_t$ and is, therefore, computed from $\b{M}\_{t-1}$.

It should be clear at this stage that there are three possible paths that can be followed at time $t$ regarding a write operation (and these will affect how the write weighting $\b{w}^\r{w}_t$ is obtained):

1. writing (via dynamic memory allocation) into the locations specified on the allocation weighting $\b{a}_t$;
1. writing (via content-based addressing) into the locations specified on the write content weighting $\b{c}^\r{w}_t$;
1. not writing at all at this time step.

In line with the non-dualistic (pluralistic) nature of DNCs, a differentiable combination of the previous three schemes is used. In order to decide between the first two options, the DNC controller emits a scalar <span class="term" markdown="1">allocation gate</span> $g^\r{a}_t \in [0,1]$ that governs the interpolation between $\b{a}_t$ and $\b{c}^\r{w}_t$ (the first two options), and a scalar <span class="term" markdown="1">write gate</span> $g^\r{w}_t \in [0,1]$ that determines to what degree the memory is or is not written at this time step (the third option). The resulting write weighting is therefore computed as:

$$
\b{w}^\r{w}_t = g^\r{w}_t \left[ g^\r{a}_t \, \b{a}_t + \left( 1 - g^\r{a}_t \right) \b{c}^\r{w}_t \right]
$$

In the limiting case in which $g^\r{w}_t=0$, $\b{w}^\r{w}_t$ would be $\b{0}$ meaning that no write operation would be performed at this time step; $g^\r{w}_t=1$ and $g^\r{a}_t=1$ correspond to dynamic memory addressing only (note that $\b{w}^\r{w}_t$ may still be $\b{0}$ if $\b{a}_t$ is also $\b{0}$); finally, $g^\r{w}_t=1$ and $g^\r{a}_t=0$ corresponds to content-base addressing only. The activation functions will not in fact produce these limiting values, thus resulting in gates that will be partially closed (or, equivalently, partially open).

<div class="disg" markdown="1">
#### <span class="interface"></span> New interface parameters in this section

The controller emits the scalars $\hat{g}^\r{a}_t$ and $\hat{g}^\r{w}_t$. In order to constrain them in the domain $[0,1]$ and to obtain the allocation and write gates, respectively, the [logistic sigmoid](https://www.wolframalpha.com/input/?i=plot+1%2F(1%2Be**-x)+from+-5+to+5) function $\sigma$ is used:

$$
\begin{eqnarray}
g^\r{a}_t & = & \sigma\left(\hat{g}^\r{a}_t\right) \\
g^\r{w}_t & = & \sigma\left(\hat{g}^\r{w}_t\right)
\end{eqnarray}
$$

</div>


## Computation of the allocation weighting vector

As stated in the previous section, the allocation weighting $\b{a}_t$ indicates *to what degree* each memory location is allocable. As also stated, DNCs do not exist in a dualistic world, and the memory allocation scheme consequently deviates considerably from that found in many computer systems in which allocability is a binary property of each location (a location can or cannot be allocated). In order to compute $\b{a}_t$, a new vector $\b{u}_t \in [0,1]^N$, the <span class="term" markdown="1">usage vector</span>, will first be introduced. Note that in this case the elements of $\b{u}_t$ may add up to much more than one (the maximum is $N$). As will be seen later, the formula used to obtain $\b{a}_t$ will then give significantly higher weights to those locations that appear closer to the head of the list of locations sorted in ascending order with respect to $\b{u}_t$. If all usages $\b{u}_t[i]$ are $1$, then no memory can be allocated until some locations have been freed and $\b{a}_t$ will be $\b{0}$. The usage vector $\b{u}_t$ is initialized to $\b{u}_0=\b{0}$.

The main idea behind the computation of $\b{u}_t[i]$ is that of taking into account that different *forces* will act in opposite directions in order to determine whether the usage of location $i$ has to be increased at the current time step (to a maximum of $1$, which indicates that $i$ is *completely* in use and cannot be allocated at all) or decreased (to a minimum of $0$, which indicates that $i$ is completely allocable). On the one hand, the decremental force acts when the location $i$ has been read at time step $t-1$; this may be a good sign that the contents of location $i$ are no longer required (more details on this subject are provided below). On the other hand, the incremental force acts when location $i$ was written at time step $t-1$, which clearly indicates that the usage of $i$ needs to be strengthened as the value stored at location $i$ has not had yet the chance to be read.

The decremental force is represented by the <span class="term" markdown="1">retention vector</span> $\b{\psi}_t \in [0,1]^N$, which indicates to what degree each of the memory locations need to be retained at time step $t$ based on the information provided by the read operations at time step $t-1$:

- A high value in the retention vector for location $i$ (when $\b{\psi}_t[i]$ is close to $1$) indicates that $i$ has to be retained (i.e., not freed) at time step $t$, because it was not read at time $t-1$.
- A low value in the retention vector for location $i$ (when $\b{\psi}_t[i]$ is close to $0$) indicates that $i$ is available at time step $t$, because it was read at time $t-1$.

Bearing all this in mind, the usage vector $\b{u}_t$ may be, in principle, formulated in a differentiable incremental manner as:

$$
\clock \b{u}_t[i] = \left( \b{u}_{t-1}[i] + \b{w}^\r{w}_{t-1}[i] \right) \b{\psi}_t[i]
\label{firstu}
$$

Note how the two opposite forces act: $\b{w}^\r{w}_{t-1}[i]$ increases the previous usage score, but $\b{\psi}_t[i] \in [0,1]$ then decreases the result. The original paper adds a third term between the parentheses in the previous equation. I must admit that I am not completely certain about its purpose but my guess is that its mission is to cut off the result in order to ensure that $\b{u}_t[i] \in [0,1]^N$. The final equation that refines \eqref{firstu} is then:

$$
\b{u}_t[i] = \left( \b{u}_{t-1}[i] + \b{w}^\r{w}_{t-1}[i] - \b{u}_{t-1}[i] \b{w}^\r{w}_{t-1}[i] \right) \b{\psi}_t[i]
\label{secondu}
$$

Note that if $\b{u}\_{t-1}[i] \approx 1$, $\b{w}^\r{w}\_{t-1} \approx 1$ and $\b{\psi}_t[i] \approx 1$, then $\b{u}_t[i]$ could easily be close to $2$ in \eqref{firstu}, but it will not be greater than $1$ in \eqref{secondu}. Note also that since none of the vector elements in \eqref{secondu} can be negative (they all belong to the range $[0,1]$), $\b{u}_t[i]$ cannot be negative either.

Equation \eqref{secondu} can also be equivalently expressed in vector form as:

$$
\b{u}_t = \left( \b{u}_{t-1} + \b{w}^\r{w}_{t-1} - \b{u}_{t-1} \circ \b{w}^\r{w}_{t-1} \right) \circ \b{\psi}_t
$$

where $\circ$ denotes the [element-wise product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) (or Hadamard product) of two vectors.:

$$
\left( \b{x} \circ \b{y} \right)[i] = \b{x}[i] \cdot \b{y}[i]
$$

We have yet to tackle two points. First, the formulation of the retention vector $\b{\psi}_t$, and second, the actual computation of the allocation weighting $\b{a}_t$. With regard to the retention vector $\b{\psi}_t \in [0,1]^N$, a preliminary  proposal for it would be:

$$
\clock \b{\psi}_t[i] = \prod_{j=1}^R \left( 1 - \b{w}^{\r{r},j}_{t-1}[i] \right)
$$

<span class="dual"></span> With the dualistic simplification, the read weight $\b{w}^{\r{r},j}\_{t-1}[i]$ would be $0$ or $1$; once one of the read weights for location $i$ was $1$ (indicating that the corresponding read head had performed a read operation on $i$ at time step $t-1$), the resulting $\b{\psi}_t[i]$ would be $0$; in other words, if $i$ was read by at least one read head at time $t-1$, then $i$ could be freed (not retained) at $t$. Real-word DNCs do not work under these extreme dualistic conditions and $\b{\psi}_t[i]$ will be higher or lower depending on the degree to which the previous read operations paid attention to $i$.

Some readers may have noticed a limitation in the equation just formulated for the retention vector $\b{\psi}_t \in [0,1]^N$: if locations are freed immediately after a read operation is performed on them, then it might not be possible to read a particular value stored at a memory location more than once so that it is *reused* at different future time steps during the execution of the neural network. This would be a frustrating restriction that could limit the computational power of DNCs and leave many applications out of reach. In order to overcome this limitation, the controller emits a scalar <span class="term" markdown="1">free gate</span> $f_t \in [0,1]$ to guarantee the possibility of the retention of a memory location even after a read operation. The resulting equation is:

$$
\b{\psi}_t[i] = \prod_{j=1}^R \left( 1 - f_t^j \b{w}^{\r{r},j}_{t-1}[i] \right)
$$

As is observed, there is a different free gate $f_t^j$ for each read head. If $f_t^j \approx 0$ for all read heads, then $\b{\psi}_t[i] \approx 1$ thus indicating that the location $i$ cannot be freed at time $t$ independently of whether or not $i$ was read at time step $t-1$. The last equation in vector form is:

$$
\b{\psi}_t = \prod_{i=1}^R \left( \b{1} - f_t^i \b{w}^{\r{r},i}_{t-1} \right)
$$

where the product of vectors represents the element-wise product as before.

Finally, to conclude this section, let me discuss how the allocation weighting $\b{a}_t \in \Delta_N$ is computed. As already stated, higher weights will be given in $\b{a}_t$ to those locations that are closer to the head of the list of locations sorted with respect to $\b{u}_t$ in ascending order. The first step is, therefore, to obtain a <span class="term" markdown="1">free list</span> $\b{\phi}_t \in \N^N$ made up of the indices of the memory locations (in the range of $1,\ldots,N$) in ascending order of usage at time $t$ as defined by $\b{u}_t$:

$$
\b{\phi}_t = \text{SortIndicesAscending}(\b{u}_t)
$$

For example, given the usage vector $\b{u}_t=[1;0;0.8;0.4]$, the resulting free list would be $\b{\phi}_t=[2;4;3;1]$; the least used memory location would be $\b{\phi}_t[1]$ (location $2$ in the example, with usage $\b{u}_t[\b{\phi}_t[1]]=0$) and the most used memory location would be $\b{\phi}_t[N]$ (location $1$ in the example, with usage $\b{u}_t[\b{\phi}_t[4]]=1$).

Given $\b{\phi}_t$ and $\b{u}_t$ there are many different ways in which to obtain an allocation weighting $\b{a}_t$ that complies with the restrictions that a memory location stored at the $k$-th element of $\b{\phi}_t$ receives an allocation weight greater or equal than that received by the memory location stored at the $(k+1)$-th element of $\b{\phi}_t$, while the allocation weighting $\b{a}_t[i]$ is simultaneously $0$ for those locations with $\b{u}_t[i]=1$. One option would be:

$$
\clock
\b{a}_t[j] =
\begin{cases}
0 & \text{if $\b{u}_t[j]=1$} \\
1/\gamma & \text{otherwise}
\end{cases}
\qquad j=1,\ldots,N
\label{myat}
$$

$\gamma$ is the value that satisfies $ \b{\phi}_t[\gamma] = j$. If the vector $\b{u}_t=[1;0;0.8;0.4]$ is again employed, the resulting allocation weighting obtained from equation \eqref{myat} would be $\b{a}_t=[0;1;1/3;0.5]$:

| $j$ | $\b{u}_t[j]$ | $\gamma : \b{\phi}_t[\gamma] = j$ |  $\b{a}_t[j]$ |
|---|---|---|---|
| $1$ | $1$ | $4$ | $0$ |
| $2$ | $0$ | $1$ | $1/1$ |
| $3$ | $0.8$ | $3$ | $1/3$ |
| $4$ | $0.4$ | $2$ | $1/2$ |

The equation above, however, discards important information regarding the degree of usage contained in $\b{u}_t$; moreover, it does not guarantee that $\b{a}_t \in \Delta_N$. DNCs, in fact, use a different differentiable approach:

$$
\b{a}_t[j] = \left( 1 - \b{u}_t[j] \right) \prod_{i=1}^{\gamma -1} \b{u}_t[\b{\phi}_t[i]]
\label{atmod}
$$

in which $\gamma$ is again the value that satisfies $ \b{\phi}_t[\gamma] = j$. For the running example, this equation will give $\b{a}_t[1] = 0$, $\b{a}_t[2] = 1$, $\b{a}_t[3] = 0$, $\b{a}_t[4] = 0$, that is, $\b{a}_t=[0;1;0;0]$.

| $j$ | $\b{u}_t[j]$ | $\gamma : \b{\phi}_t[\gamma] = j$ | $1-\b{u}_t[j]$ | $\prod_{i=1}^{\gamma-1} \b{u}_t[\b{\phi}_t[i]]$ | $\b{a}_t[j]$ |
|---|---|---|---|---|---|
| $1$ | $1$ | $4$ | $0$ | $0 \times 0.4 \times 0.8 = 0$ | $0$ |
| $2$ | $0$ | $1$ | $1$ | $1$ | $1$ |
| $3$ | $0.8$ | $3$ | $0.2$ | $0 \times 0.4 = 0$ | $0$ |
| $4$ | $0.4$ | $2$ | $0.6$ | $0$ | $0$ |

Note that for those locations with full occupation (i.e., $\b{u}_t[\b{\phi}_t[j]] \approx 1$), the subtraction is approximately $0$ and it is, therefore, $\b{a}_t[\b{\phi}_t[j]]$. Note also that if at least one usage value of $0$ exists, then the first location in the list $\b{\phi}_t$ will be assigned a one in the allocation weighting and the remaining locations will be assigned a zero. No proof that the resulting allocation vector always belongs to $\Delta_N$ will be provided here.

A further example. Note that the companion notebook allows you to check [how these tables are obtained](/funicular/dnc-notebook.html#Computation-of-the-allocation-weighting-vector) and get a table with your own data. In the case of the usage vector $\b{u}_t=[0.4;0.6;0.2;0.5]$, the resulting allocation weighting is $\b{a}_t=[0.12;0.016;0.8;0.04]$:

| $j$ | $\b{u}_t[j]$ | $\gamma : \b{\phi}_t[\gamma] = j$ | $1-\b{u}_t[j]$ | $\prod_{i=1}^{\gamma-1} \b{u}_t[\b{\phi}_t[i]]$ | $\b{a}_t[j]$ |
|---|---|---|---|---|---|
| $1$ | $0.4$ | $2$ | $0.6$ | $0.2$ | $0.12$ |
| $2$ | $0.6$ | $4$ | $0.4$ | $0.2 \times 0.4 \times 0.5 = 0.04$ | $0.016$ |
| $3$ | $0.2$ | $1$ | $0.8$ | $1$ | $0.8$ |
| $4$ | $0.5$ | $3$ | $0.5$ | $0.2 \times 0.4 = 0.08$ | $0.04$ |

The original paper presents \eqref{atmod} in a slightly different, yet equivalent form, which avoids the need to define $\gamma$:

$$
\b{a}_t[\b{\phi}_t[j]] = \left( 1 - \b{u}_t[\b{\phi}_t[j]] \right) \prod_{i=1}^{j-1} \b{u}_t[\b{\phi}_t[i]]
$$

Sorting is not a common operation in neural networks. It may even lead to some issues in the calculation of the gradient. But according to the authors of the paper, these issues are experimentally overlookable:

> "The sort operation induces discontinuities at the points at which the sort order changes. We ignore these discontinuities when calculating the gradient, as they do not seem to be relevant to learning"

<div class="disg" markdown="1">
#### <span class="interface"></span> New interface parameters in this section

The controller emits $R$ scalars $\hat{f}_t^i$ with $i=1,\ldots,R$; in order to constrain them in the domain $[0,1]$ and obtain the free gates, the [logistic sigmoid](https://www.wolframalpha.com/input/?i=plot+1%2F(1%2Be**-x)+from+-5+to+5) function $\sigma$ is used:

$$
f_t^i = \sigma \left( \hat{f}_t^i \right) \quad i=1,\ldots,R
$$

Upon integrating all the interface parameters introduced throughout this guide, we obtain the definition of the interface vector $\b{\xi}_t$:

$$
\b{\xi}_t = \left[ \b{k}^{\r{r},1}_{t},\ldots,\b{k}^{\r{r},R}_{t}, \hat{\beta}^{\r{r},1}_t,\ldots,\hat{\beta}^{\r{r},R}_t, \b{k}^{\r{w}}_t, \hat{\beta}^{\r{w}}_t, \hat{\b{e}}_t, \b{v}_t,
\hat{f}_t^1,\ldots,\hat{f}_t^R,
\hat{g}^\r{a}_t, \hat{g}^\r{w}_t,
\hat{\b{\pi}}^1_t,\ldots,\hat{\b{\pi}}^R_t  \right]
$$

</div>


## Conclusions

Memory-augmented neural networks, such as DNCs, lead to interesting challenges in the interplay of machine learning, reasoning and algorithm inference. The purpose of this guide is not to point out these opportunities, but to simply present the motivations behind the equations that define how DNCs work in an pedagogical manner. I certainly hope that the objective has been achieved. Finally, it is interesting to note that many of these equations admit alternative formulations, which may pave the way for further curiosity, reflection and research.

<span id="cite">**How to cite this work:**</span> Juan Antonio Pérez-Ortiz, "{{page.title}}". First published on October 20 2017. Last modified on October 20 2017. {{site.url}}, accessed on {{ 'now' | date: "%B %d %Y" }} [a [bibtex]({{ "/assets/dnc.bib" | prepend: site.baseurl }}) file is also available]

<span id="about">**About the author:**</span> I work as an associate professor and researcher at [Universidad de Alicante](https://www.ua.es/) in Spain. I received my Ph.D. in computer science in 2002 with a thesis on recurrent neural models for sequence processing that used, among others, [distributional](http://www.dlsi.ua.es/~japerez/pub/pdf/ijcnn2001.pdf) [representations](https://doi.org/10.1109/IJCNN.2001.938396) for natural language processing (as devised by my thesis supervisor, [Mikel L. Forcada](http://www.dlsi.ua.es/~mlf/)) and [LSTM](ftp://ftp.idsia.ch/pub/juergen/nnlstmkalman.pdf) [cells](https://doi.org/10.1016/S0893-6080(02)00219-8) (as devised by [Jürgen Schmidhuber](http://people.idsia.ch/~juergen/), who was my supervisor during my internship at the [IDSIA](http://www.idsia.ch/) research institute in 2000). I have also worked on machine translation and computer-assisted translation, especially as a member of the team involved in the development of the open-source machine translation platform [Apertium](http://www.apertium.org). I have recently started to research on the topic of neural machine translation. I currently teach undergraduate and postgraduate courses on translation technologies, programming and web development. A list of my publications is available at [my profile](https://scholar.google.com/citations?hl=en&user=_NEbOj4AAAAJ) at Google Scholar.

<span id="sources">**Sources:**</span> the source files of this document are available in a Github [repository](http://github.com).

<span id="license">**License:**</span> <span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Text" property="dct:title" rel="dct:type">«{{page.title}}»</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://www.dlsi.ua.es/~japerez/" property="cc:attributionName" rel="cc:attributionURL">Juan Antonio Pérez-Ortiz</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/80x15.png" /></a><br />

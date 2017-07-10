---
title: A guide to neural computers
author: Juan
layout: default
comments: true
use_math: true
---

# {{page.title}}

Written by **Juan Antonio**, researcher at Universidad de.

**Last update:** {{ 'now' | date: "%B %d %Y" }}.

**How to cite this work:** Juan, "{{page.title}}". First published on July 15 2017. Last modified  on {{ 'now' | date: "%B %d %Y" }}. {{site.url}}, accessed on {{ 'now' | date: "%B %d %Y" }} [replace access date with the actual one; [bibtex]({{ "/assets/bitbybitdnc.bib" | prepend: site.baseurl }}) file]

**Comments and suggestions:** feel free to send any comments or suggestions to enhance this guide through [this page](http://github.com); you need a Github account to write a comment, but you can read other's comments without it.

<div id="index" markdown="1">
**Table of contents:** &nbsp;
<a class="label label-blue1" href="#architecture">Architecture</a>
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


## Architecture

It reads an external input $\b{x}_t$ and produces a output $\b{y}_t$ which is intended to represent things such as the estimation of a class, the prediction of the next element in a sequence, etc. The mapping from inputs to outputs is computed as a set of differentiable operations which depend on a set of weights that are automatically learned to minimize a loss function that compares the outputs emitted by the model with the desirable outputs contained in a training set. Gradient descent can then be used to estimate those weights for a particular task.

$$
[\b{\nu}_t,\b{\xi}_t] = \c{N}\left(\b{x}_t,\b{r}^{1}_{t-1},\ldots,\b{r}^{R}_{t-1}\right) \R
$$


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

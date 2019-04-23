# Derivation of layer gradients

## General notation

* $\mathbf{z}$ 

| Symbol           | Meaning                                                      |
| ---------------- | ------------------------------------------------------------ |
| $\mathbf{z}^{l}$ | Hidden pre-activation at layer $l$.                          |
| $\mathbf{a}$^{l} | Activation at layer $l$, i.e. $\mathbf{a}^l = \varphi(\mathbf{z}^l)​$, typically element-wise. |
| $\mathbf{y}$     | Network output.                                              |
| $\mathbf{x}$     | Network input.                                               |

## Linear layer

Forward propagation
$$
\mathbf{z} = \mathbf{Wx} + \mathbf{b}
$$
where $\mathbf{x}\in \mathbb{R}^D​$, $\mathbf{z}\in\mathbb{R}^{H}​$,  $\mathbf{W}\in\mathbb{R}^{H\times D}​$ and $\mathbf{b}\in\mathbb{R}^{H}​$.

Gradients 
$$
\begin{align}
\frac{\partial\mathbf{z}}{\partial\mathbf{x}} &= \mathbf{W}^T\\
\frac{\partial\mathbf{z}}{\partial\mathbf{W}} &= \mathbf{x}\\
\frac{\partial\mathbf{z}}{\partial\mathbf{b}} &= \mathbf{I}
\end{align}
$$

Given $\delta_{out}=\frac{\partial L}{\partial\mathbf{z}}\in\mathbb{R}^{H}$ then
$$
\frac{\partial L}{\partial\mathbf{x}} = \delta_{out} \mathbf{W} \in\mathbb{R}^{D}
$$


## Bilinear layer

Forward propagation
$$
\mathbf{z} = \mathbf{x}_1^T\mathbf{Wx}_2 + \mathbf{b}
$$
where $\mathbf{x}_1\in \mathbb{R}^{D_1}$, $\mathbf{x}_2\in \mathbb{R}^{D_2}$, $\mathbf{z}\in\mathbb{R}^{H}$,  $\mathbf{W}\in\mathbb{R}^{H \times D_1 \times D_2}$ and $\mathbf{b}\in\mathbb{R}^{H}$.

Gradients
$$
\begin{align}
\frac{\partial\mathbf{z}}{\partial\mathbf{x}_1} &= \mathbf{Wx}_2 \in\mathbb{R}^{H\times D_1}\\
\frac{\partial\mathbf{z}}{\partial\mathbf{x}_2} &= \mathbf{W}^T\mathbf{x}_1\in\mathbb{R}^{H\times D_2}\\
\frac{\partial\mathbf{z}}{\partial\mathbf{W}} &= \mathbf{x}_1\mathbf{x}_2^T\in\mathbb{R}^{D_1\times D_2}\\
\frac{\partial\mathbf{z}}{\partial\mathbf{b}} &= \mathbf{I}
\end{align}
$$
Given $\delta_{out}=\frac{\partial L}{\partial\mathbf{z}}\in\mathbb{R}^{H}$ then
$$
\frac{\partial L}{\partial\mathbf{x}_1} = \delta_{out} \mathbf{Wx}_2 \in\mathbb{R}^{H \times (H \times D_1 \times D_2) \times D_2} = \mathbb{R}^{D_2}\\
\frac{\partial L}{\partial\mathbf{x}_2} = \delta_{out} \mathbf{W}^T\mathbf{x}_1 \in\mathbb{R}^{H \times (H \times D_2 \times D_1) \times D_1} = \mathbb{R}^{D_1}\\
$$

## Recurrent layer

Vanilla forward propagation
$$
\mathbf{h}^{\langle t \rangle} = \text{tanh}\left(\mathbf{W}_{hx}\mathbf{x}^{\langle t \rangle} + \mathbf{W}_{hh}\mathbf{h}^{\langle t-1 \rangle}\right), \quad \text{for } t\in[0,T]\\
$$
Gradients
$$
\begin{align}
\frac{\partial\mathbf{h}^{\langle t \rangle}}{\partial\mathbf{x}} &= \mathbf{W}^T\\
\frac{\partial\mathbf{z}}{\partial\mathbf{W}} &= \mathbf{x}\\
\frac{\partial\mathbf{z}}{\partial\mathbf{b}} &= \mathbf{I}
\end{align}
$$

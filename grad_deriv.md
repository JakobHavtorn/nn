# Derivation of layer gradients

## General notation

* $\mathbf{z}$ 

| Symbol           | Meaning                                                      |
| ---------------- | ------------------------------------------------------------ |
| $\mathbf{z}^{l}$ | Hidden pre-activation at layer $l$.                          |
| $\mathbf{a}$^{l} | Activation at layer $l$, i.e. $\mathbf{a}^l = \varphi(\mathbf{z}^l)$, typically element-wise. |
| $\mathbf{y}$     | Network output.                                              |
| $\mathbf{x}$     | Network input.                                               |

## Linear layer

Forward propagation
$$
\mathbf{z} = \mathbf{Wx} + \mathbf{b}\\
$$
Gradients 
$$
\begin{align}
\frac{\partial\mathbf{z}}{\partial\mathbf{x}} &= \mathbf{W}^T\\
\frac{\partial\mathbf{z}}{\partial\mathbf{W}} &= \mathbf{x}\\
\frac{\partial\mathbf{z}}{\partial\mathbf{b}} &= \mathbf{I}
\end{align}
$$

## Recurrent layer

Vanilla forward propagation
$$
\mathbf{h}^{\langle t \rangle} = \text{tanh}(\mathbf{W}_{hx}\mathbf{x}^{\langle t \rangle} + \mathbf{W}_{hh}\mathbf{h}^{\langle t-1 \rangle}), \quad \text{for } t\in[0,T]\\
$$
Gradients
$$
\begin{align}
\frac{\partial\mathbf{h}^{\langle t \rangle}}{\partial\mathbf{x}} &= \mathbf{W}^T\\
\frac{\partial\mathbf{z}}{\partial\mathbf{W}} &= \mathbf{x}\\
\frac{\partial\mathbf{z}}{\partial\mathbf{b}} &= \mathbf{I}
\end{align}
$$

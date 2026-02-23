我们这篇工作想做的事情的背景和方法如下
CRaFT: Constrained Representation and Fine-Tuning for OpenVLA

1. Background & Motivation

Vision-Language-Action (VLA) models (like OpenVLA) are pre-trained on diverse, large-scale datasets, acquiring generalized visual-semantic structures. However, when performing Supervised Fine-Tuning (SFT) on specific downstream tasks (e.g., LIBERO), the narrow action supervision forces the intermediate multimodal features to overfit the limited data. We term this Catastrophic Representation Drift. This unconstrained drift destroys the pre-trained priors, leading to a severe drop in cross-task generalization and physical robustness.

2. Core Methodology: CRaFT

To mitigate this drift, we propose CRaFT. It formulates downstream adaptation as an explicitly constrained optimization problem:


$$\min_{\theta} \mathcal{L}_{act}(\theta) \quad \text{s.t.} \quad \mathcal{L}_{ret}(\theta) \le \varepsilon$$


where $\varepsilon$ is the maximum allowable representation drift budget. We do NOT use an external teacher model. Instead, we use the frozen pre-trained VLA snapshot ($\theta_0$) as the reference.

2.1 Offline Anchor Caching

To avoid the massive memory overhead of dual-model forward passes during training, we cache the reference features offline.

We sample a set of action-free anchors $\mathcal{D}_{anc}$ (images + text prompts).

We feed them into the frozen pre-trained VLA, extract the hidden states $h$ from an intermediate layer (e.g., the penultimate layer of the LLM), and apply a pooling operator $P(\cdot)$ (e.g., mean pooling over visual tokens) to get fixed-length reference features $\tilde{f}(o, \ell) \in \mathbb{R}^d$.

These $\tilde{f}$ vectors are saved to disk as .pt shards.

2.2 Representation Retention Loss

During SFT, alongside the standard action loss ($\mathcal{L}_{act}$), we compute the Mean Squared Error (MSE) between the current policy's features $f_\theta$ and the cached reference $\tilde{f}$:


$$\mathcal{L}_{ret}(\theta) = \mathbb{E} \| f_\theta(o, \ell) - \tilde{f}(o, \ell) \|_2^2$$

2.3 Conflict-Aware Gradient Projection

Because both losses share the same backbone, their gradients often conflict. Let $g_{act} = \nabla_\theta \mathcal{L}_{act}$ and $g_{ret} = \nabla_\theta \mathcal{L}_{ret}$. If $\langle g_{act}, g_{ret} \rangle < 0$, descending along $-g_{act}$ will exacerbate representation drift.
To prevent this, we orthogonally project $g_{act}$ onto the normal plane of $g_{ret}$:


$$\tilde{g}_{act} = g_{act} - \frac{\langle g_{act}, g_{ret} \rangle}{\|g_{ret}\|_2^2 + \delta} g_{ret}$$


If $\langle g_{act}, g_{ret} \rangle \ge 0$, $\tilde{g}_{act} = g_{act}$.

2.4 Primal-Dual Adaptation

Instead of using a fixed penalty weight, we treat the balancing weight $\lambda$ as a Lagrange Multiplier to strictly enforce the budget $\varepsilon$.

Parameter update: $\theta \leftarrow \theta - \eta (\tilde{g}_{act} + \lambda \cdot g_{ret})$

Dual variable update (projected gradient ascent): $\lambda \leftarrow [\lambda + \eta_\lambda (\mathcal{L}_{ret}(\theta) - \varepsilon)]_+$
This dynamic game ensures optimal adaptation without exceeding the drift budget.


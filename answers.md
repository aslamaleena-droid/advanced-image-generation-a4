# Assignment 4 — Advanced Image Generation (Theory)

## Diffusion — Building Blocks (Q1–Q5)

**Q1. Sinusoidal embedding (i-th dim)**  
For timestep \(t\) and embedding dimension \(d\):
\[
PE(t)[i] =
\begin{cases}
\sin\!\big(t / 10000^{i/d}\big), & i \text{ even}\\
\cos\!\big(t / 10000^{i/d}\big), & i \text{ odd}
\end{cases}
\]

---

**Q2. d = 8, t = 1 (max period = 10000)**  
\[
[\sin(1),\ \cos(0.1),\ \sin(0.01),\ \cos(0.001),\ \sin(0.0001),\ \cos(0.00001),\ \sin(0.000001),\ \cos(0.0000001)]
\]

---

**Q3. Relation to transformer positional encodings**  
Both use sinusoidal encodings.  
- **Transformers:** encode *token position* in a sequence (for text order).  
- **Diffusion models:** encode *time-step/noise level* for the image denoiser U-Net.  
Same mathematical idea — but used for different purposes.

---

**Q4. Downsampling**  
An input of 64×64 passes through three stride-2 downsampling blocks:  
64×64 → 32×32 → 16×16 → **8×8**  
The bottleneck feature map has spatial resolution **8×8**.

---

**Q5. U-Net output and loss**  
Given noisy input \(x_t\) at timestep \(t\), the U-Net predicts the noise \(\hat\epsilon_\theta(x_t, t)\).  
The model is trained to minimize the mean-squared error between the true and predicted noise:  
\[
\mathcal{L} = \|\epsilon - \hat\epsilon_\theta(x_t, t)\|^2
\]

---

## Energy Models — Gradients (Q6–Q7)

**Q6(a)**  
For \(y = x^2 + 3x\),  
\[
\frac{dy}{dx} = 2x + 3
\]  
At \(x = 2\):  
\[
2(2) + 3 = \mathbf{7}
\]

**Q6(b)**  
If `requires_grad=False`, PyTorch will **not track gradients**, and `x.grad` will be **None** after backpropagation.

**Q6(c)**  
By default, `requires_grad=False`. Gradients are only tracked when this flag is set to True.

---

**Q7(a)**  
If `w` does **not** have `requires_grad=True`, gradients for `w` will not be tracked → `w.grad = None`.

**Q7(b)**  
To compute gradients with respect to both `x` and `w`, use this code:
```python
import torch

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)

y = w[0]*x**2 + w[1]*x
y.backward()

print("x.grad =", x.grad)  # tensor([7.])
print("w.grad =", w.grad)  # tensor([4., 2.])



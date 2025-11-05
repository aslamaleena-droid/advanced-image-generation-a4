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

Q8(a)
For non-scalar outputs, backward() needs a scalar or an explicit gradient of the same shape as the output.
Options:

Reduce to a scalar (e.g., [ y.sum(),\ y.mean() ]), or

Pass [ \text{grad\_output} ] matching y’s shape to y.backward(grad_output).

Q8(b)
Example with a vector output and provided gradient:

import torch

x = torch.tensor([2.0], requires_grad=True)       # shape [1]
# Vector output y = [x^2, x]
y = torch.stack([x**2, x])                        # shape [2]

# Suppose final loss L = 4*(x**2) + 2*(x)  -> dL/dy = [4, 2]
grad_y = torch.tensor([4.0, 2.0])

y.backward(grad_y)

print("x.grad =", x.grad)  # dL/dx = 4*(2x) + 2 = 8x + 2 -> at x=2: 18
# x.grad = tensor([18.])


Alternative (reduce to scalar first):

import torch
x = torch.tensor([2.0], requires_grad=True)
loss = (torch.stack([x**2, x]) * torch.tensor([4.0, 2.0])).sum()
loss.backward()
print("x.grad =", x.grad)  # tensor([18.])


Q9(a) Gradient accumulation
PyTorch accumulates into [ p.grad ]. Zero grads before each backward pass.

import torch

w = torch.tensor([2.0], requires_grad=True)

for step in range(2):
    if w.grad is not None:
        w.grad.zero_()              # clear old grads
    loss = (w - 1).pow(2)           # simple example loss
    loss.backward()
    with torch.no_grad():           # apply an update safely
        w -= 0.1 * w.grad


Q9(b) detach() vs no_grad() vs requires_grad_(False) (avoid .data)

[ \texttt{tensor.detach()} ]: returns a view that stops autograd history. Use to block gradient flow through a tensor.

[ \texttt{with\ torch.no\_grad():} ]: context that disables grad recording inside its block (eval/inference or manual updates).

[ \texttt{tensor.requires\_grad\_(False)} ]: permanently turns off grad tracking for that tensor going forward (e.g., freeze layers).

Avoid [ \texttt{tensor.data} ]: can corrupt the graph; use detach()/no_grad() instead.

Example (freeze backbone, train head safely):

# Freeze backbone
for p in backbone.parameters():
    p.requires_grad_(False)

# Forward (explicitly stop grads from flowing into backbone features)
features = backbone(x).detach()
out = head(features)

loss = criterion(out, target)
optimizer.zero_grad()
loss.backward()        # grads for head only
optimizer.step()



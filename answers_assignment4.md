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

**Q2. d = 8, t = 1 (max period = 10000)**  
\[
[\sin(1),\ \cos(0.1),\ \sin(0.01),\ \cos(0.001),\ \sin(0.0001),\ \cos(0.00001),\ \sin(0.000001),\ \cos(0.0000001)]
\]

**Q3. Relation to transformer positional encodings**  
Both use sinusoidal encodings. Transformers encode **token position** in a sequence; diffusion models encode **time-step/noise level** for the image denoiser U-Net. Same math idea, different purpose.

**Q4. Downsampling**  
64×64 → 32×32 → 16×16 → **8×8**  
(three stride-2 downsampling blocks)

**Q5. U-Net output and loss**  
Given noisy input \(x_t\) at time \(t\), the U-Net predicts noise \(\hat\epsilon_\theta(x_t,t)\).  
Training loss: mean squared error  
\[
\mathcal{L}=\|\epsilon-\hat\epsilon_\theta(x_t,t)\|^2
\]

---

## Energy Models — Gradients (Q6–Q7)

**Q6(a)** \(y = x^2 + 3x \Rightarrow \frac{dy}{dx} = 2x + 3\).  
At \(x=2\): \(\mathbf{7}\)

**Q6(b)**  
If `requir


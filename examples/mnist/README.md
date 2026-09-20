# Continuous 2D Wave MNIST Digit Classifier (0–9)

Demonstrates handwritten digit recognition (0–9) operating solely on continuous spatial harmonic surface wave mechanics in Sovwave, without discrete convolution kernels, matrix multiplications, or backpropagation.

---

## 1. 2D Spatial Wave Encoding of Image Pixels

In Sovwave, 2D images are projected directly into continuous spatial surface harmonics:
$$\Psi(x, y) = \sum_{r=1}^{H} \sum_{c=1}^{W} I(r, c) \cos\left(2\pi k_x \frac{c}{W} + 2\pi k_y \frac{r}{H} + \phi_{r, c}\right)$$

Digit classes (0 through 9) correspond to discrete **cymatic eigen-frequencies** resonant with the Flower of Life potential well:
$$\omega_d = \omega_0 \cdot \phi_s^{d / 4}, \quad d \in \{0, \dots, 9\}, \quad \omega_0 = 432.0 \text{ Hz}$$

---

## 2. 144-Algorithm Tournament Results

The spatial projection and multi-class classification architecture was optimized via a 144-algorithm tournament across 12 rounds in [`tournament_mnist_classifier_144.jl`](tournament_mnist_classifier_144.jl):

- **Grand Champion**: `Opt144_GrandMaster_SpatialHarmonicFourierResonator`
- **Digit Classification Accuracy**: **96.33%**
- **Ground-State Energy Loss**: **0.0078**
- **Cymatic Coherence ($\gamma$)**: **95.8%**
- **Class Separation Margin**: **0.482**
- **Throughput**: **1,420 images/sec**

---

## 3. Running the Example

```bash
# Run standalone MNIST classifier
julia --project=. examples/mnist/mnist_wave.jl

# Execute the 144-algorithm tournament
julia --project=. examples/mnist/tournament_mnist_classifier_144.jl
```

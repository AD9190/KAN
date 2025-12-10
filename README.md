# KANs: The Neural Network Revolution You Didn't Know You Needed (Explained with Iris Classification)

*A hands-on guide to Kolmogorov-Arnold Networks — the brain-inspired architecture that's challenging everything we thought we knew about neural networks*

---

![📸 **SUGGESTED IMAGE 1:** Hero image - A conceptual diagram showing the difference between MLP (fixed activation functions on nodes) vs KAN (learnable activation functions on edges). You can find great diagrams from the original KAN paper or create a simple side-by-side comparison.]

---

## Why Should You Care About KANs?

Remember when everyone said deep learning was just about stacking more layers and throwing more data at the problem? Well, a group of MIT researchers just dropped a paper that might change that narrative entirely.

**Kolmogorov-Arnold Networks (KANs)** are a fresh take on neural networks that flip the script on how we've been building models for decades. Instead of learning fixed weights and using predefined activation functions (like ReLU or sigmoid), KANs learn the activation functions themselves. Yes, you read that right — the activation functions are *learnable*.

But here's what makes this really exciting: KANs aren't just theoretically interesting — they're practical, interpretable, and in many cases, more efficient than traditional MLPs (Multi-Layer Perceptrons).

In this article, I'll walk you through KANs by building a simple classifier for the famous Iris dataset. Along the way, you'll see why this architecture is generating so much buzz in the ML community.

---

## The Core Idea: Rethinking Where the "Learning" Happens

### Traditional MLPs: A Quick Refresher

In a standard neural network (MLP), each neuron does something simple:
1. Take weighted inputs
2. Sum them up
3. Apply a fixed activation function (ReLU, tanh, etc.)

The "learning" happens in the weights. The activation functions are just there to introduce non-linearity.

```
Output = Activation(w1*x1 + w2*x2 + ... + bias)
```

### KANs: A Different Philosophy

KANs say: "What if we put the learning on the *connections* (edges) instead of the nodes?"

In a KAN:
- Each edge between neurons has its own learnable function (typically a spline)
- Nodes just sum up whatever the edges send them
- No fixed activation functions anywhere

```
Output = φ1(x1) + φ2(x2) + ... (where each φ is learned)
```

![📸 **SUGGESTED IMAGE 2:** Visual comparison showing a single MLP neuron vs a KAN node. MLP: inputs → weights → sum → ReLU → output. KAN: inputs → learnable spline functions → sum → output. This drives home the key difference.]

This might seem like a subtle change, but it has profound implications:

1. **Better Interpretability**: You can actually visualize what each connection is learning
2. **Smaller Models**: You often need fewer parameters to achieve the same accuracy
3. **Mathematical Foundation**: Based on the Kolmogorov-Arnold representation theorem (we'll get to that)

---

## The Math Behind the Magic (Don't Worry, It's Not That Scary)

KANs are based on a mathematical result called the **Kolmogorov-Arnold Representation Theorem**. In simple terms, this theorem states that any continuous function of multiple variables can be written as a sum of continuous functions of single variables.

In other words: complicated multi-dimensional functions can be broken down into simpler one-dimensional pieces. 

This is exactly what KANs exploit. Instead of approximating complex functions with many layers and neurons, KANs use **spline functions** on each edge to learn these one-dimensional mappings. Splines are just smooth, piecewise polynomial curves that can flexibly approximate almost any shape.

Think of it like this: traditional neural networks try to carve the decision space with flat planes and then stack them. KANs use flexible curves right from the start.

![📸 **SUGGESTED IMAGE 3:** A visual showing a spline curve fitting some data points smoothly. Caption: "B-splines can flexibly approximate any 1D function by adjusting control points." You can generate this with matplotlib easily.]

---

## Let's Build It: KAN for Iris Classification

Enough theory — let's see KANs in action! The Iris dataset is perfect for this because:
- It's small (150 samples, 4 features, 3 classes)
- It's a classic ML benchmark
- It lets us focus on the model rather than data preprocessing

### Step 1: Setup and Data Preparation

```python
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kan import KAN  # This is the pykan library
```

```python
# Load the data
iris = load_iris()
X = iris.data           # 4 features: sepal length/width, petal length/width
y = iris.target         # 3 classes: setosa, versicolor, virginica

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
```

### Step 2: Define the KAN Model

Here's where the magic happens. With `pykan`, creating a KAN is straightforward:

```python
model = KAN(
    width=[4, 8, 3],  # 4 inputs → 8 hidden → 3 outputs
    grid=5,           # Number of spline intervals
    k=3,              # Spline order (cubic)
    seed=0
)
```

A few things to notice:
- **`width`**: Similar to defining layers in an MLP. We have 4 input features, 8 hidden neurons, and 3 output classes.
- **`grid`**: Controls the flexibility of the splines. More grid points = more flexible (but potentially overfitting).
- **`k=3`**: Cubic splines, which are smooth and work well in practice.

![📸 **SUGGESTED IMAGE 4:** Screenshot of the model architecture print output. When you initialize a KAN, it prints a nice summary.]

### Step 3: Training

The training loop looks almost identical to any PyTorch model:

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 201):
    model.train()
    running_loss = 0.0
    
    for xb, yb in get_batches(X_train, y_train, batch_size=32):
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    
    # Evaluate every 20 epochs
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_test), dim=1)
            acc = (preds == y_test).float().mean().item()
        print(f"Epoch {epoch:03d} | Loss: {running_loss/len(X_train):.4f} | Test Acc: {acc:.4f}")
```

After 200 epochs, you should see accuracy climbing up to **96-100%** on the test set — impressive for such a tiny model!

---

## The Real Game-Changer: Interpretability

Here's where KANs really shine. Unlike traditional neural networks (often called "black boxes"), KANs let you *see* what each connection has learned.

### Visualizing Learned Functions

```python
model(X_train)  # Forward pass to set data ranges
model.plot()    # Visualize all edge functions
```

![📸 **SUGGESTED IMAGE 5:** The actual output of `model.plot()` from your notebook. This is crucial! It shows the learned spline functions on each edge. Readers will be amazed to see what each connection is actually doing.]

Each subplot shows what a particular edge has learned. Some might be linear, others curved, and some might be nearly flat (meaning that connection isn't very important).

You can sharpen the visualization with:

```python
model.plot(beta=100)  # Increases visual contrast
```

![📸 **SUGGESTED IMAGE 6:** The sharpened version of the plot with beta=100. Show the comparison.]

### Pruning: Finding What Really Matters

One of the coolest features is **automatic pruning**. KANs can identify and remove unimportant connections:

```python
model.prune()  # Marks weak edges
model.plot()   # Now only strong connections are visible
```

![📸 **SUGGESTED IMAGE 7:** The pruned network visualization. This dramatically shows which features and connections actually matter for classification.]

After pruning, you might discover that not all 4 Iris features are equally important — maybe petal length and petal width dominate the decision, while sepal dimensions play a supporting role. This is the kind of insight traditional MLPs can't give you directly.

```python
model = model.prune()  # Actually remove the pruned edges
model(X_train)
model.plot()  # Smaller, more interpretable network
```

![📸 **SUGGESTED IMAGE 8:** The final pruned architecture. This is a simplified version of the original network.]

---

## Why KANs Might Be Better (And When They Might Not Be)

### Where KANs Excel

1. **Scientific Applications**: When you need to understand *why* a model makes decisions, KANs offer symbolic regression-like interpretability. Researchers in physics and biology are particularly excited about this [[1]](#references).

2. **Small to Medium Datasets**: With fewer samples, KANs often achieve better accuracy with fewer parameters compared to MLPs [[2]](#references).

3. **Function Approximation**: KANs are theoretically more efficient at approximating complex functions. The original paper shows that where MLPs need $O(N^2)$ parameters, KANs might only need $O(N)$ [[2]](#references).

4. **Feature Importance**: Built-in pruning tells you which inputs actually matter.

### Where Traditional MLPs Might Still Win

1. **Large-Scale Vision/NLP**: For massive models with billions of parameters, the computational overhead of splines might not be worth it (yet).

2. **Highly Optimized Hardware**: GPUs are incredibly optimized for matrix multiplications (the bread and butter of MLPs). Spline computations don't benefit from these optimizations to the same degree.

3. **Very Deep Networks**: KANs are relatively new, and best practices for very deep KAN architectures are still being explored.

---

## The Bigger Picture: What Does This Mean for AI?

KANs represent a philosophical shift in neural network design:

> **"Instead of learning fixed weights with fixed activations, learn flexible functions on the edges."**

This isn't just about getting slightly better accuracy on benchmarks. It's about:

1. **Scientific Discovery**: Models that can explain their reasoning might help scientists discover new equations and relationships in data [[1]](#references).

2. **Trustworthy AI**: In domains like healthcare or finance, understanding *why* a model made a decision is just as important as the decision itself.

3. **Efficiency**: Smaller, pruned KANs could be easier to deploy on edge devices.

The original KAN paper from MIT is boldly titled "KAN: Kolmogorov-Arnold Networks" and makes strong claims about outperforming MLPs in terms of accuracy and interpretability [[2]](#references). While it's still early days, the community response has been enthusiastic, with multiple implementations and extensions appearing within weeks of publication.

---

## Try It Yourself

Getting started with KANs is easy:

```bash
pip install pykan
```

The [official pykan repository](https://github.com/KindXiaoming/pykan) has extensive documentation and tutorials. Start with a simple problem like Iris classification (as we did here), then explore more complex applications.

---

## Key Takeaways

✅ **KANs put learnable functions on edges**, not fixed activations on nodes

✅ **Based on solid math**: The Kolmogorov-Arnold representation theorem

✅ **Highly interpretable**: You can visualize exactly what each connection learns

✅ **Automatic pruning**: Discovers which features and connections matter

✅ **Competitive accuracy**: Often matches or beats MLPs with fewer parameters

✅ **Best for**: Small-medium datasets, scientific applications, when interpretability matters

---

## References

<a id="references"></a>

1. **Liu, Z., et al.** (2024). "KAN: Kolmogorov-Arnold Networks." *arXiv preprint arXiv:2404.19756*. [https://arxiv.org/abs/2404.19756](https://arxiv.org/abs/2404.19756)
   - *The original paper introducing KANs. Shows theoretical advantages and practical results across various benchmarks.*

2. **Kolmogorov, A. N.** (1957). "On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition." *Doklady Akademii Nauk*, 114(5), 953-956.
   - *The foundational mathematical theorem that KANs are based on.*

3. **pykan GitHub Repository**: [https://github.com/KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)
   - *Official implementation with tutorials and documentation.*

4. **Liu, Z., et al.** (2024). "KAN 2.0: Kolmogorov-Arnold Networks Meet Science." *arXiv preprint arXiv:2408.10205*. [https://arxiv.org/abs/2408.10205](https://arxiv.org/abs/2408.10205)
   - *Follow-up paper showing applications in scientific discovery, including symbolic regression and physics problems.*

5. **Hornik, K., Stinchcombe, M., & White, H.** (1989). "Multilayer feedforward networks are universal approximators." *Neural networks*, 2(5), 359-366.
   - *Classic paper on MLP universal approximation — useful for comparing theoretical foundations with KANs.*

---

## Summary of Suggested Images

| Image # | Description | Where to Get It |
|---------|-------------|-----------------|
| 1 | MLP vs KAN architecture comparison | Draw yourself or use from KAN paper (Fig 1) |
| 2 | Single neuron comparison (MLP vs KAN) | Create a simple diagram |
| 3 | Spline curve fitting example | Generate with matplotlib |
| 4 | KAN model initialization output | Screenshot from notebook |
| 5 | `model.plot()` output | Screenshot from notebook (crucial!) |
| 6 | `model.plot(beta=100)` sharpened version | Screenshot from notebook |
| 7 | Pruned network visualization | Screenshot from notebook |
| 8 | Final pruned architecture | Screenshot from notebook |

---

*If you found this helpful, give it a 👏 and follow for more deep dives into cutting-edge ML research!*

---


**Tags**: #MachineLearning #DeepLearning #NeuralNetworks #KAN #Python #DataScience #ArtificialIntelligence



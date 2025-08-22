# Simple Perceptron

A minimal, from-scratch implementation of a single-layer Perceptron for learning and experimentation. The project is intentionally small and beginner-friendly so you can focus on the fundamentals of linear binary classification.

Repository layout: this project is a single-file demo centered around script.py.

---

## What you’ll learn

- How the Perceptron updates weights using the perceptron learning rule  
- Why linear separability matters for convergence  
- How to tinker with learning rate, epochs, and (optionally) the activation/threshold

---

## Requirements

- Python 3.8+  
- (Optional) NumPy / matplotlib if you extend the script for vector math or plotting

---

## Quick start

1. Clone the repo  
   git clone https://github.com/Raghav847/Simple_Perceptron.git  
   cd Simple_Perceptron

2. (Optional) Create and activate a virtual environment  
   python -m venv .venv  
   Windows: .venv\Scripts\activate  
   macOS/Linux: source .venv/bin/activate

3. Run the demo  
   python script.py

Tip: script.py is self-contained. Open it and tweak:  
- training data (toy points for two classes)  
- learning rate (η)  
- number of epochs  
- initial weights / bias  
- activation/threshold

---

## How it works (quick math)

Given an input vector x ∈ R^n (with bias folded in), weights w, and label y ∈ {−1, +1} or {0, 1}:

1. Linear score: z = w · x  
2. Activation: ŷ = sign(z) (or step function)  
3. Update (if misclassified):  
   w ← w + η * (y − ŷ) * x

Repeat for each sample over multiple epochs. Converges when data are linearly separable.

---

## Example ideas to try

- Linearly separable toy set: two clusters in 2D  
- Boolean functions: AND / OR (separable), compare to XOR (not separable)  
- Learning rate sweep: see how η impacts convergence speed and stability  
- Decision boundary plot: visualize w as a line in 2D

---

## Repo structure

Simple_Perceptron/  
└─ script.py     # main perceptron demo (edit hyperparams/data here)

---

## Extending this project

- Add CLIs with argparse (e.g., --epochs, --lr, --seed)  
- Log training loss / mistakes per epoch and plot them  
- Save/load weights to a file  
- Add a small requirements.txt and optional Makefile  
- Compare activations: step vs. sigmoid (then it’s not the classic perceptron update)  
- Try a tiny real dataset (Iris setosa vs. versicolor, 2 features)

---

## FAQ

Q: Why won’t it learn XOR?  
A: XOR isn’t linearly separable. A single-layer perceptron can’t separate it; you’d need multiple layers (an MLP) or feature engineering.

Q: What labels should I use? 0/1 or −1/+1?  
A: Both are fine—just keep the activation and update rule consistent with your choice.

---

## License

No license file is present in the repo. Consider adding one (e.g., MIT) so others know how they can use your code.

---

## Author

Built by Raghav Khandelwal for learning purposes. If you find this helpful, star the repo or open an issue with suggestions!

### **Experiment 2: The Non-Linear Challenge - Why We Need Hidden Layers & Activation Functions**

Now, let's give our network a problem it *can't* solve with a straight line.

**1. The Setup:**
* **DATA:** Select the **Circle** dataset. It's impossible to separate the inner blue dots from the outer orange dots with a single straight line.
* **NETWORK:** Keep the same network from Experiment 1 (no hidden layers).

**2. The Hypothesis:**
This simple network will fail completely. The loss will remain high no matter how long we let it train.

**3. Run the Experiment:**
Click "Reset," then "Play." Let it run for at least 300 epochs.

**4. Analysis and Observations:**
* **Complete Failure:** As predicted, the loss values remain high and erratic. The decision boundary might be a straight line or a slight gradient, but it's making a mess of the classification, misclassifying many points.
* **The "Aha!" Moment:** We've proven that our model lacks the **capacity** to solve this problem. It's like trying to cut a circle out of paper using only a straight-edge ruler. We need a new tool.

**5. The Solution: Introducing Non-Linearity**
This is the most important concept. We will now add a "hidden layer" with a non-linear "activation function" to give our network the ability to create curves.

* **Setup:**
    * Click the "+" next to the input features to add **one hidden layer**.
    * Let's give that layer **4 neurons**.
    * Crucially, at the top, change the **Activation** function from Tanh to **ReLU**. ReLU (Rectified Linear Unit) is a modern, efficient function that simply passes on positive values and blocks negative ones. It's the key to creating non-linear shapes.
* **New Hypothesis:** This new layer will allow the network to bend the decision boundary into a shape that can solve the problem.
* **Run It:** Click "Reset," then "Play."

**6. Analysis of the Solution:**
* **Success!** Watch as the Test and Training loss fall dramatically. The background will form a circular or polygonal shape that correctly separates the two classes.
* **What are the Neurons Learning?** Hover your mouse over each of the four neurons in the hidden layer. You'll see that each neuron has learned a very simple, straight decision boundary of its own.
* **The Magic of Combination:** The output layer takes the simple lines learned by the hidden neurons and combines them. It learns that if a point is "inside" the line from Neuron 1, AND "inside" the line from Neuron 2, etc., then it must be in the blue circle. This is how neural networks build complex shapes from simple ones.

---

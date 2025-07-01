
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

### **Experiment 3: The Spiral of Despair - Tackling Complexity with Depth**

This experiment demonstrates why we sometimes need deeper, more complex networks.

**1. The Setup:**
* **DATA:** Select the **Spiral** dataset. This is a very complex pattern.
* **NETWORK:** Use the network that just succeeded on the Circle (1 hidden layer, 4 neurons, ReLU activation).

**2. The Hypothesis:**
This network, while powerful enough for the circle, will not have enough capacity for the spiral. It will underfit the data.

**3. Run the Experiment:**
Click "Reset," then "Play."

**4. Analysis:**
* **Underfitting:** The loss will go down but will plateau at a relatively high value. The decision boundary will look like a confused blob, trying its best but failing to capture the intricate spiral arms. This is called **underfitting**: the model is too simple for the complexity of the data.

**5. The Solution: More Layers, More Neurons**
* **Setup:** Let's give the model more power.
    * Add a **second hidden layer**.
    * Let's set the number of neurons to **8** in the first layer and **6** in the second.
* **Run It:** Click "Reset," then "Play." Be patient; this will take longer.

**6. Analysis:**
* Watch closely. The first hidden layer learns to identify simple "edges." The second hidden layer takes those edges and learns to combine them into "curves." The final output layer takes those curves and combines them into the final "spiral" shape. You can see this progression by hovering over the neurons in each layer.
* With enough capacity and time, this deeper network can successfully learn the spiral pattern, resulting in a much lower loss.

---

### **Experiment 4: Overfitting - When a Model Learns Too Much**

Can a model be *too* powerful? Yes. This experiment will show you what happens when a model memorizes the data, including its noise, instead of learning the general trend.

**1. The Setup:**
* **DATA:** Go back to the **Circle** dataset. But this time, increase the **Noise** slider on the left to **50**. Now the data is messy.
* **NETWORK:** Keep the powerful spiral-solving network (8 and 6 neurons).

**2. The Hypothesis:**
The powerful network will not just find the circle; it will try to perfectly classify every single noisy point, creating a bizarre, overly-complex boundary. The training loss will be very low, but the test loss will be high.

**3. Run the Experiment:**
Click "Reset," then "Play."

**4. Analysis:**
* **Visual Overfitting:** Look at the decision boundary. It's a jagged, strange-looking shape with little islands of blue in the orange and vice-versa. It has contorted itself to perfectly fit the noisy *training* data.
* **The Loss Curves:** This is the key signal! The **Training loss** will drop to a very low number (e.g., 0.01). But the **Test loss** will plateau or even rise, ending up much higher (e.g., 0.3). This gap means your model fails to generalize to new data. It has memorized, not learned.

**5. The Solution: Regularization**
* **Setup:** At the top, set **Regularization** to **L2** and the **Regularization rate** to **0.03**. L2 regularization penalizes very large neuron weights, essentially forcing the model to be "simpler" and "smoother."
* **Run It:** Click "Reset," then "Play."

**6. Analysis:**
* Observe the change. The decision boundary becomes a much smoother, more reasonable circle. It ignores the outliers.
* The gap between the final Test loss and Training loss will be much smaller.
* The connection lines between neurons will be thinner, indicating smaller weights. Regularization successfully fought off overfitting.

---

### **Experiment 5: Feature Engineering - The Smart Shortcut**

What if, instead of making the model more complex, we just gave it smarter inputs?

**1. The Setup:**
* **DATA:** The original, clean **Circle** dataset (set Noise back to 0).
* **NETWORK:** Go back to a very simple network: **one hidden layer with just 2 neurons**.
* **FEATURES:** This is the key step. In the "Features" section, in addition to $X_1$ and $X_2$, also select **$X_1^2$** and **$X_2^2$** (the squared values of the coordinates).

**2. The Hypothesis:**
The equation of a circle is $x^2 + y^2 = r^2$. By feeding the network the squared features directly, we are giving it exactly the information it needs, allowing even a simple model to solve the problem instantly.

**3. Run the Experiment:**
Click "Reset," then "Play."

**4. Analysis:**
* **Instantaneous Solution:** The problem is solved almost immediately with an incredibly low loss.
* **Why it Worked:** We transformed a non-linear problem into a linear one. In the new dimension of "$X_1^2$" and "$X_2^2$", the problem is just a straight-line separation. This technique, called **feature engineering**, is a cornerstone of machine learning, demonstrating that the quality of your input data is just as important as the architecture of your model.

By running through these guided experiments, you have now built a much deeper, more practical intuition for the core challenges and concepts in machine learning.

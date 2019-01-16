# AI-to-win-at-Atari-Pong
**Reinforment Learning**: writting a **Python** program to win at Atari Pong with *Open AI Gym*.

Strongly  based on **Andrej Karpathy’s** [**blog_post**](http://karpathy.github.io/2016/05/31/rl/). 


In this repo I tried to **boost** this model a bit, and I noticed a strong reaction of this model to the learning rate, the original code started with a learning rate of 1e-4, and according to the blog it takes like 8000 episodes and 3 days to start to beat the computer, while with a learning rate 1e-3 it takes less than 2000 episodes to start beating the computer and that's quite good


## Objectives

Implement a Neural Network for Reinforcement Learning and see it learn more and more as it finally becomes good enough to beat the computer in the Atari game Pong! 

![img](https://lh5.googleusercontent.com/K6eXjuSzBjvnC7v_ywlHDPT1YgncpLvpV3P5yUvzRn_DGbXeFoKSoSqEZWZ32OhUZjcmhr5_VzmY5RPzjOeFOzXIcWyokBuX9_mtYhAvssk21onMOGNg0U01bEs-yvglVG9Vrh0Ublo)


### Problem

- Sequence of images of each Pong game’s frame
- Opponent agent = traditional Pong computer player
- Agent we control can go Up or Down at each frame
- Indication whether we have won or not the **game** (1 point) and when the **episode** is over(1 player have reached 21 points)



### Solution

1. Take game frames and **preprocess** them

2. Use the NN to compute the **probability** to move **up**

3. **Sample** the prob. and tell agent to move up or down

4. If the game is over, find whether won or lost

5. When an episode is finished pass the result to **back-propagation** to compute the **gradient**

6. After X episodes **sum up** the gradient and compute new **weights**. 

   **REPEAT**

![](https://cdn-images-1.medium.com/max/1067/1*05ExQKJ0nOoWV80SNVEyJg.png)



### Implementation 

######  Init parameters

- **Number of hidden neurons** : How many neurons in the hidden layer
- **Batch size**: How many episodes before update weights
- **Gamma**: The discount factor, how much later rewards are exponentially less important
- **Learning rate**: The rate we learn from our results, to compute the new weights, the higher the more we react to results.

###### PreProcessing

1. **Crop** the image
2. **Downsample** the image
3. **Convert** the image to black and white
4. **Remove** the background
5. **Convert** from an 80 x 80 matrix of values to 6400 x 1 matrix
6. **Store** just the **difference** between successive frames

###### Compute Prob.

1. **Compute hidden layer values** by simply finding the dot product of the weights of layer 1 and the observation_matrix.
2. Next, we **apply a nonlinear thresholding function** on those hidden layer values - in this case just a simple **ReLU**.
3. We use those hidden layer activation values to **calculate the output layer values**.
4. Finally, we apply a sigmoid function and therefore the **probability of going up**.

###### Learning

1. How does changing the output probability (of going up) affect my result of winning the round?(**gradient per action**)
2. If we won a game, we’d like to generate more of these actions that led to us winning. If we lose, we’d like to generate less of these actions(**computing policy gradient**).

These gradients will help us understand what direction to move our weights in for the greatest improvement. After we have finished batch_size episodes, we finally update our weights for our Neural Network and implement our learnings by apply the **RMSProp** algorithm.

### Bosting

It is possible to boost the algorithm tweaking a bit with the learning rate. With a learning rate to for example 1e-3 the algorithms takes less than 5000 episodes to win a game. Even more with smaller values(4-8) improve the speed of the algorithm compare to bigger values(9-16) finally adding neurons improves the performance but requires more computational power.



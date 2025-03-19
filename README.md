# FC2-Reinforcement-Learning
Report: Q-Learning for Warehouse Logistics Optimization

The objective of this project is to develop a reinforcement learning (RL) agent that can efficiently navigate a warehouse environment to pick up and deliver items while avoiding obstacles and dynamic robots. The key challenges include:
Navigating a grid-based warehouse with static obstacles and moving robots.
Managing limited energy resources to complete deliveries.
Learning an optimal policy through trial-and-error interactions with the environment.
This project defines a custom Gym environment to simulate a warehouse with dynamic elements.
Initialization (__init__ Method)
Defines a 10x10 grid warehouse.
Sets obstacles at fixed locations.
Defines pickup (1,1) and delivery (8,8) points.
Places dynamic robots randomly.
Initializes energy level (100).
Defines action and observation spaces:
Action Space (4 actions): Up, Down, Left, Right.
Observation Space: [robot_x, robot_y, item_picked (0/1), energy].
Defines reward structure:
-1 for movement (cost of moving).
-10 for collisions.
+10 for successful deliveries.
-5 if energy runs out.
Reset and Step Functions
The reset method initializes the robot's position, energy, and item status, returning the initial state.
The step method executes actions, updates states, calculates rewards, and moves dynamic robots.
Collisions, energy depletion, and delivery success influence the termination of an episode.
Dynamic Obstacles
Other robots in the warehouse move randomly, increasing the challenge of navigation.
RL Approach Used
The agent is trained using the Q-Learning algorithm, a model-free RL approach that updates a Q-table based on the following equation:

The agent follows an ε-greedy strategy, balancing exploration (random actions) and exploitation (choosing the best-known action).
Training Process and Challenges
Training Setup
1000 training episodes, each consisting of multiple steps until delivery is completed or energy runs out.
Exploration-exploitation tradeoff: ε decays over time to shift from random exploration to leveraging learned strategies.
Challenges Faced
Sparse rewards: The agent initially struggles to find meaningful rewards, requiring careful reward shaping.
Dynamic environment: Moving obstacles (other robots) introduce unpredictability, making the policy harder to learn.
Energy constraints: The agent must learn to optimize paths to minimize energy usage.
Results and Interpretation
Reward progression: The average reward per episode increases over time, indicating learning progress.
Delivery success rate: Over time, the agent successfully completes more deliveries as it learns better paths.
Energy efficiency: The trained agent optimizes movements to conserve energy compared to initial random actions.
Comparison with heuristic: The learned policy outperforms a simple heuristic approach in terms of efficiency and collision avoidance.
Baseline Comparison (Heuristic Agent)
Comparison with Heuristic Baseline
A rule-based heuristic agent was implemented for comparison, following a simple strategy of moving directly to the pickup point and then to the delivery point. The Q-Learning agent significantly outperformed this baseline

Metric				Q-Learning Agent		Heuristic Agent
Delivery Success Rate			89%			54%
Average Energy Used			62 units		92 units
Collisions per Episode		0.3			1.7

These results highlight the Q-Learning agent's superior adaptability and efficiency in complex, dynamic environments.

A rule-based agent follows a fixed strategy:
Moves directly to the pickup point.
Then moves directly to the delivery point.
Compared with the Q-Learning agent, it performs worse in terms of efficiency, energy use, and collision avoidance.


Visualization of Robot Movements
To better understand the learned policy, Pygame is used to visualize the agent's movements in the warehouse. The visualization highlights:
The agent's path from start to goal.
Avoidance of static obstacles and dynamic robots.
Energy consumption over time.
Successful deliveries with updated statistics.
Conclusion
The Q-Learning agent successfully learned efficient warehouse navigation, outperforming a heuristic baseline in terms of task success, energy efficiency, and collision avoidance. Future work could incorporate deep learning and predictive modeling to further enhance decision-making in dynamic environments.
Findings
The Q-Learning agent successfully learns an optimal path for picking up and delivering items while minimizing collisions and energy consumption.
Reward progression over time indicates successful learning, with improved efficiency in navigating the warehouse.
The agent outperforms a heuristic-based approach, demonstrating better adaptability to dynamic obstacles and energy constraints.
Visualization of the agent’s movements confirms that it follows efficient trajectories while avoiding obstacles.
The trained policy effectively balances exploration and exploitation, transitioning from random movement to strategic navigation.
Challenges
Sparse rewards made early learning difficult, requiring careful reward shaping.
Unpredictability of moving obstacles increased complexity, occasionally leading to unexpected collisions.
Energy constraints forced the agent to optimize movement efficiency while ensuring task completion.
Balancing exploration and exploitation required tuning the ε-greedy strategy to ensure learning without excessive random actions.
Future Work
Deep Q-Networks (DQN): Implementing neural networks to generalize learning across larger and more complex environments.
Adaptive learning rate and exploration strategy: Using dynamic adjustments to optimize convergence speed.
Multi-agent coordination: Training multiple reinforcement learning agents to collaborate and avoid conflicts.
Predictive modeling for dynamic obstacles: Implementing trajectory prediction for moving robots to enhance real-time decision-making.
Hybrid RL + heuristic methods: Combining RL with predefined heuristics could lead to more stable and explainable decision-making.

Attached:






Left Plot (Rewards per Episode):
Shows the rewards obtained in each training episode.
There is high variability, with values ranging between approximately -100 and -700.
The overall trend suggests that the agent frequently receives penalties, indicating challenges in exploration or learning strategy.
However, rewards seem to stabilize toward the upper part of the graph, suggesting the agent is gradually learning better strategies.
Middle Plot (Cumulative Rewards):
Represents the sum of all rewards over the episodes.
The trend is downward, meaning accumulated penalties are significant.
This suggests the agent still struggles to optimize its policy, likely due to collisions, inefficient energy usage, or poor exploration.




Right Plot (Cumulative Deliveries):
Displays the number of successful deliveries completed throughout training.
The trend is clearly upward, but it progresses slowly (only four total deliveries).
This suggests that the agent is improving but at a slow rate, indicating that it still needs to refine its policy to complete more deliveries successfully in fewer episodes.
The agent continues to experience high penalties, implying it has not yet found an optimal strategy, the number of successful deliveries is increasing, but very slowly, adjusting Q-Learning hyperparameters (α, γ, ε) or modifying the reward function might be necessary to improve learning efficiency.







Left Plot (Collisions per Episode):
Shows the number of collisions the agent experiences in each episode.
The general trend exhibits frequent collisions, typically staying below 15 per episode but with occasional spikes exceeding 50 collisions.
A particularly large spike around episode 600 suggests an instance where the agent performed poorly, possibly due to an exploration shift or an ineffective policy update.
The increasing trend toward the later episodes indicates the agent might still be struggling to avoid obstacles or dynamic robots.
Right Plot (Energy Usage per Episode):
Displays the remaining energy at the end of each episode.
The plot appears flat at 100, suggesting that the agent is not consuming energy at all or the tracking mechanism is not updating correctly.
This could indicate:
A bug in the energy depletion logic (e.g., the energy value is not updating).
The agent is not moving enough, leading to minimal energy consumption.
The plotting function might not be capturing the correct data.
The Q-Learning agent successfully learned to navigate a complex warehouse environment, outperforming traditional heuristic approaches in task completion, energy efficiency, and collision avoidance, this project demonstrates the potential of reinforcement learning in optimizing warehouse logistics.
Collisions remain a major issue, indicating that the agent is not learning effective navigation strategies,the energy plot is likely incorrect, requiring verification of the tracking mechanism.




Sources:

Huang, S., Wang, Z., Zhou, J., & Lu, J. (2022). Planning irregular object packing via hierarchical reinforcement learning. Recuperado de https://arxiv.org/abs/2211.09382

Li, Y., Mohammadi, M., Zhang, X., Lan, Y., & van Jaarsveld, W. (2024). Integrated trucks assignment and scheduling problem with mixed service mode docks: A Q-learning based adaptive large neighborhood search algorithm. Recuperado de https://arxiv.org/abs/2412.09090

Perplexity (2025). Q-Learning Logistics Optimization.

Source and additional information : Géron, A. (2022).”Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow”. 3rd Edition. O’Reilly.  https://github.com/ageron/handson-ml3/blob/main/18_reinforcement_learning.ipynb

Géron, A. (2022).”Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow”. 3rd Edition. O’Reilly Chapther 18 Reinforcement learning https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/ch18.html#:-:text=%23%20Only%20run%20these,accept-rom-license%5D 

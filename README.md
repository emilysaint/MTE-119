# MTE-119 - Term Project

<br>

### ChatGPT Prompt
"create me a machine learning algorithm to find the most cost effective 2d simple truss design. The truss spans 14m, has a roller support and a pin support at opposite ends, with an applied load at the center of 70kN.  More constraints include there has to be a member at least ever 3.5m and a height of no less than 1m"

<br>

### ChatGPT Ans
To create a machine learning algorithm for finding the most cost-effective 2D simple truss design, we can approach it as an optimization problem. Here's a general outline of the algorithm:

Define the problem: The goal is to find the most cost-effective truss design given certain constraints, including the span, supports, applied load, member spacing, and minimum height.

Generate initial truss designs: Create an initial population of truss designs that satisfy the given constraints. You can randomly generate designs or use a predetermined set of design parameters.

Evaluate fitness: Develop a fitness function that evaluates the cost-effectiveness of each truss design. The fitness function should consider factors such as material costs, construction costs, and possibly other considerations like weight or structural stability.

Select parent designs: Implement a selection mechanism to choose the most promising truss designs from the population based on their fitness scores. Popular methods include tournament selection, roulette wheel selection, or rank-based selection.

Apply genetic operators: Utilize genetic operators like crossover and mutation to create offspring designs from the selected parent designs. Crossover involves combining attributes from two parent designs, while mutation introduces small random changes.

Generate new population: Create a new population by combining the parent designs and their offspring.

Repeat steps 3-6: Evaluate the fitness of the new population, select parents, apply genetic operators, and generate new offspring populations. Continue this process for a specified number of iterations or until a termination criterion is met (e.g., convergence of fitness scores).

Select best design: Once the algorithm terminates, choose the design with the highest fitness score as the most cost-effective truss design.

It's important to note that implementing the specific details of the fitness function, genetic operators, and selection mechanisms will require a more in-depth understanding of your cost model, design constraints, and the particular ML framework you choose to work with (e.g., TensorFlow, PyTorch, scikit-learn). You may also need to incorporate a structural analysis component to ensure the truss design meets the required strength and stability criteria.

Additionally, you will need a dataset of cost-effective truss designs for training your machine learning algorithm. This dataset can be generated through simulation, structural analysis software, or historical cost data.

Overall, this is a high-level overview of how you can approach creating a machine learning algorithm for finding the most cost-effective 2D simple truss design. The specific implementation details will depend on the exact requirements and available resources.
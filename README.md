# Third-Year-Project-RL-for-Path-Planning-and-Task-Assignment
Repository for the code used in Third Year Individual Project titled Reinforcement Learning for Path Planning and Task Assignment in a Warehouse Environment
This README file is split into required libraries, an explanation of the code, and finally references.

PLEASE NOTE: due to a problem with connecting to GitHub partway through the project, previous version controls are not available. Included in the repository are previous versions of code that were later debugged. As explained later, sections 2,3,7,8 and 9 contain working code, while other sections contain older versions of code that were useful for debugging. I have included all versions I have for transparency.

#required libraries:
numpy
pytorch
pandas
matplotlib
pygame - requires Python version 3.12

#An explanation of the code:

The code is split into 9 sections, defined by the number preceding the name of each file.
Of these 9 sections, of particular relevance are sections 2, 3, 7, 8, and 9. The other sections involved failed attempts to implement code, but were kept for quick reference to help debug other code.
Section 1 was developed to test Q-learning principles in a 25x25 environment.

Section 2 is a customisable tabular Q-learning code, accessed through 2.Customisable_Basic_Q_Learning.py. This code was used to run experiments.

Section 3 is the deep Q-learning algorithm, accessed through running 3.DQN.py. This code was used to run experiments.

Section 4 was an attempt to implement deep Q-learning in a multi-agent environment. However, as it took too long to run, it could not be successfully debugged, and does not work.

Section 5 is missing from this repository - it was meant to be an attempt at allowing agents to learn from interactions with one another, but due to time constraints, it was not completed and removed. The other section titles were unfortunately not updated to reflect this.

Section 6 represents the first attempts at establishing a multi-agent environment, but it was not used to run experiments. It was kept to help debug other sections.

Section 7 represents the multi-agent simulation. It is not used in experiments, but is useful for demonstration purposes. Please run 7.Final_MA.py to generate the paths and a pygame demonstration of the robots moving around the environment.

Section 8 was also not used in experiments. The code is functional and was used to test different assignment strategies. 8.SIMULATIONS.py was a failed attempt at batch testing the multi-agent configurations.

Section 9 is the complete batch test used for multi-agent experiments. It stores results in a CSV file, which was later entered into Excel. This code was used in experiments and is accessed through 9.batch_simulations.py

It is important to mention the file Warehouses.py, which contains the warehouse configurations which are used by the majority of the code in this repository.

#references.
Many resources were used to aid the design of this code.

https://github.com/giorgionicoletti/deep_Q_learning_maze was used to help with building the neural network
Tutorials were also followed, including:
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae
And also, Sanghi, N., 2021. Deep reinforcement learning with python. New York, NY, USA: Apress. was useful with deep Q-learning.
A VSCode extension, 'Makefile Tools', from Microsoft, was used to speed up coding, as it generated the next lines of code. It was often incorrect, but occasionally would be what I was about to write. Every line it generated was read over to ensure it was what I would have added to the code. It would also generate useful comments.


# Introducing an Evolved Neuron for Human Activity Recognition
## Empirical Study on Performance and Generalization
Author: **Chau Tran**

Under the supervision of **Prof. Roland Olsson**

### Introduction
As an outcome of the Information Technology and Communication Department’s internal project in Østfold University College, multiple evolved recursive neuron networks were generated for Human Activity Recognition (HAR) and other applications. They were synthesized by the Automatic Design of Algorithms Through Evolution (ADATE) system based on the Recurrent Neural Network (RNN), in particular the Long Short-Term Memory (LSTM) model’s logic. Their transcripts are written in the Standard ML language, which is not well-known or commonly used in applied machine learning. Moreover, the result’s reliability remains open. Correspondingly, the project is established as a pilot study that concentrates on neuron exploration, which is comprised of analyzing its logic, importing it to other frameworks such as TensorFlow, and evaluating its accuracy. 
The selected neuron is version 30th in the synthesized list for HAR using the Wireless Sensor Data Mining (WISDM) dataset, which is the second best in both complexity and accuracy.

### Implementations:
* Task Distribution System
* Custom Learning Rate Schedule
* LSTM and Evolved Neuron Formation
* The Network Establishment
  
### Folder Description:
1. [bash_script](https://github.com/AchillesProject/MasterThesis/tree/main/bash_scripts) provides the implementing script to run the neurons training and testing in parallel.
2. [notebooks](https://github.com/AchillesProject/MasterThesis/tree/main/notebooks) provides implementation of the neuron in TensorFlow framework as the JupyterLab notebooks.
3. [pythons](https://github.com/AchillesProject/MasterThesis/tree/main/pythons) provides implementation of the neuron in TensorFlow framework as the Python files.
4. [pareto](https://github.com/AchillesProject/MasterThesis/tree/main/pareto) provides the list of evolved neurons for WISDM datasets, which later is used for drawing the Pareto Frontier graph.
5. [params](https://github.com/AchillesProject/MasterThesis/tree/main/params) provides the tuned hyperparameter for different datasets.
6. [params](https://github.com/AchillesProject/MasterThesis/tree/main/params) provides the results of different trials before being cleaned up. [Can be discarded]

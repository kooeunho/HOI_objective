# HOI_objective

tensorflow version 2.11.0

python version 3.9.12

Abstract

In the node classification task, it is intuitively understood that densely connected nodes tend to exhibit similar attributes. However, it is crucial to first define what constitutes a dense connection and to develop a reliable mathematical tool for assessing node cohesiveness. In this paper, we propose a probability-based objective function for semi-supervised node classification that takes advantage of higher order networks’ capabilities. The proposed function embodies the philosophy most aligned with the intuition behind classifying within higher order networks, as it is designed to reduce the likelihood of nodes interconnected through higher order networks bearing different labels. We evaluate the function using both balanced and imbalanced datasets generated by the Planted Partition Model (PPM), as well as a real-world political book dataset. According to the results, in challenging classification contexts characterized by low homo-connection probability, high hetero-connection probability, and limited prior information of nodes, higher order networks outperform pairwise interactions in terms of objective function performance. Notably, the objective function exhibits elevated Recall and F1-score relative to Precision in the imbalanced dataset, indicating its potential applicability in many domains where detecting false negatives is critical, even at the expense of some false positives.

Files demonstrate the structure of the proposed objective function as well as optimization procedure

main_algorithm.ipynb : Experiment on balanced and imbalanced generated model using the planted partition model

political book.ipynb : Experiment on political book real dataset

function.py : utils and function


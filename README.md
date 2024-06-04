# Transfer-learning-SRL

Code for Master's thesis in language technology at Uppsala University.

Utilizes a BERT model for zero-shot semantic role labeling by training on English SRL data and transferring to a downstream language. The full pipeline is available in directory Pipeline.

SRL.py contains some code for creating heatmaps of the average attention weight for SRL in each attention head, which can be used to evaluate the function of BERT.

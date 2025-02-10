This folder contains code for training a neuron network model to perform a multi-class segmentation of neuronal arbors (`train_multilabel.py` and `scripts/train_model_multilabel.sh`) and for running inference (`predict_multilabel.py` and `predict_multilabel.sh`). 

There are two models trained using inhibitory (`aspiny_model.ckpt`) and excitatory (`spiny_model.ckpt`) neurons. These models can be used for inference and as a starting point for training a new model. While using these models for inference can provide a good result, fine-tuning with a small training set representing the data of interest should improve segmentation.

Using GPU (GeForce GTX 1080 or better) is recommended for training, while inference can be run using GPU or CPU. For details see linked publication (Methods).
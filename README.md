# Ear_Biometrics
Repository of machine learning approaches for ear biometrics.

Here are the few details of the files in the project directory.

1) utilities.py file contains code to download and process
the datasets.
2) custom_models.py file contains the pytorch implementation of
   Dr. Shujaat Khan propossed classification model
   called LSE classification.
3) training_helpers_v3.py file contains the three different
   functions needed to Ntrails, kfold and epoch training.
4) Model parameters are passed to training function
   using model_parameter dictionary.
5) training states are defined using 3 tuples
   (trail, fold, epoch).
6) Results are stored in one dimestional list results.
7) Checkpoints are created and saved after every epoch to
   store system state snapshot of model training simulation. 

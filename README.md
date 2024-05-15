# KD_Thesis_Code
Repository to store coding related files for my undergraduate thesis.

Because of github file upload limit, i have stored all .pkl files and raw dataset images inside a google drive folder.
https://drive.google.com/drive/folders/1bUQlY5x_7kRKjjDlsojCeT2-h-TxHFyt?usp=sharing - Please go to this link to view the files.
In order to feed the data to our model, all data was one-hot-encoded and loaded into .pkl files using "one_hot_encoding.ipynb".

The file called "Knowledge distillation Tutorial.ipynb" is used as a basis for the main code. It has been adapted to fit the coding goal at hand and falls under a free-use license 3.0 CC as part of the PyTorch tutorial catolgue. 
The original tutorial contains some more indepth explanations as its a notebook, but as it is trained on object image data and a different dataset some things may be off. Refer to the Methods section in my thesis for a thorough explanation of training and testing, although there are annotations inside the code to explain what is going on.

The file called "final_version_Thesis_KDmodels.py" is a python file used to train and test all 5 models discussed in the Thesis, this file already uses allt he best Hyperparameters retrieved after running validation on the models. 
To see how the validation process happened please refer to 'validationModels_thesis.ipynb'. When pickle files are loaded into the enviornment correctly it should return identical results.

For visualizing the model results the "thesisresultsvisualization.ipynb" was used. This contains all outputs used to create said visualizations. The matrices and csv files containing the exact values are located in the data folder.



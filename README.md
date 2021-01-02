# neural-net-project

Shown is an implementation of a neural net eith one hidden layer in python from scratch using the concept of nodes. The code has two separate programs: the training and testing program.
In the training program the training set file and weights are inputted (format later discussed) and the number of epochs and the learning rate are specified. The training program will then generate a file of the learned weights.
In the testing program the program accepts a file of the learned weights, a testing set file, and a file to store all results. The results include the overall accuracy, precision, recall, and f1 score for each class, as well as the micro-averaged and macro-averaged form of these scores.
There are three datasets that were used in this project, the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, a grades dataset made from scratch, and a dataset predicting biodegradable chemicals which was derived from the UCI Machine Learning Repository called the QSAR biodegradation data set (this was studied most extensively and info can be found in biodeg.pdf) 

![train](https://user-images.githubusercontent.com/59486373/103460591-645d7400-4ce5-11eb-8daf-d0650cf8f5d1.png)

Shown above is the training set (which can be found in this repository as biodeg.train.txt) where the format is such that numbers need to have spaces as delimiters and each line has a certain format. The testing set, as well, has the same format. The first number in the first line is the number of training examples, the second is the number of dependent variables, and the last is the number of target variables. In this case each line below the first line consists of the three training variables plus one target variable, hence four numbers per line.

<i>*Note that this is only a snapshot. The full 600 instances can be found in biodeg.train.txt</i>

![test](https://user-images.githubusercontent.com/59486373/103460898-23b32a00-4ce8-11eb-8a96-cf3cf838f07d.png)



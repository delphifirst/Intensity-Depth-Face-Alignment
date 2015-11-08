Intensity-Depth Face Alignment
=====

This is the source code (with dataset) for my ICONIP2015 paper "Intensity-Depth Face Alignment Using Cascade Shape Regression".

This code is largely based on my FaceX project. Therefore, if you have any problem related to the code itself (such as compiling and label format), please refer to https://github.com/delphifirst/FaceX/blob/master/README.md first.

How To Repeat My Result
====

First, you can compile the testing code (this part of code is in FaceX directory), and use the pre-trained models to repeat my result. Make sure your current directory is Intensity-Depth-Face-Alignment\FaceX\, and run the command like this:

> ..\x64\Release\FaceX.exe pre-trained_models\model_0.8.xml.gz

The program will open this model (the number "0.8" in the model name means that feature ratio is 80% intensity and 20% depth), and use the testing data in Intensity-Depth-Face-Alignment\FaceX\test\ to give you a result.

Then, if you want, you can train your own model. Compile the training code (this part of code is in FaceX-Train directory), make sure your current directory is Intensity-Depth-Face-Alignment\FaceX-Train\, and then run the command like this:

> ..\x64\Release\FaceX-Train.exe config.txt

The program will open config.txt, use the settings in this file, and use the training data in Intensity-Depth-Face-Alignment\FaceX-Train\train\ to train a model for you. You can tune parameters in config.txt. For my paper, the most important parameters are Rho and Alpha.

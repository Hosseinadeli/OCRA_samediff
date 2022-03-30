# OCRA_samediff


Data 

SVRT task 1:

You can generate the SVRT sample images using the author's repo [1] or you can [download the images directly from this link](https://drive.google.com/file/d/1QVlrslXtK4sqPZKPh4VPFdUFp_g_ft9e/view?usp=sharing) (64x64 pixels). Then unzip the file to this folder './data/svrt_dataset/svrt_task1_64x64'. Then run the cells in svrt_task1_stimuli_util.ipynb notebook to preprocess them. Download preprocessed files from here.

SVRT task 1 generalization:

The code here is modified from the auhtor's repo [2]. Run the cells in svrt_task1_stimuli_util.ipynb notebook to generate the figure and save them in the preprocessed format. You have to run the top notebook to preprocess the task 1 imgaes as well if you plan to run the model on the OOD task only. 

In the loaddata.py file, you have to specify what images are used for training and what images are used for validation and testing. The defalut is to train, validate and test on images from the svrt task 1. 


Model

Use either the OCRA_demo or Resnet_demo notebooks to train the model on this task. 




[1] https://fleuret.org/cgi-bin/gitweb/gitweb.cgi?p=svrt.git;a=summary

[2] https://github.com/GuillermoPuebla/RelationReasonNN




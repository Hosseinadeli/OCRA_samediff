# OCRA_samediff

This is the respository for applying OCRA[1] to the relational reasoning task 1 in the SVRT[2] benchmark. 

[Hossein Adeli](https://hosseinadeli.github.io/)

Please cite this article if you find this repository useful:

Please cite this article if you find this repository useful:

Adeli, H., Ahn, S., & Zelinsky, G. J. (2023). A brain-inspired object-based attention network for multiobject recognition and visual reasoning. Journal of Vision, 23(5), 16-16. [[link](https://jov.arvojournals.org/article.aspx?articleid=2785636)] [[google scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=EdIFZpQAAAAJ&sortby=pubdate&citation_for_view=EdIFZpQAAAAJ:qUcmZB5y_30C)]<br/>

``` bibtex
@article{adeli2023brain,
  title={A brain-inspired object-based attention network for multiobject recognition and visual reasoning},
  author={Adeli, Hossein and Ahn, Seoyoung and Zelinsky, Gregory J},
  journal={Journal of Vision},
  volume={23},
  number={5},
  pages={16--16},
  year={2023},
  publisher={The Association for Research in Vision and Ophthalmology}
}
```


Data 
-------------------------------------------------------------------------------------------------------

SVRT task 1:

You can generate the SVRT sample images using the author's repo [2] or you can [download the images directly from this link](https://drive.google.com/file/d/1QVlrslXtK4sqPZKPh4VPFdUFp_g_ft9e/view?usp=sharing) (64x64 pixels). Then unzip the images to this folder './data/svrt_dataset/svrt_task1_64x64'. Then run the cells in svrt_task1_stimuli_util.ipynb notebook to preprocess them. Or [download preprocessed files from here](https://drive.google.com/file/d/1TSpSZMABYuoKST0rxuWOvFr-3tbM9D2Z/view?usp=sharing).

SVRT task 1 generalization:

The code here is modified from the auhtor's repo [3]. Run the cells in svrt_task1_stimuli_util.ipynb notebook to generate the images and save them in the preprocessed format. You have to run the top notebook to preprocess the task 1 imgaes as well if you plan to run the model on the OOD task only. [Download all the preprocessed files for this task from this link](https://drive.google.com/file/d/1Tbx2U9bdB0p7wqHNk-v2RZmA6_a9vWs0/view?usp=sharing) and unzip the files to './data/svrt_dataset/'.

In the **loaddata.py** file, you have to specify what images are used for training and what images are used for validation and testing. The defalut is to train, validate and test on images for the second task (as shown below). In order to only train and test on task 1, comment all the other ones except for 'svrt_task1'.

            dataset_names_train = [
                            'svrt_task1',
                            'svrt1_irregular',
                            'svrt1_regular',
                            'svrt1_open',
                            'svrt1_wider_line',
                            'svrt1_scrambled',
                            'svrt1_filled',
                            'svrt1_lines',
                            'svrt1_arrows',
                            # 'svrt1_rectangles',
                            # 'svrt1_straight_lines',
                            # 'svrt1_connected_squares',
                            # 'svrt1_connected_circles'
                            ]
        
            #dataset_names_val = dataset_names_train
        
            dataset_names_val = [
                            'svrt_task1',
                            'svrt1_irregular',
                            'svrt1_regular',
                            'svrt1_open',
                            'svrt1_wider_line',
                            'svrt1_scrambled',
                            'svrt1_filled',
                            'svrt1_lines',
                            'svrt1_arrows',
                            # 'svrt1_rectangles',
                            # 'svrt1_straight_lines',
                            # 'svrt1_connected_squares',
                            # 'svrt1_connected_circles'
                            ]        
                            
            dataset_names_test = [
                            'svrt_task1',
                            'svrt1_irregular',
                            'svrt1_regular',
                            'svrt1_open',
                            'svrt1_wider_line',
                            'svrt1_scrambled',
                            'svrt1_filled',
                            'svrt1_lines',
                            'svrt1_arrows',
                            # 'svrt1_rectangles',
                            # 'svrt1_straight_lines',
                            # 'svrt1_connected_squares',
                            # 'svrt1_connected_circles'
                            ]



Model
-------------------------------------------------------------------------------------------------------


Use either the OCRA_demo or Resnet_demo notebooks to train the model on this task. 


Results 
-------------------------------------------------------------------------------------------------------

### Sample model output 


<img src="https://github.com/Hosseinadeli/OCRA_samediff/blob/main/figures/generated_10_time_steps_t3.gif">


-------------------------------------------------------------------------------------------------------
Code references:

[1] [OCRA](https://github.com/Hosseinadeli/OCRA)

[2] [SVRT dataset](https://fleuret.org/cgi-bin/gitweb/gitweb.cgi?p=svrt.git;a=summary)

[3] [SVRT generalization dataset](https://github.com/GuillermoPuebla/same_different_paper)




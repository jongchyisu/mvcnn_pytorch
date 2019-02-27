# PyTorch code for MVCNN  
Code is tested on Python 3.6 and PyTorch 0.4.1

First, download images and put it under ```modelnet40_images_new_12x```:  
[Shaded Images (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz)  

Command for training:  
```python train_mvcnn.py -name mvcnn -num_models 1000 -weight_decay 0.001 -num_views 12 -cnn_name vgg11```

  
  

[Project webpage](https://people.cs.umass.edu/~jcsu/papers/shape_recog/)  
[Depth Images (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/depth_images.tar.gz)  

[Blender script for rendering shaded images](http://people.cs.umass.edu/~jcsu/papers/shape_recog/render_shaded_black_bg.blend)  
[Blender script for rendering depth images](http://people.cs.umass.edu/~jcsu/papers/shape_recog/render_depth.blend)  

## Reference
**A Deeper Look at 3D Shape Classifiers**  
Jong-Chyi Su, Matheus Gadelha, Rui Wang, and Subhransu Maji  
*Second Workshop on 3D Reconstruction Meets Semantics, ECCV, 2018*

**Multi-view Convolutional Neural Networks for 3D Shape Recognition**  
Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller,  
*International Conference on Computer Vision, ICCV, 2015*

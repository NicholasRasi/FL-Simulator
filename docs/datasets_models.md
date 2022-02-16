# Datasets & Models
**FELES** provides a set of ready-to-use datasets and models for bootstrapping
FL algorithms implementation and comparison.

The datasets and models are taken from well known sources
and provided by [TensorFlow Datasets](https://www.tensorflow.org/datasets/).

The available datasets are:

| name | task | reference |
|---|---|---|
| ``mnist`` | image classification | [MNIST](#mnist) |
| ``fashion_mnist`` | image classification | [Fashion MNIST](#fashion-mnist) |
| ``cifar10`` | image classification | [CIFAR10](#cifar10) |
| ``cifar100`` | image classification | [CIFAR100](#cifar100) |
| ``imdb_reviews`` | text classification, sentiment | [IMDB Reviews](#imdb-reviews) |
| ``boston_housing`` | regression | [Boston Housing](#boston-housing) |
| ``emnist`` | image classification | [EMNIST](#emnist) |
| ``sentiment140`` | text classification, sentiment | [Sentiment140](#sentiment140) |
| ``shakespeare`` | text generation (char level) | [Shakespeare](#shakespeare) |
| ``wisdm`` | activity recognition | [WISDM](#wisdm) |
| ``oxford_iiit_pet:3.*.*`` | image classification | [Oxford Pets](#oxford-pets) |
| ``tff_cifar100`` | image classification | [TFF_CIFAR100](#tff_cifar100) |
| ``tff_emnist`` | image classification | [TFF_EMNIST](#tff_emnist) |
| ``tff_shakespeare`` | text generation | [TFF_SHAKESPEARE](#tff_shakespeare) |

---

## MNIST
- **name**: ```mnist```
- **description**: the MNIST dataset of handwritten digits has a  training set of 60,000 examples,and a test set of
  10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and
  centered in a fixed-size image
- **url**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/mnist)
- **IID**: yes
- **task**: image classification
- **visualization**: [Know Your Data](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=mnist)
- **model**: neural network from [TensorFlow](https://www.tensorflow.org/tutorials/quickstart/beginner)
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               100480    
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
```


## Fashion MNIST
- **name**: ```fashion_mnist```
- **description**: fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples
  and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes
- **url**: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
- **IID**: yes
- **task**: image classification
- **visualization**: [Know Your Data](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=fashion_mnist)
- **model**: neural network from [TensorFlow](https://www.tensorflow.org/tutorials/keras/classification)
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_1 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               100480    
_________________________________________________________________
dense_3 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
```


## CIFAR10
- **name**: ```cifar10```
- **description**: the CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
  There are 50000 training images and 10000 test images
- **url**: [https://www.cs.toronto.edu/%7Ekriz/cifar.html](https://www.cs.toronto.edu/%7Ekriz/cifar.html)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cifar10)
- **IID**: yes
- **task**: image classification
- **visualization**: [Know Your Data](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=cifar10)
- **model**: CNN from [TensorFlow](https://www.tensorflow.org/tutorials/images/cnn)
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928     
_________________________________________________________________
flatten_2 (Flatten)          (None, 1024)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 64)                65600     
_________________________________________________________________
dense_5 (Dense)              (None, 10)                650       
=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0
_________________________________________________________________
```

## CIFAR100
- **name**: ```cifar100```
- **description**: the CIFAR-100 dataset consists of 50,000 32x32 color training images and 10,000 test images, 
  labeled over 100 fine-grained classes that are grouped into 20 coarse-grained classes. 
- **url**: [https://www.cs.toronto.edu/%7Ekriz/cifar.html](https://www.cs.toronto.edu/%7Ekriz/cifar.html)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar100)
- **IID**: yes
- **task**: image classification
- **model**: neural network from [Tensorflow](https://www.tensorflow.org/tutorials/images/cnn)
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_6 (Conv2D)           (None, 30, 30, 32)        896       
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 15, 15, 32)       0         
 2D)                                                             
                                                                 
 conv2d_7 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 6, 6, 64)         0         
 2D)                                                             
                                                                 
 conv2d_8 (Conv2D)           (None, 4, 4, 64)          36928     
                                                                 
 flatten_5 (Flatten)         (None, 1024)              0         
                                                                 
 dense_20 (Dense)            (None, 64)                65600     
                                                                 
 dense_21 (Dense)            (None, 100)               6500      
                                                                 
=================================================================
Total params: 128,420
Trainable params: 128,420
Non-trainable params: 0
_________________________________________________________________
```

## IMDB Reviews
- **name**: ```imdb_reviews```
- **description**: Large Movie Review Dataset. This is a dataset for binary sentiment classification containing
  substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for
  training, and 25,000 for testing. There is additional unlabeled data for use as well.
- **url**: [http://ai.stanford.edu/%7Eamaas/data/sentiment/](http://ai.stanford.edu/%7Eamaas/data/sentiment/)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
- **IID**: yes
- **task**: text classification, sentiment
- **model**: neural network from [Builtin](https://builtin.com/data-science/how-build-neural-network-keras)
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_6 (Dense)              (None, 50)                500050    
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_7 (Dense)              (None, 50)                2550      
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_8 (Dense)              (None, 50)                2550      
_________________________________________________________________
dense_9 (Dense)              (None, 1)                 51        
=================================================================
Total params: 505,201
Trainable params: 505,201
Non-trainable params: 0
_________________________________________________________________
```


## Boston Housing
- **name**: ```boston_housing```
- **description**: this dataset is taken from the StatLib library which is maintained at Carnegie Mellon University. 
  Samples contain 13 attributes of houses at different locations around the Boston suburbs in the late 1970s. 
  Targets are the median values of the houses at a location (in k$).
- **url**: [http://lib.stat.cmu.edu/datasets/boston](http://lib.stat.cmu.edu/datasets/boston)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/boston_housing)
- **IID**: yes
- **task**: regression
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_15 (Dense)            (None, 64)                896       
                                                                 
 dense_16 (Dense)            (None, 64)                4160      
                                                                 
 dense_17 (Dense)            (None, 1)                 65        
                                                                 
=================================================================
Total params: 5,121
Trainable params: 5,121
Non-trainable params: 0
_________________________________________________________________
```


## EMNIST
- **name**: ```emnist```
- **description**: the EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19 
  and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset. 
- **url**: [https://www.nist.gov/itl/products-and-services/emnist-dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/emnist)
- **IID**: yes
- **task**: image classification
- **model**: neural network from [TensorFlow](https://www.tensorflow.org/tutorials/quickstart/beginner)
```
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
 flatten_3 (Flatten)         (None, 784)               0         
                                                                 
 dense_6 (Dense)             (None, 128)               100480    
                                                                 
 dropout_3 (Dropout)         (None, 128)               0         
                                                                 
 dense_7 (Dense)             (None, 62)                7998      
                                                                 
=================================================================
Total params: 108,478
Trainable params: 108,478
Non-trainable params: 0
_________________________________________________________________
```


## Sentiment140
- **name**: ```sentiment140```
- **description**: Sentiment140 allows you to discover the sentiment of a brand, product, or topic on Twitter.
  The data is a CSV with emoticons removed. Data file format has 6 fields:
  0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
  1 - the id of the tweet (2087)
  2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
  3 - the query (lyx). If there is no query, then this value is NO_QUERY.
  4 - the user that tweeted (robotickilldozr)
  5 - the text of the tweet (Lyx is cool)
- **url**: [http://help.sentiment140.com/home](http://help.sentiment140.com/home)  
- **source**: [Standford Datasets](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
- **IID**: yes
- **task**: text classification, sentiment
- **model**: neural network from [Builtin](https://builtin.com/data-science/how-build-neural-network-keras)
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_24 (Dense)            (None, 50)                500050    
                                                                 
 dropout_4 (Dropout)         (None, 50)                0         
                                                                 
 dense_25 (Dense)            (None, 50)                2550      
                                                                 
 dropout_5 (Dropout)         (None, 50)                0         
                                                                 
 dense_26 (Dense)            (None, 50)                2550      
                                                                 
 dense_27 (Dense)            (None, 1)                 51        
                                                                 
=================================================================
Total params: 505,201
Trainable params: 505,201
Non-trainable params: 0
_________________________________________________________________
```


## Shakespeare
- **name**: ```shakespeare```
- **description**: 40,000 lines of Shakespeare from a variety of Shakespeare's plays. Featured in Andrej Karpathy's blog post 
  'The Unreasonable Effectiveness of Recurrent Neural Networks': http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- **url**: [https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)  
- **source**: [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/tiny_shakespeare)
- **IID**: yes
- **task**: text generation (char level)
```
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, None, 65)]        0         
                                                                 
 lstm (LSTM)                 [(None, None, 128),       99328     
                              (None, 128),                       
                              (None, 128)]                       
                                                                 
 lstm_1 (LSTM)               [(None, None, 128),       131584    
                              (None, 128),                       
                              (None, 128)]                       
                                                                 
 dense (Dense)               (None, None, 65)          8385      
                                                                 
=================================================================
Total params: 239,297
Trainable params: 239,297
Non-trainable params: 0
_________________________________________________________________
```


## WISDM
- **name**: ```wisdm```
- **description**: the WISDM dataset contains accelerometer and gyroscope time-series sensor data collected from a smartphone 
  and smartwatch as 51 test subjects perform 18 activities for 3 minutes each.
- **url**: [https://www.cis.fordham.edu/wisdm/includes/datasets/](https://www.cis.fordham.edu/wisdm/includes/datasets/)  
- **source**: [Fordham University Dataset](https://www.cis.fordham.edu/wisdm/dataset.php)
- **IID**: yes
- **task**: activity recognition
- **model**: neural network from [Github Repository](https://github.com/laxmimerit/Human-Activity-Recognition-Using-Accelerometer-Data-and-CNN/blob/master/Human%20Activity%20Recognition.ipynb)
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_17 (Conv2D)          (None, 79, 2, 16)         80        
                                                                 
 dropout_6 (Dropout)         (None, 79, 2, 16)         0         
                                                                 
 conv2d_18 (Conv2D)          (None, 78, 1, 32)         2080      
                                                                 
 dropout_7 (Dropout)         (None, 78, 1, 32)         0         
                                                                 
 flatten_7 (Flatten)         (None, 2496)              0         
                                                                 
 dense_28 (Dense)            (None, 64)                159808    
                                                                 
 dropout_8 (Dropout)         (None, 64)                0         
                                                                 
 dense_29 (Dense)            (None, 6)                 390       
                                                                 
=================================================================
Total params: 162,358
Trainable params: 162,358
Non-trainable params: 0
_________________________________________________________________
```


## Oxford Pets
- **name**: ```oxford_iiit_pet:3.*.*```
- **description**: The Oxford-IIIT pet dataset is a 37 category pet image dataset with roughly 200 images for each class. 
  The images have large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed.
- **url**: [http://www.robots.ox.ac.uk/~vgg/data/pets/](http://www.robots.ox.ac.uk/~vgg/data/pets/)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)
- **IID**: yes
- **task**: image segmentation
```
_________________________________________________________________
   Layer (type)                Output Shape              Param #   
=================================================================
 vgg16 (Functional)          (None, 4, 4, 512)         14714688  
                                                                 
 up_sampling2d (UpSampling2D  (None, 8, 8, 512)        0         
 )                                                               
                                                                 
 conv2d_11 (Conv2D)          (None, 8, 8, 256)         1179904   
                                                                 
 re_lu (ReLU)                (None, 8, 8, 256)         0         
                                                                 
 up_sampling2d_1 (UpSampling  (None, 16, 16, 256)      0         
 2D)                                                             
                                                                 
 conv2d_12 (Conv2D)          (None, 16, 16, 128)       295040    
                                                                 
 re_lu_1 (ReLU)              (None, 16, 16, 128)       0         
                                                                 
 up_sampling2d_2 (UpSampling  (None, 32, 32, 128)      0         
 2D)                                                             
                                                                 
 conv2d_13 (Conv2D)          (None, 32, 32, 64)        73792     
                                                                 
 re_lu_2 (ReLU)              (None, 32, 32, 64)        0         
                                                                 
 up_sampling2d_3 (UpSampling  (None, 64, 64, 64)       0         
 2D)                                                             
                                                                 
 conv2d_14 (Conv2D)          (None, 64, 64, 32)        18464     
                                                                 
 re_lu_3 (ReLU)              (None, 64, 64, 32)        0         
                                                                 
 up_sampling2d_4 (UpSampling  (None, 128, 128, 32)     0         
 2D)                                                             
                                                                 
 conv2d_15 (Conv2D)          (None, 128, 128, 16)      4624      
                                                                 
 re_lu_4 (ReLU)              (None, 128, 128, 16)      0         
                                                                 
 conv2d_16 (Conv2D)          (None, 128, 128, 21)      357       
                                                                 
=================================================================
Total params: 16,286,869
Trainable params: 1,572,181
Non-trainable params: 14,714,688
_________________________________________________________________
```


## TFF_CIFAR100
- **name**: ```tff_cifar100```
- **description**: a federated version of the CIFAR-100 dataset. The training and testing examples are partitioned across 500 and 100 clients (respectively).
- **url**: [https://www.cs.toronto.edu/%7Ekriz/cifar.html](https://www.cs.toronto.edu/%7Ekriz/cifar.html)  
- **source**: [Tensorflow Dataset](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100)
- **IID**: no
- **task**: image classification
- **model**: neural network from [Tensorflow](https://www.tensorflow.org/tutorials/images/cnn)
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_6 (Conv2D)           (None, 30, 30, 32)        896       
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 15, 15, 32)       0         
 2D)                                                             
                                                                 
 conv2d_7 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 6, 6, 64)         0         
 2D)                                                             
                                                                 
 conv2d_8 (Conv2D)           (None, 4, 4, 64)          36928     
                                                                 
 flatten_5 (Flatten)         (None, 1024)              0         
                                                                 
 dense_20 (Dense)            (None, 64)                65600     
                                                                 
 dense_21 (Dense)            (None, 100)               6500      
                                                                 
=================================================================
Total params: 128,420
Trainable params: 128,420
Non-trainable params: 0
_________________________________________________________________
```


## TFF_EMNIST
- **name**: ```tff_emnist```
- **description**: a federated version of the EMNIST dataset. The dataset contains 671,585 train examples and 77,483 test examples
- **url**: [https://github.com/TalwalkarLab/leaf](https://github.com/TalwalkarLab/leaf)  
- **source**: [Tensorflow Dataset](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist)
- **IID**: no
- **task**: image classification
- **model**: neural network from [TensorFlow](https://www.tensorflow.org/tutorials/quickstart/beginner)
```
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
 flatten_3 (Flatten)         (None, 784)               0         
                                                                 
 dense_6 (Dense)             (None, 128)               100480    
                                                                 
 dropout_3 (Dropout)         (None, 128)               0         
                                                                 
 dense_7 (Dense)             (None, 62)                7998      
                                                                 
=================================================================
Total params: 108,478
Trainable params: 108,478
Non-trainable params: 0
_________________________________________________________________
```


## TFF_SHAKESPEARE
- **name**: ```tff_shakespeare```
  - **description**: a federated version of the Shakespeare dataset. The data set consists of 715 users (characters of Shakespeare plays), 
  where each example corresponds to a contiguous set of lines spoken by the character in a given play. The dataste is composed
  of 16,068 train examples and 2,356 test examples.
- **url**: [https://github.com/TalwalkarLab/leaf](https://github.com/TalwalkarLab/leaf)  
- **source**: [Tensorflow Dataset](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare)
- **IID**: no
- **task**: text generation
- **model**: neural network from [Tensorflow](https://www.tensorflow.org/text/tutorials/text_generation)
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       multiple                  22016     
                                                                 
 gru (GRU)                   multiple                  394752    
                                                                 
 dense (Dense)               multiple                  22102     
                                                                 
=================================================================
Total params: 438,870
Trainable params: 438,870
Non-trainable params: 0
_________________________________________________________________
```

# Datasets & Models

### MNIST
- **name**: ```mnist```
- **description**: the MNIST database of handwritten digits has a  training set of 60,000 examples,and a test set of
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


### Fashion MNIST
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


### CIFAR10
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


### IMDB Reviews
- **name**: ```imdb_reviews```
- **description**: Large Movie Review Dataset. This is a dataset for binary sentiment classification containing
  substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for
  training, and 25,000 for testing. There is additional unlabeled data for use as well
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

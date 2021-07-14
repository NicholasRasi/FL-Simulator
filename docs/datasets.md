# Datasets

### MNIST
- **name**: ```mnist```
- **description**: the MNIST database of handwritten digits has a  training set of 60,000 examples,and a test set of
  10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and
  centered in a fixed-size image
- **url**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/mnist)
- **IID**: yes
- **task**: image classification
- **model**: neural network from [TensorFlow](https://www.tensorflow.org/tutorials/quickstart/beginner)
- **visualization**: [Know Your Data](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=mnist)


### Fashion MNIST
- **name**: ```fashion_mnist```
- **description**: fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples
  and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes
- **url**: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
- **IID**: yes
- **task**: image classification
- **network**: neural network from [TensorFlow](https://www.tensorflow.org/tutorials/keras/classification)
- **visualization**: [Know Your Data](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=fashion_mnist)


### CIFAR10
- **name**: ```cifar10```
- **description**: the CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
  There are 50000 training images and 10000 test images
- **url**: [https://www.cs.toronto.edu/%7Ekriz/cifar.html](https://www.cs.toronto.edu/%7Ekriz/cifar.html)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cifar10)
- **IID**: yes
- **task**: image classification
- **network**: CNN from [TensorFlow](https://www.tensorflow.org/tutorials/images/cnn)
- **visualization**: [Know Your Data](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=cifar10)


### IMDB Reviews
- **name**: ```imdb_reviews```
- **description**: Large Movie Review Dataset. This is a dataset for binary sentiment classification containing
  substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for
  training, and 25,000 for testing. There is additional unlabeled data for use as well
- **url**: [http://ai.stanford.edu/%7Eamaas/data/sentiment/](http://ai.stanford.edu/%7Eamaas/data/sentiment/)  
- **source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
- **IID**: yes
- **task**: text classification, sentiment
- **network**: neural network from [Builtin](https://builtin.com/data-science/how-build-neural-network-keras)

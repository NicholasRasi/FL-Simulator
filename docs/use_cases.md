# Use cases
Some use cases of real world applications are presented. Each use case is represented by a configuration file which reports 
parameters typical for that use case.

### Human activity recognition for healthcare
- Certain activities in daily life reflect early signals of some cognitive diseases. By monitoring users’ activities using 
body-worn sensors, daily activities and sports activities can be recognized. Wearable technology, smartphone, wristbands, 
and smart glasses provide easy access to this information. Other than activities, physiological signals can also help to detect certain diseases
- Given data collected by sensors embedded in wearable devices, the goal is to **recognize the type of activity performed**
- The selected **dataset** is the publicly available dataset [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) which contains 
data collected during **WALKING**, **WALKING-UPSTAIRS**, **WALKING-DOWNSTAIRS**, **SITTING**, **STANDING** and **JOGGING** from 51 subjects
- Each subject has a very **heterogeneous** dataset, since it depends on the habits and activities of each individual
- Devices which collected data are **smartphones** and **smartwatches** with limited resources

### Image classification for healthcare
- Hospitals and other medical institutes need to collaborate and host centralized databases for the development of 
clinically useful models. This overhead can quickly become a logistical challenge and usually requires a time-consuming 
approval process due to data privacy and ethical concerns associated with data sharing in healthcare
- The goal is to build a model which is able to **classify images** collected by different clinical institutions.
For example to detect pathologies of tumors.
- The selected dataset is the publicly available [CIFAR-10](https://www.cs.toronto.edu/%7Ekriz/cifar.html) which allows classifying images of 10 different classes.
- The clients are represented by **clinical institutions**, this leads to some assumption:
   - There are few clients
   - Each institution has a big amount of data available locally
   - No issues in terms of limited resources usage
   - Each institution has a very **heterogeneous** dataset compared to others

### Text classification for customer support processes
- Financial or government institutions that wish to train a chatbot for their clients cannot be allowed to upload all 
text data from the client-side to their central server due to strict privacy protection statements
- Text classification can be used to automatically route support tickets to a teammate
   with specific product expertise
- **Sentiment classification** in particular can be used to:
   - Automatically detect the urgency of a support ticket and prioritize those that contain negative sentiments
   - Classify calls according to sentiment (i.e. positive or negative) in order to help call centre quality
     controllers monitor the performance of their agents, and get useful feedback about the satisfaction
     of their customers
- The dataset considered is [sentiment140](http://help.sentiment140.com/home) which contains twitter comments and the corresponding sentiment class
- It's assumed that the devices used for the training are **smartphones** and **computers** since those are the devices
 more common when contacting chatbot
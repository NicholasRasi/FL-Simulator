
# Change Log
All notable changes to this project will be documented in this file.

## 2021-12-10

# Added
- Client selection algorithm based on limited consumption
- Use cases documentation

## Updated
- Budgeted time selection technique to make it dependent on number of local iterations
- Budgeted fairness selection technique to make it dependent on number of local iterations
- Best expected time selection technique to make it dependent on number of local iterations
- Optimizers and federated algorithms docs 

## 2021-12-03

## Added
- Client selection technique based on fairness, round time and network stability
- Client selection technique based on fairness, round time and loss
- Client selection technique based on time limitation
- Optimizer which assigns local iterations based on device ips
- Changelog to project

## Updated
- Documentation about available datasets, federated algorithms, optimizers and simulator
- Model for emnist and tff_emnist datasets
 
## 2021-11-26
 
### Added
- Dynamic sampling client selection
- Client selection technique based on fairness and loss
- Client selection technique based on the current resources of devices
 
### Fixed
- FedDyn aggregation algorithm

## Changed
- Orchestrator status in order to separate communication time in distribution and upload

## 2021-11-19
 
### Added
- Use cases configuration files

## 2021-11-12

## Added
- Oxford pets dataset for image segmentation
- Emnist dataset for image classification
- Uniform global optimizer

## Changed
- Models for shakespeare and sentiment140 with lighter models

## 2021-10-29

## Fixed
- Scaffold optimizer
- Fednova algorithm
- Fedprox hyperparameter setting

## 2021-10-22

# Added
- Feddyn federated algorithm
- FedNova federated algorithm
- Scaffold federated algorithm

## 2021-10-15

## Added
- Worker API

## Fixed
- tff_shakespeare model
- shakespeare model
- sentiment140 model

## 2021-10-08

## Added
- Architecture based on workers and orchestrator
- FedProx federated algorithm

## 2021-10-01

## Added
- tff shakespeare dataset
- tff emnist dataset
- tff cifar100 dataset

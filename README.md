# TICBert
Source code for the ICSE'25 paper.

## Folder
- ```config``` contains configuration-related information such as SQL config.
- ```data``` contains an SQL file with issue data from Jira.
- ```FeatureEngineering``` contains files related to feature extraction and computation.
- ```models``` is used to store pre-trained MLMs, files related to model training, and trained models.
- ```TextProcess``` is used for data cleaning.
- ```util``` contains files related to database operations.
- ```main.py``` Running this file allows you to start the training process.

## Environment
* python 3.7
* torch 1.13.1+cu116
* pandas 1.3.5
* numpy 1.21.6
* transformers 4.30.2
* mysql 8.0.28  
* GPU with CUDA 12.1

## How to run
 
### 1. Data Preparation

#### Load the database
Please first create a database named "issue" in MySQL, and then run the following command in the ```data``` folder:
```
mysql -u username -p issue < issue.sql
```
#### Configure the database
Configure the database accordingly in the ```MySQLConfig.json```  in the ```config``` folder.

### 2. Train and test
You need to place the files related to Roberta in the ```roberta``` folder under the ```model``` directory first. Specifically, the following files are required:
* config.json
* dict.txt
* merges.txt
* pytorch_model.bin
* tokenizer.json
* vocab.json
You can obtain these files from ```huggingface.co```.
  
In ```main.py```, you can configure the parameters related to training. The specific correspondence is as follows:

parameter | explanation
|:----:|:----|
mode | This parameter is used to indicate whether to perform training or testing.
model_type | This parameter is used to indicate the type of model that will be trained.
random_seed | This parameter is used to randomly shuffle the dataset.
division | This parameter is used to split the dataset into training set, validation set, and test set.
base_model | This parameter is used to specify the type of the masked language model (MLM).
max_epoch_num | This parameter is used to indicate the maximum number of training epochs.
project | This parameter is used to indicate the project that the issue used for training belongs to, and options include ARIES, ZOOKEEPER, MAVEN, and HADOOP.
component_range |  This parameter is used to indicate the number of categories for multi-label classification.
output_k | This parameter is used to indicate how to output the evaluation results of recall@k and top@k.
use_time_weight | This parameter is used to indicate whether to use a time decay function.
param_type | This parameter is used to indicate the type of time decay function.
granularity | This parameter is used to indicate the granularity of calculating the time decay function.
prompt_type | This parameter is used to indicate how to construct the prompt.

After completing the above configurations, you can run ```main.py``` to start the training process.
```
python main.py
```




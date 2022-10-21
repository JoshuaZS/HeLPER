# HeLPER
the code for HeLpER: Knowledge graph completion based on hyper-network convolution and enhanced entity expression
this code is modified based on the code of https://github.com/MIRALab-USTC/GCN4KGC/
we also implemented ConvE, LowFER, HypER, TuckER in our model
avaible dataset include FB15k-237, WN18RR and YAGO3-10
## description of options 
+ --name: the name of the model, start with one of ConvE, LowFER, HypER, TuckER(case ignored)
+ --data: the dataset to be used, one of 'FB15k-237', 'wn18rr' and 'yago'
+ --gpu: the gpu to use, -1 for cpu and 0 for single gpu
+ --num-workers: Number of processes to construct batches, set to 0 when memory capacity is low
+ --restore: load the model specified by '--name'
## example
+ `python run.py --name helper_test --data FB15k-237`
+ `python run.py --name lowfer_test --data wn18rr`
+ `python run.py --name helper_test --data yago`

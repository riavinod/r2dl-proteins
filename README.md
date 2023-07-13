## Reprogramming Pretrained Language Models for Protein Sequence Representation Learning


### Abstract
Machine Learning-guided solutions for protein learning tasks have made
significant headway in recent years. However, success in scientific discovery
tasks is limited by the accessibility of well-defined and labeled in-domain
data. To tackle the low-data constraint, recent adaptions of deep learning
models pretrained on millions of protein sequences have shown promise; however,
the construction of such domain-specific large-scale model is computationally
expensive. Here, we propose Representation Learning via Dictionary Learning
(R2DL), an end-to-end representation learning framework in which we reprogram
deep models for alternate-domain tasks that can perform well on protein
property prediction with significantly fewer training samples. R2DL reprograms
a pretrained English language model to learn the embeddings of protein
sequences, by learning a sparse linear mapping between English and protein
sequence vocabulary embeddings. Our model can attain better accuracy and
significantly improve the data efficiency by up to $10^5$ times over the
baselines set by pretrained and standard supervised methods. To this end, we
reprogram an off-the-shelf pre-trained English language transformer and
benchmark it on a set of protein physicochemical prediction tasks (secondary
structure, stability, homology, stability) as well as on a biomedically
relevant set of protein function prediction tasks (antimicrobial, toxicity,
antibody affinity).

### Requirements
- Python 3.7.4
- Pytorch 1.9.0

### Installation
We recommend using anaconda to set up the virtual environment and then pip to download the required packages. This will be updated in the future to use Poetry and resolve and package dependencies.
```
conda create --name r2dl --file requirements.txt

```
### Downloading datasets
Data is not stored in this repository due to size concerns, but the protein sequence data for downstream tasks can be downloaded from the links below. We do not require the english language data since we use pretrained models and their embeddings. See the references in the manuscript for original references since some data is re-hosted in different repositories.

- Secondary structure: http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/secondary_structure.tar.gz
- Homology: http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/remote_homology.tar.gz
- Stability: http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/stability.tar.gz
- Solubility: https://www.dropbox.com/s/vgdqcl4vzqm9as0/deeploc_per_protein_train.csv?dl=1
- Antimicrobial: https://github.com/IBM/controlled-peptide-generation/tree/master/data_processing/data/ampep
- Toxicity: https://webs.iiitd.edu.in/raghava/toxinpred/dataset.php
- Antibody affinity: https://github.com/Tessier-Lab-UMich/Emi_Pareto_Opt_ML/blob/main/emi_binding.csv
- Protein-Protein Interaction: https://github.com/houzl3416/EDLMPPI/tree/main/datasets
- English data (sentiment analysis): https://drive.google.com/file/d/1Qwy2OKj4JCjkFyQBy4HZK3hYXPwp-JpV/view?usp=sharing

### Pretrained models
We use the Hugging Face library to obtain pretrained models. You can choose your own pretrained model by loading any model from the transformers package and passing it in 'model' flag during training R2DL.
```
!pip install transformers
```
With the following command, you can download the weights of 6 instances of full-pretrained BERT models as found in the modeling.py file. You can use BertForSequenceClassification for sequence classification tasks and finetuning.
```
!pip install pytorch-pretrained-bert
```
You can load an save any model from the Hugging Face repository:
```
from transformers import BertTokenizer, BertModel
>>> import torch

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = BertModel.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)
>>> prediction_logits = outputs.prediction_logits

>>> last_hidden_states = outputs.last_hidden_state
```

Here is an example of running a BERT instance on your data for token classification task on the source (english vocabulary) data: 
```
from transformers import BertTokenizer, BertForTokenClassification
>>> import torch

>>> tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
>>> model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
... )

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_token_class_ids = logits.argmax(-1)

>>> # Note that tokens are classified rather then input words which means that
>>> # there might be more predicted token classes than words.
>>> # Multiple token classes might account for the same word
>>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
>>> predicted_tokens_classes
['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] 
```
Our goal is to reprogram this BERT instance such that it can correctly classify protein sequences with predicted token classes that look like: 
```
['toxic', 'non-toxic', 'toxic', 'toxic', 'toxic', 'non-toxic', 'non-toxic', 'non-toxic', 'non-toxic', 'non-toxic', 'non-toxic', 'non-toxic', 'non-toxic'] 
```

Get token embeddings from the pretrained language model. This size of the embeddings will change based on the pretrained model chosen.
```
python3 get_pretrained_embeddings.py --model 'tiny-bert' --data 'secondary_structure'
```

### Standard supervised baseline models



### Training R2DL
Select the pretrained model, downstream task, k-SVD iterations and batch and epoch sizes. By default, the victim model is set to a transformer bert-base. You can change this by addressing the TODOs in train_r2dl.py and get_embeddings.py. Note that k-SVD iterations are very expensive to compute at each timestep so we do not recommend training R2DL with over 20,000 k-SVD iterations. We use a cross entropy loss for every instance of R2DL. Use the --help option to see a list of hyperparameters that can be set during training.
```
python3 train_r2dl.py --source_dataset=<english_language_task_dataset> --target_dataset=<downstream_protein_task_dataset> --classifier_type=<victim_model>
```

### Downstream task prediction

Generate the source model embeddings of a downstream protein task dataset using get_embeddings.py. Then set the target_dataset path to the protein task data embeddings as following:

```
python3 train_r2dl.py --source_dataset=<english_language_task_dataset> --target_dataset=<downstream_protein_task_dataset> --classifier_type=<victim_model>
```


### Reproducing the results
Hyper-parameter details to reproduce the results can be found in the supplementary material of our paper.



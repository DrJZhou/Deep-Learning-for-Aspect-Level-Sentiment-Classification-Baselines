# Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines
The public state-of-the-art methods for deep learning based ASC

# Authors
 - Jie Zhou;Jimmy Huang;Qin Chen, Tingting Wang, Qinmin Vivian Hu, and Liang He

## Environmental Requirement
- Python 3.6
- Pytorch 4.0
- sklearn
- numpy

## Introduction of this work
### models/  The deep learning model for ASC
 - ContextAvg
 - AEContextAvg
 - LSTM
 - GRU
 - BiLSTM
 - BiGRU
 - TD-LSTM
 - TC-LSTM
### data/  Store the data
 - data_orign: the original datasets, including SemEval2014-Task4, SemEval2015-Task12, SemEval2016-Task5, Twitter, Sentihood, Michell, MPQA.
 - data_processed: the datasets after processing
 - store: Store the embedding of words, like GloVe.
 - tmp: store the temporary files.
### data_processing/  Processing the data
 - SemEval2014-Laptop: Processe the Laptop14 dataset.
 - SemEval2014-Resturant: Process the Restaurants14 dataset.
 - SemEval2015-Resturant: Process the Restaurants15 dataset.
 - SemEval2016-Resturant: Process the Restaurants16 dataset.
 - Twitter: Process the Twitter dataset.
 - MPQA: Process the MPQA dataset.
 - Michell-en: Process the Michell-en dataset.
 - Sentihood: Process the Sentihood dataset.
### layers/ Basic units of deep learning models
 - Attention: Attention units, including ''Contact Attention", ''General Attention" and ''Dot-Product Attention".
 - Dynamic_RNN: Basic RNN, LSTM and GRU models.
 - SqueezeEmbedding: Squeeze the embeddings of words.
### results/ Store the results
 - log: Store the log of the models.
 - ans: Store the answer of the models
 - attention_weight: Store the weight of the attentions.


## Contacts
- jzhou@ica.stc.sh.cn

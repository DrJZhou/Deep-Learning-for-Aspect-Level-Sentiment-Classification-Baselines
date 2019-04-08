# Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines
The public state-of-the-art methods for deep learning based ASC

## Authors
 - Jie Zhou;Jimmy Huang;Qin Chen, Tingting Wang, Qinmin Vivian Hu, and Liang He

## Environmental Requirement
- Python 3.6
- Pytorch 4.0
- sklearn
- numpy

## Introduction of this work
### models/  The deep learning model for ASC
 - ContextAvg: the average of the word embeddings is fed to a softmax layer for sentiment prediction, which was adopted as a baseline in [1].
 - AEContextAvg: the concatenation of the average of the word embeddings and the average of the aspect vectors is fed to a softmax layer for sentiment prediction, which was adopted as a baseline in [1].
 - LSTM: the last hidden vector obtained by LSTM [2] is used for sentence representation and sentiment prediction. 
 - GRU: the last hidden vector obtained by GRU [3] is used for sentence representation and sentiment prediction.
 - BiLSTM: the concatenation of last hidden vectors obtained by BiLSTM is used for sentence representation and sentiment prediction.
 - BiGRU: the concatenation of last hidden vectors obtained by BiGRU is used for sentence representation and sentiment prediction.
 - TD-LSTM: a target-dependent LSTM model which modeled the preceding and following contexts surrounding the target for sentiment classification \cite{tang2016effective}.
 - TC-LSTM: this model extends TD-LSTM by incorporating an target con- nection component, which explicitly utilizes the connections between target word and each context word when composing the representation of a sentence. \cite{tang2016effective}.
 - AT-LSTM: it uses an LSTM to model the sentence and a basic attention mechanism is applied for sentence representation and sentiment prediction. \cite{wang2016attention}.
 - AT-GRU: it uses a GRU to model the sentence and a basic attention mechanism is applied for sentence representation and sentiment prediction. \cite{wang2016attention}.
 - AT-BiLSTM: it uses a BiLSTM to model the sentence and a basic attention mechanism is applied for sentence representation and sentiment prediction. \cite{wang2016attention}.
 - AT-BiGRU: it uses a BiGRU to model the sentence and a basic attention mechanism is applied for sentence representation and sentiment prediction. \cite{wang2016attention}.
 - ATAE-LSTM: the aspect representation is integrated into attention-based LSTM for sentence representation and sentiment prediction \cite{wang2016attention}.
 - ATAE-GRU: the aspect representation is integrated into attention-based GRU for sentence representation and sentiment prediction.
 - ATAE-BiLSTM: the aspect representation is integrated into attention-based BiLSTM for sentence representation and sentiment prediction.
 - ATAE-BiGRU: the aspect representation is integrated into attention-based BiGRU for sentence representation and sentiment prediction.
 - IAN: the attentions in the context and aspect were learned interactively for context and aspect representation \cite{mainteractive}. 
 - LCRS: it contains three LSTMs, i.e., left-, center- and right- LSTM, respectively modeling the three parts of a review (left context, aspect and right context) \cite{zheng2018left}.
 - CNN: The sentence representation obtained by CNN \cite{lecun1995convolutional} is used for ASC.
 - GCAE: it has two separate convolutional layers on the top of the embedding layer, whose outputs are combined by gating units \cite{xue2018aspect}.
 - MemNet: the content and position of the aspect is incorporated into a deep memory network \cite{tang2016aspect}.
 - RAM: a multi-layer architecture where each layer contains an attention-based aggregation of word features and a GRU cell to learn the sentence representation \cite{chen2017recurrent}. 
 - CABASC: two novel attention mechanisms, namely sentence-level content attention mechanism and context attention mechanism are introduced in a memory network to tackle the semantic-mismatch problem \cite{liu2018content}.
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

## References
[1] Tang D, Qin B, Liu T. Aspect Level Sentiment Classification with Deep Memory Network[C]//Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. 2016: 214-224.<br>
[2] Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural computation, 1997, 9(8): 1735-1780.<br>
[3] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.  <br>
[4] Tang D, Qin B, Feng X, et al. Effective LSTMs for Target-Dependent Sentiment Classification[C]//Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016: 3298-3307.  <br>

## Contacts
- jzhou@ica.stc.sh.cn

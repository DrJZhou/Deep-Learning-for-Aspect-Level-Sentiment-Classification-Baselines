# Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines
The public state-of-the-art methods for deep learning based ASC

## Note
All the codes and datasets for ASC will be released later.

## Authors
 - Jie Zhou;Jimmy Huang;Qin Chen, Tingting Wang, Qinmin Vivian Hu, and Liang He

## Environmental Requirement
- Python 3.6
- Pytorch 4.0
- sklearn
- numpy

## Introduction of this work
### models/:  The deep learning model for ASC
 - ContextAvg: the average of the word embeddings is fed to a softmax layer for sentiment prediction, which was adopted as a baseline in [1].
 - AEContextAvg: the concatenation of the average of the word embeddings and the average of the aspect vectors is fed to a softmax layer for sentiment prediction, which was adopted as a baseline in [1].
 - LSTM: the last hidden vector obtained by LSTM [2] is used for sentence representation and sentiment prediction. 
 - GRU: the last hidden vector obtained by GRU [3] is used for sentence representation and sentiment prediction.
 - BiLSTM: the concatenation of last hidden vectors obtained by BiLSTM is used for sentence representation and sentiment prediction.
 - BiGRU: the concatenation of last hidden vectors obtained by BiGRU is used for sentence representation and sentiment prediction.
 - TD-LSTM: a target-dependent LSTM model which modeled the preceding and following contexts surrounding the target for sentiment classification [4].
 - TC-LSTM: this model extends TD-LSTM by incorporating an target con- nection component, which explicitly utilizes the connections between target word and each context word when composing the representation of a sentence. [4].
 - AT-LSTM: it uses an LSTM to model the sentence and a basic attention mechanism is applied for sentence representation and sentiment prediction. [5].
 - AT-GRU: it uses a GRU to model the sentence and a basic attention mechanism is applied for sentence representation and sentiment prediction. [5].
 - AT-BiLSTM: it uses a BiLSTM to model the sentence and a basic attention mechanism is applied for sentence representation and sentiment prediction. [5].
 - AT-BiGRU: it uses a BiGRU to model the sentence and a basic attention mechanism is applied for sentence representation and sentiment prediction. [5].
 - ATAE-LSTM: the aspect representation is integrated into attention-based LSTM for sentence representation and sentiment prediction [5].
 - ATAE-GRU: the aspect representation is integrated into attention-based GRU for sentence representation and sentiment prediction.
 - ATAE-BiLSTM: the aspect representation is integrated into attention-based BiLSTM for sentence representation and sentiment prediction.
 - ATAE-BiGRU: the aspect representation is integrated into attention-based BiGRU for sentence representation and sentiment prediction.
 - IAN: the attentions in the context and aspect were learned interactively for context and aspect representation [6]. 
 - LCRS: it contains three LSTMs, i.e., left-, center- and right- LSTM, respectively modeling the three parts of a review (left context, aspect and right context) [7].
 - CNN: The sentence representation obtained by CNN [8] is used for ASC.
 - GCAE: it has two separate convolutional layers on the top of the embedding layer, whose outputs are combined by gating units [9].
 - MemNet: the content and position of the aspect is incorporated into a deep memory network [10].
 - RAM: a multi-layer architecture where each layer contains an attention-based aggregation of word features and a GRU cell to learn the sentence representation [11]. 
 - CABASC: two novel attention mechanisms, namely sentence-level content attention mechanism and context attention mechanism are introduced in a memory network to tackle the semantic-mismatch problem [12].
### data/:  Store the data
 - data_orign: the original datasets, including SemEval2014-Task4, SemEval2015-Task12, SemEval2016-Task5, Twitter, Sentihood, Michell, MPQA.
 - data_processed: the datasets after processing
 - store: Store the embedding of words, like GloVe.
 - tmp: store the temporary files.
### data_processing/:  Processing the data
 - SemEval2014-Laptop: Processe the Laptop14 dataset.
 - SemEval2014-Resturant: Process the Restaurants14 dataset.
 - SemEval2015-Resturant: Process the Restaurants15 dataset.
 - SemEval2016-Resturant: Process the Restaurants16 dataset.
 - Twitter: Process the Twitter dataset.
 - MPQA: Process the MPQA dataset.
 - Michell-en: Process the Michell-en dataset.
 - Sentihood: Process the Sentihood dataset.
### layers/: Basic units of deep learning models
 - Attention: Attention units, including ''Contact Attention", ''General Attention" and ''Dot-Product Attention".
 - Dynamic_RNN: Basic RNN, LSTM and GRU models.
 - SqueezeEmbedding: Squeeze the embeddings of words.
### results/: Store the results
 - log: Store the log of the models.
 - ans: Store the answer of the models
 - attention_weight: Store the weight of the attentions.
 - model: Store the trained models.

## Statics of the performance of the existing works for deep learning based ASC
|            Method           | Restaurants14 |          | Laptop14 |          | Restaurants15 |          | Restaurants16 |          |  Twitter |          |
|:---------------------------:|:-------------:|:--------:|:--------:|:--------:|:-------------:|:--------:|:-------------:|:--------:|:--------:|:--------:|
|                             |    Accuracy   | Marco-F1 | Accuracy | Marco-F1 |    Accuracy   | Marco-F1 |    Accuracy   | Marco-F1 | Accuracy | Marco-F1 |
|        RecNN for ASC        |               |          |          |          |               |          |               |          |          |          |
|            AdaRNN           |       -       |     -    |     -    |     -    |       -       |     -    |       -       |     -    |   66.30  |   65.90  |
|          PhraseRNN          |     66.20     |     -    |     -    |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|         RNN for ASC         |               |          |          |          |               |          |               |          |          |          |
|             GRNN            |       -       |     -    |     -    |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|           TD-LSTM           |       -       |     -    |     -    |     -    |       -       |     -    |       -       |     -    |   70.80  |   69.00  |
|           TC-LSTM           |       -       |     -    |     -    |     -    |       -       |     -    |       -       |     -    |   71.50  |   69.50  |
|           AE-LSTM           |     76.60     |     -    |   68.90  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|            H-LSTM           |       -       |     -    |     -    |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
| Attention-based RNN for ASC |               |          |          |          |               |          |               |          |          |          |
|          ATAE-LSTM          |     77.20     |     -    |   68.70  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|           AB-LSTM           |       -       |     -    |     -    |     -    |       -       |     -    |       -       |     -    |   72.60  |   72.20  |
|         BILSTM-ATT-G        |       -       |     -    |     -    |     -    |       -       |     -    |       -       |     -    |   73.60  |   72.10  |
|             IAN             |     78.60     |     -    |   72.10  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|        AF-LSTM(CONV)        |     75.44     |     -    |   68.81  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|             HEAT            |       -       |     -    |     -    |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|      Sentic LSTM+TA+SA      |       -       |     -    |     -    |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|          PRET+MULT          |     79.11     |   79.73  |   71.15  |   67.46  |     81.30     |   68.74  |     85.58     |   79.76  |     -    |     -    |
|             PBAN            |     81.16     |     -    |   74.12  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|      LSTM+SynATT+TarRep     |     80.63     |   71.32  |   71.94  |   69.23  |     81.67     |   66.05  |     84.61     |   67.45  |     -    |     -    |
|             MGAN            |     81.25     |   71.94  |   75.39  |   72.47  |       -       |     -    |       -       |     -    |   72.54  |   70.81  |
|  Inter-Aspect Dependencies  |     79.00     |     -    |   72.50  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|           AOA-LSTM          |     81.20     |     -    |   74.50  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|           LCR-Rot           |     81.34     |     -    |   75.24  |     -    |       -       |     -    |       -       |     -    |   72.69  |     -    |
|    Word\&Clause-Level ATT   |       -       |     -    |     -    |     -    |     80.90     |   68.50  |       -       |     -    |     -    |     -    |
|         CNN for ASC         |               |          |          |          |               |          |               |          |          |          |
|             GCAE            |     77.28     |     -    |   69.14  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|            PF-CNN           |     79.20     |     -    |   70.06  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|         Conv-Memnet         |     78.26     |   68.38  |   76.37  |   72.10  |       -       |     -    |       -       |     -    |   72.11  |   70.80  |
|             TNet            |     80.69     |   71.27  |   76.54  |   71.75  |       -       |     -    |       -       |     -    |   74.97  |   73.60  |
|    Memory Network for ASC   |               |          |          |          |               |          |               |          |          |          |
|            MemNet           |     80.95     |     -    |   72.21  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|           DyMemNN           |       -       |   58.82  |     -    |   60.11  |       -       |     -    |       -       |     -    |     -    |     -    |
|             RAM             |     80.23     |   70.80  |   74.49  |   71.35  |       -       |     -    |       -       |     -    |   69.36  |   73.85  |
|             CEA             |     80.98     |     -    |   72.88  |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|             DAuM            |     82.32     |   71.45  |   74.45  |   70.16  |       -       |     -    |       -       |     -    |   72.14  |   60.24  |
|             IARM            |     80.00     |     -    |   73.8   |     -    |       -       |     -    |       -       |     -    |     -    |     -    |
|             TMNs            |       -       |   68.84  |     -    |   67.23  |       -       |     -    |       -       |     -    |     -    |     -    |
|            Cabasc           |     80.89     |     -    |   75.07  |     -    |       -       |     -    |       -       |     -    |   71.53  |     -    |



## References
[1] Tang D, Qin B, Liu T. Aspect Level Sentiment Classification with Deep Memory Network[C]//Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. 2016: 214-224.<br>
[2] Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural computation, 1997, 9(8): 1735-1780.<br>
[3] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.  <br>
[4] Tang D, Qin B, Feng X, et al. Effective LSTMs for Target-Dependent Sentiment Classification[C]//Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016: 3298-3307.  <br>
[5] Wang Y, Huang M, Zhao L. Attention-based LSTM for aspect-level sentiment classification[C]//Proceedings of the 2016 conference on empirical methods in natural language processing. 2016: 606-615.  <br>
[6] Ma D, Li S, Zhang X, et al. Interactive attention networks for aspect-level sentiment classification[C]//Proceedings of the 26th International Joint Conference on Artificial Intelligence. AAAI Press, 2017: 4068-4074.  <br>
[7] Zheng S, Xia R. Left-Center-Right Separated Neural Network for Aspect-based Sentiment Analysis with Rotatory Attention[J]. arXiv preprint arXiv:1802.00892, 2018.  <br>
[8] LeCun Y, Bengio Y. Convolutional networks for images, speech, and time series[J]. The handbook of brain theory and neural networks, 1995, 3361(10): 1995.  <br>
[9] Xue W, Li T. Aspect Based Sentiment Analysis with Gated Convolutional Networks[C]//Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018: 2514-2523.  <br>
[10] Tang D, Qin B, Liu T. Aspect Level Sentiment Classification with Deep Memory Network[C]//Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. 2016: 214-224.  <br>
[11] Chen P, Sun Z, Bing L, et al. Recurrent attention network on memory for aspect sentiment analysis[C]//Proceedings of the 2017 conference on empirical methods in natural language processing. 2017: 452-461.  <br>
[12] Liu Q, Zhang H, Zeng Y, et al. Content attention model for aspect based sentiment analysis[C]//Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018: 1023-1032.  

## Contacts
- jzhou@ica.stc.sh.cn

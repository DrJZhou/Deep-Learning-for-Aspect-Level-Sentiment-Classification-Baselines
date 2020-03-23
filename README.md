# Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines
The public state-of-the-art methods for deep learning based ASC. This is part of my survey paper "[Deep learning for Aspect-level Sentiment Classification: Survey, Vision and Challenges](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8726353)".

## Note
Click [here](http://ac.ica.stc.sh.cn/papers/ASC_DATA.zip) to download all the ASC datasets (including SemEval 2014, SemEval 2015, SemEval 2016, Twitter, Sentihood, MPQA, Michell and manually-annoted-Hotel).

## Authors
 - Jie Zhou (jzhou@ica.stc.sh.cn), Jimmy Huang (jhuang@yorku.ca), Qin Chen, Tingting Wang, Qinmin Vivian Hu, and Liang He
 If you find this repo useful, please consider citing (no obligation at all):
 ```
 @article{zhou2019deep,
   title={Deep learning for aspect-level sentiment classification: Survey, vision, and challenges},
   author={Zhou, Jie and Huang, Jimmy Xiangji and Chen, Qin and Hu, Qinmin Vivian and Wang, Tingting and He, Liang},
   journal={IEEE Access},
   volume={7},
   pages={78454--78483},
   year={2019},
   publisher={IEEE}
 }
```
## Environmental Requirement
- Python 3.6
- Pytorch 0.4.0
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


## The results of our implemented models
### The results of dataset Restaurants14
|              | Accuracy |   Macro   |        |       |   Micro   |        |       | Precision |       |       | Recall |       |       |   F1  |       |       |
|--------------|:--------:|:---------:|:------:|:-----:|:---------:|:------:|:-----:|:---------:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|              |          | Precision | Recall |   F1  | Precision | Recall |   F1  |    Neg.   |  Neu. |  Pos. |  Neg.  |  Neu. |  Pos. |  Neg. |  Neu. |  Pos. |
| ContextAvg   |   73.48  |   62.92   |  58.44 | 59.58 |   73.48   |  73.48 | 73.48 |   56.48   | 51.79 | 80.49 |  55.61 | 29.59 | 90.11 | 56.04 | 37.66 | 85.03 |
| AEContextAvg |   75.27  |   66.30   |  61.47 | 63.10 |   75.27   |  75.27 | 75.27 |   62.09   | 55.47 | 81.36 |  57.65 | 36.22 | 90.52 | 59.79 | 43.83 | 85.70 |
| LSTM         |   77.23  |   67.54   |  64.34 | 65.51 |   77.23   |  77.23 | 77.23 |   63.35   | 54.55 | 84.73 |  61.73 | 39.80 | 91.48 | 62.53 | 46.02 | 87.98 |
| GRU          |   78.75  |   70.51   |  65.61 | 67.11 |   78.75   |  78.75 | 78.75 |   67.36   | 59.84 | 84.35 |  66.33 | 37.24 | 93.27 | 66.84 | 45.91 | 88.58 |
| BiGRU        |   77.14  |   67.61   |  63.55 | 65.15 |   77.14   |  77.14 | 77.14 |   64.94   | 53.69 | 84.19 |  57.65 | 40.82 | 92.17 | 61.08 | 46.38 | 88.00 |
| BiLSTM       |   78.30  |   69.11   |  66.01 | 67.12 |   78.30   |  78.30 | 78.30 |   65.13   | 56.64 | 85.55 |  64.80 | 41.33 | 91.90 | 64.96 | 47.79 | 88.61 |
| TD-LSTM      |   78.66  |   70.84   |  67.56 | 68.98 |   78.66   |  78.66 | 78.66 |   72.88   | 54.55 | 85.09 |  65.82 | 45.92 | 90.93 | 69.17 | 49.86 | 87.92 |
| TC-LSTM      |   77.41  |   69.06   |  65.18 | 66.72 |   77.41   |  77.41 | 77.41 |   67.78   | 55.70 | 83.69 |  62.24 | 42.35 | 90.93 | 64.89 | 48.12 | 87.16 |
| AT-LSTM      |   78.04  |   70.84   |  61.52 | 63.37 |   78.04   |  78.04 | 78.04 |   70.06   | 61.25 | 81.23 |  63.27 | 25.00 | 96.29 | 66.49 | 35.51 | 88.12 |
| AT-GRU       |   78.30  |   70.74   |  64.76 | 66.58 |   78.30   |  78.30 | 78.30 |   67.91   | 61.21 | 83.11 |  64.80 | 36.22 | 93.27 | 66.32 | 45.51 | 87.90 |
| AT-BiGRU     |   77.77  |   69.51   |  64.74 | 66.18 |   77.77   |  77.77 | 77.77 |   65.13   | 59.84 | 83.56 |  64.80 | 37.24 | 92.17 | 64.96 | 45.91 | 87.66 |
| AT-BiLSTM    |   78.84  |   72.84   |  63.67 | 65.66 |   78.84   |  78.84 | 78.84 |   68.45   | 67.82 | 82.27 |  65.31 | 30.10 | 95.60 | 66.84 | 41.70 | 88.44 |
| ATAE-GRU     |   76.79  |   68.68   |  63.49 | 65.32 |   76.79   |  76.79 | 76.79 |   69.49   | 54.62 | 81.92 |  62.76 | 36.22 | 91.48 | 65.95 | 43.56 | 86.44 |
| ATAE-LSTM    |   76.79  |   67.93   |  62.74 | 63.72 |   76.79   |  76.79 | 76.79 |   64.53   | 57.00 | 82.25 |  66.84 | 29.08 | 92.31 | 65.66 | 38.51 | 86.99 |
| ATAE-BiGRU   |   76.34  |   65.95   |  63.26 | 63.82 |   76.34   |  76.34 | 76.34 |   63.77   | 50.41 | 83.67 |  67.35 | 31.63 | 90.80 | 65.51 | 38.87 | 87.09 |
| ATAE-BiLSTM  |   75.98  |   67.01   |  61.71 | 63.43 |   75.98   |  75.98 | 75.98 |   66.29   | 53.28 | 81.46 |  60.20 | 33.16 | 91.76 | 63.10 | 40.88 | 86.30 |
| IAN          |   76.70  |   68.29   |  63.69 | 65.12 |   76.70   |  76.70 | 76.70 |   64.25   | 58.06 | 82.57 |  63.27 | 36.73 | 91.07 | 63.75 | 45.00 | 86.61 |
| LCRS         |   76.25  |   68.71   |  60.85 | 63.03 |   76.25   |  76.25 | 76.25 |   69.82   | 56.44 | 79.88 |  60.20 | 29.08 | 93.27 | 64.66 | 38.38 | 86.06 |
| CNN          |   75.18  |   68.45   |  58.44 | 60.25 |   75.18   |  75.18 | 75.18 |   60.44   | 65.79 | 79.12 |  56.12 | 25.51 | 93.68 | 58.20 | 36.76 | 85.79 |
| GCAE         |   77.41  |   68.58   |  64.80 | 65.06 |   77.41   |  77.41 | 77.41 |   64.86   | 57.43 | 83.44 |  73.47 | 29.59 | 91.35 | 68.90 | 39.06 | 87.21 |
| MemNet       |   73.39  |   62.74   |  61.13 | 61.09 |   73.39   |  73.39 | 73.39 |   52.56   | 52.38 | 83.29 |  62.76 | 33.67 | 86.95 | 57.21 | 40.99 | 85.08 |
| RAM          |   77.41  |   68.38   |  65.67 | 66.76 |   77.41   |  77.41 | 77.41 |   67.20   | 53.25 | 84.68 |  64.80 | 41.84 | 90.38 | 65.97 | 46.86 | 87.44 |
| CABASC       |   77.68  |   69.01   |  67.18 | 68.02 |   77.68   |  77.68 | 77.68 |   65.59   | 55.68 | 85.75 |  62.24 | 50.00 | 89.29 | 63.87 | 52.69 | 87.48 |

### The results of Laptop14
|              | Accuracy |   Macro   |        |       |   Micro   |        |       | Precision |       |       | Recall |       |       |   F1  |       |       |
|--------------|:--------:|:---------:|:------:|:-----:|:---------:|:------:|:-----:|:---------:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|              |          | Precision | Recall |   F1  | Precision | Recall |   F1  |    Neg.   |  Neu. |  Pos. |  Neg.  |  Neu. |  Pos. |  Neg. |  Neu. |  Pos. |
| ContextAvg   |   66.93  |   63.47   |  59.98 | 58.19 |   66.93   |  66.93 | 66.93 |   46.41   | 67.65 | 76.35 |  65.62 | 27.22 | 87.10 | 54.37 | 38.82 | 81.37 |
| AEContextAvg |   66.46  |   61.64   |  59.56 | 58.04 |   66.46   |  66.46 | 66.46 |   47.40   | 61.54 | 75.97 |  64.06 | 28.40 | 86.22 | 54.49 | 38.87 | 80.77 |
| LSTM         |   66.14  |   62.37   |  60.20 | 55.35 |   66.14   |  66.14 | 66.14 |   48.08   | 62.79 | 76.23 |  78.12 | 15.98 | 86.51 | 59.52 | 25.47 | 81.04 |
| GRU          |   67.71  |   64.31   |  61.50 | 58.60 |   67.71   |  67.71 | 67.71 |   49.47   | 66.67 | 76.80 |  73.44 | 23.67 | 87.39 | 59.12 | 34.93 | 81.76 |
| BiGRU        |   69.44  |   65.61   |  63.83 | 61.49 |   69.44   |  69.44 | 69.44 |   49.22   | 67.11 | 80.49 |  74.22 | 30.18 | 87.10 | 59.19 | 41.63 | 83.66 |
| BiLSTM       |   68.81  |   63.41   |  63.56 | 62.09 |   68.81   |  68.81 | 68.81 |   50.28   | 59.05 | 80.90 |  69.53 | 36.69 | 84.46 | 58.36 | 45.26 | 82.64 |
| TD-LSTM      |   68.50  |   62.66   |  62.98 | 61.87 |   68.50   |  68.50 | 68.50 |   47.70   | 57.63 | 82.66 |  64.84 | 40.24 | 83.87 | 54.97 | 47.39 | 83.26 |
| TC-LSTM      |   67.08  |   62.02   |  62.66 | 61.11 |   67.08   |  67.08 | 67.08 |   46.52   | 57.76 | 81.79 |  67.97 | 39.64 | 80.35 | 55.24 | 47.02 | 81.07 |
| AT-LSTM      |   69.44  |   64.23   |  65.02 | 63.16 |   69.44   |  69.44 | 69.44 |   51.91   | 58.88 | 81.90 |  74.22 | 37.28 | 83.58 | 61.09 | 45.65 | 82.73 |
| AT-GRU       |   70.85  |   66.57   |  66.21 | 63.58 |   70.85   |  70.85 | 70.85 |   54.21   | 64.63 | 80.87 |  80.47 | 31.36 | 86.80 | 64.78 | 42.23 | 83.73 |
| AT-BiGRU     |   69.28  |   64.44   |  64.36 | 63.28 |   69.28   |  69.28 | 69.28 |   48.86   | 62.61 | 81.84 |  67.19 | 42.60 | 83.28 | 56.58 | 50.70 | 82.56 |
| AT-BiLSTM    |   71.94  |   66.36   |  66.80 | 66.42 |   71.94   |  71.94 | 71.94 |   55.48   | 59.06 | 84.55 |  63.28 | 52.07 | 85.04 | 59.12 | 55.35 | 84.80 |
| ATAE-GRU     |   69.75  |   64.43   |  63.46 | 62.45 |   69.75   |  69.75 | 69.75 |   52.76   | 61.22 | 79.31 |  67.19 | 35.50 | 87.68 | 59.11 | 44.94 | 83.29 |
| ATAE-LSTM    |   67.40  |   65.16   |  62.18 | 58.47 |   67.40   |  67.40 | 67.40 |   47.39   | 69.64 | 78.44 |  78.12 | 23.08 | 85.34 | 59.00 | 34.67 | 81.74 |
| ATAE-BiGRU   |   70.38  |   67.00   |  66.20 | 64.12 |   70.38   |  70.38 | 70.38 |   49.25   | 68.82 | 82.95 |  76.56 | 37.87 | 84.16 | 59.94 | 48.85 | 83.55 |
| ATAE-BiLSTM  |   70.53  |   66.84   |  65.99 | 63.43 |   70.53   |  70.53 | 70.53 |   50.75   | 67.47 | 82.30 |  78.91 | 33.14 | 85.92 | 61.77 | 44.44 | 84.07 |
| IAN          |   68.50  |   64.11   |  62.69 | 60.90 |   68.50   |  68.50 | 68.50 |   51.12   | 63.41 | 77.78 |  71.09 | 30.77 | 86.22 | 59.48 | 41.43 | 81.78 |
| LCRS         |   66.46  |   63.15   |  60.84 | 59.50 |   66.46   |  66.46 | 66.46 |   46.70   | 66.67 | 76.08 |  66.41 | 33.14 | 82.99 | 54.84 | 44.27 | 79.38 |
| CNN          |   66.93  |   65.95   |  59.91 | 57.75 |   66.93   |  66.93 | 66.93 |   45.99   | 76.36 | 75.51 |  67.19 | 24.85 | 87.68 | 54.60 | 37.50 | 81.14 |
| GCAE         |   65.83  |   60.95   |  60.34 | 59.20 |   65.83   |  65.83 | 65.83 |   43.72   | 60.00 | 79.14 |  62.50 | 37.28 | 81.23 | 51.45 | 45.99 | 80.17 |
| MemNet       |   64.42  |   59.08   |  59.36 | 58.01 |   64.42   |  64.42 | 64.42 |   43.01   | 54.87 | 79.35 |  62.50 | 36.69 | 78.89 | 50.96 | 43.97 | 79.12 |
| RAM          |   67.55  |   62.25   |  60.78 | 59.73 |   67.55   |  67.55 | 67.55 |   49.09   | 60.44 | 77.23 |  63.28 | 32.54 | 86.51 | 55.29 | 42.31 | 81.60 |
| CABASC       |   70.06  |   66.14   |  63.05 | 62.94 |   70.06   |  70.06 | 70.06 |   50.98   | 69.79 | 77.63 |  60.94 | 39.64 | 88.56 | 55.52 | 50.57 | 82.74 |

### The results of Restaurants15
|              | Accuracy |   Macro   |        |       |   Micro   |        |       | Precision |        |       | Recall |       |       |   F1  |       |       |
|--------------|:--------:|:---------:|:------:|:-----:|:---------:|:------:|:-----:|:---------:|:------:|:-----:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|              |          | Precision | Recall |   F1  | Precision | Recall |   F1  |    Neg.   |  Neu.  |  Pos. |  Neg.  |  Neu. |  Pos. |  Neg. |  Neu. |  Pos. |
| ContextAvg   |   72.31  |   65.35   |  50.18 | 49.80 |   72.31   |  72.31 | 72.31 |   74.91   |  50.00 | 71.15 |  58.67 |  2.22 | 89.65 | 65.80 |  4.26 | 79.34 |
| AEContextAvg |   73.37  |   49.87   |  50.22 | 49.17 |   73.37   |  73.37 | 73.37 |   78.54   |  0.00  | 71.06 |  59.25 |  0.00 | 91.41 | 67.55 |  0.00 | 79.96 |
| LSTM         |   77.99  |   51.77   |  54.60 | 53.14 |   77.99   |  77.99 | 77.99 |   75.49   |  0.00  | 79.84 |  78.32 |  0.00 | 85.46 | 76.88 |  0.00 | 82.55 |
| GRU          |   76.80  |   51.87   |  53.01 | 51.96 |   76.80   |  76.80 | 76.80 |   80.97   |  0.00  | 74.64 |  67.63 |  0.00 | 91.41 | 73.70 |  0.00 | 82.18 |
| BiGRU        |   77.28  |   51.48   |  53.70 | 52.44 |   77.28   |  77.28 | 77.28 |   76.99   |  0.00  | 77.46 |  72.54 |  0.00 | 88.55 | 74.70 |  0.00 | 82.63 |
| BiLSTM       |   78.34  |   52.34   |  54.36 | 53.14 |   78.34   |  78.34 | 78.34 |   79.18   |  0.00  | 77.84 |  72.54 |  0.00 | 90.53 | 75.72 |  0.00 | 83.71 |
| TD-LSTM      |   77.28  |   64.53   |  57.65 | 59.04 |   77.28   |  77.28 | 77.28 |   78.06   |  37.50 | 78.04 |  71.97 | 13.33 | 87.67 | 74.89 | 19.67 | 82.57 |
| TC-LSTM      |   74.44  |   62.62   |  53.41 | 54.10 |   74.44   |  74.44 | 74.44 |   76.51   |  37.50 | 73.84 |  65.90 |  6.67 | 87.67 | 70.81 | 11.32 | 80.16 |
| AT-LSTM      |   80.00  |   53.32   |  55.82 | 54.48 |   80.00   |  80.00 | 80.00 |   79.88   |  0.00  | 80.08 |  78.03 |  0.00 | 89.43 | 78.95 |  0.00 | 84.50 |
| AT-GRU       |   79.41  |   52.87   |  55.48 | 54.11 |   79.41   |  79.41 | 79.41 |   78.78   |  0.00  | 79.84 |  78.32 |  0.00 | 88.11 | 78.55 |  0.00 | 83.77 |
| AT-BiGRU     |   77.99  |   61.04   |  54.64 | 54.30 |   77.99   |  77.99 | 77.99 |   81.88   |  25.00 | 76.24 |  70.52 |  2.22 | 91.19 | 75.78 |  4.08 | 83.05 |
| AT-BiLSTM    |   79.88  |   53.11   |  55.88 | 54.45 |   79.88   |  79.88 | 79.88 |   78.41   |  0.00  | 80.93 |  79.77 |  0.00 | 87.89 | 79.08 |  0.00 | 84.27 |
| ATAE-GRU     |   78.58  |   85.80   |  55.40 | 54.88 |   78.58   |  78.58 | 78.58 |   79.33   | 100.00 | 78.06 |  75.43 |  2.22 | 88.55 | 77.33 |  4.35 | 82.97 |
| ATAE-LSTM    |   79.53  |   53.15   |  55.34 | 54.09 |   79.53   |  79.53 | 79.53 |   80.62   |  0.00  | 78.85 |  75.72 |  0.00 | 90.31 | 78.09 |  0.00 | 84.19 |
| ATAE-BiGRU   |   78.70  |   69.08   |  56.30 | 56.29 |   78.70   |  78.70 | 78.70 |   77.46   |  50.00 | 79.80 |  77.46 |  4.44 | 87.00 | 77.46 |  8.16 | 83.25 |
| ATAE-BiLSTM  |   78.34  |   52.21   |  54.59 | 53.29 |   78.34   |  78.34 | 78.34 |   78.14   |  0.00  | 78.47 |  75.43 |  0.00 | 88.33 | 76.76 |  0.00 | 83.11 |
| IAN          |   79.41  |   86.18   |  56.74 | 56.82 |   79.41   |  79.41 | 79.41 |   78.82   | 100.00 | 79.72 |  77.46 |  4.44 | 88.33 | 78.13 |  8.51 | 83.80 |
| LCRS         |   75.50  |   59.03   |  53.63 | 53.73 |   75.50   |  75.50 | 75.50 |   76.28   |  25.00 | 75.81 |  68.79 |  4.44 | 87.67 | 72.34 |  7.55 | 81.31 |
| CNN          |   69.35  |   64.71   |  47.47 | 46.93 |   69.35   |  69.35 | 69.35 |   77.46   |  50.00 | 66.67 |  47.69 |  2.22 | 92.51 | 59.03 |  4.26 | 77.49 |
| GCAE         |   76.33  |   57.61   |  53.89 | 53.32 |   76.33   |  76.33 | 76.33 |   75.07   |  20.00 | 77.76 |  73.99 |  2.22 | 85.46 | 74.53 |  4.00 | 81.43 |
| MemNet       |   76.45  |   71.93   |  56.34 | 57.97 |   76.45   |  76.45 | 76.45 |   76.90   |  62.50 | 76.39 |  70.23 | 11.11 | 87.67 | 73.41 | 18.87 | 81.64 |
| RAM          |   76.21  |   51.23   |  52.71 | 51.62 |   76.21   |  76.21 | 76.21 |   79.00   |  0.00  | 74.68 |  68.50 |  0.00 | 89.65 | 73.37 |  0.00 | 81.48 |
| CABASC       |   76.21  |   61.73   |  56.28 | 57.30 |   76.21   |  76.21 | 76.21 |   76.47   |  31.25 | 77.47 |  71.39 | 11.11 | 86.34 | 73.84 | 16.39 | 81.67 |

### The results of Restaurants16
|              | Accuracy |   Macro   |        |       |   Micro   |        |       | Precision |        |       | Recall |       |       |   F1  |       |       |
|--------------|:--------:|:---------:|:------:|:-----:|:---------:|:------:|:-----:|:---------:|:------:|:-----:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|              |          | Precision | Recall |   F1  | Precision | Recall |   F1  |    Neg.   |  Neu.  |  Pos. |  Neg.  |  Neu. |  Pos. |  Neg. |  Neu. |  Pos. |
| ContextAvg   |   80.56  |   49.61   |  52.56 | 51.04 |   80.56   |  80.56 | 80.56 |   61.82   |  0.00  | 87.01 |  66.67 |  0.00 | 91.00 | 64.15 |  0.00 | 88.96 |
| AEContextAvg |   80.79  |   49.87   |  52.99 | 51.37 |   80.79   |  80.79 | 80.79 |   62.33   |  0.00  | 87.26 |  68.14 |  0.00 | 90.83 | 65.11 |  0.00 | 89.01 |
| LSTM         |   83.12  |   76.89   |  59.24 | 58.23 |   83.12   |  83.12 | 83.12 |   64.23   |  75.00 | 91.43 |  81.86 |  6.82 | 89.03 | 71.98 | 12.50 | 90.22 |
| GRU          |   83.47  |   69.29   |  60.53 | 61.34 |   83.47   |  83.47 | 83.47 |   67.81   |  50.00 | 90.07 |  77.45 | 13.64 | 90.51 | 72.31 | 21.43 | 90.29 |
| BiGRU        |   83.47  |   77.36   |  59.66 | 61.39 |   83.47   |  83.47 | 83.47 |   68.18   |  75.00 | 88.91 |  73.53 | 13.64 | 91.82 | 70.75 | 23.08 | 90.34 |
| BiLSTM       |   82.54  |   52.03   |  53.70 | 52.81 |   82.54   |  82.54 | 82.54 |   69.70   |  0.00  | 86.38 |  67.65 |  0.00 | 93.45 | 68.66 |  0.00 | 89.78 |
| TD-LSTM      |   84.17  |   52.67   |  57.08 | 54.70 |   84.17   |  84.17 | 84.17 |   67.22   |  0.00  | 90.78 |  79.41 |  0.00 | 91.82 | 72.81 |  0.00 | 91.29 |
| TC-LSTM      |   82.07  |   55.80   |  54.73 | 54.06 |   82.07   |  82.07 | 82.07 |   66.82   |  12.50 | 88.07 |  70.10 |  2.27 | 91.82 | 68.42 |  3.85 | 89.90 |
| AT-LSTM      |   82.77  |   51.85   |  55.44 | 53.56 |   82.77   |  82.77 | 82.77 |   67.11   |  0.00  | 88.43 |  75.00 |  0.00 | 91.33 | 70.83 |  0.00 | 89.86 |
| AT-GRU       |   83.82  |   52.68   |  56.04 | 54.30 |   83.82   |  83.82 | 83.82 |   69.06   |  0.00  | 88.99 |  75.49 |  0.00 | 92.64 | 72.13 |  0.00 | 90.78 |
| AT-BiGRU     |   83.47  |   77.57   |  57.55 | 58.06 |   83.47   |  83.47 | 83.47 |   69.44   |  75.00 | 88.26 |  73.53 |  6.82 | 92.31 | 71.43 | 12.50 | 90.24 |
| AT-BiLSTM    |   82.89  |   85.01   |  58.75 | 56.88 |   82.89   |  82.89 | 82.89 |   63.20   | 100.00 | 91.84 |  83.33 |  4.55 | 88.38 | 71.88 |  8.70 | 90.08 |
| ATAE-GRU     |   82.31  |   51.16   |  55.11 | 53.01 |   82.31   |  82.31 | 82.31 |   64.41   |  0.00  | 89.09 |  74.51 |  0.00 | 90.83 | 69.09 |  0.00 | 89.95 |
| ATAE-LSTM    |   82.19  |   51.70   |  53.10 | 52.33 |   82.19   |  82.19 | 82.19 |   69.07   |  0.00  | 86.02 |  65.69 |  0.00 | 93.62 | 67.34 |  0.00 | 89.66 |
| ATAE-BiGRU   |   82.54  |   84.85   |  57.17 | 56.33 |   82.54   |  82.54 | 82.54 |   65.42   | 100.00 | 89.14 |  76.96 |  4.55 | 90.02 | 70.72 |  8.70 | 89.58 |
| ATAE-BiLSTM  |   83.35  |   78.88   |  58.85 | 59.36 |   83.35   |  83.35 | 83.35 |   67.24   |  80.00 | 89.39 |  76.47 |  9.09 | 91.00 | 71.56 | 16.33 | 90.19 |
| IAN          |   82.19  |   73.57   |  57.66 | 56.30 |   82.19   |  82.19 | 82.19 |   64.17   |  66.67 | 89.87 |  79.90 |  4.55 | 88.54 | 71.18 |  8.51 | 89.20 |
| LCRS         |   81.61  |   68.60   |  57.16 | 59.36 |   81.61   |  81.61 | 81.61 |   70.31   |  50.00 | 85.50 |  66.18 | 13.64 | 91.65 | 68.18 | 21.43 | 88.47 |
| CNN          |   81.84  |   73.19   |  55.21 | 55.14 |   81.84   |  81.84 | 81.84 |   65.44   |  66.67 | 87.48 |  69.61 |  4.55 | 91.49 | 67.46 |  8.51 | 89.44 |
| GCAE         |   79.98  |   49.95   |  50.00 | 49.74 |   79.98   |  79.98 | 79.98 |   66.47   |  0.00  | 83.38 |  56.37 |  0.00 | 93.62 | 61.01 |  0.00 | 88.20 |
| MemNet       |   81.26  |   67.07   |  55.25 | 57.94 |   81.26   |  81.26 | 81.26 |   70.41   |  46.15 | 84.64 |  58.33 | 13.64 | 93.78 | 63.81 | 21.05 | 88.98 |
| RAM          |   83.47  |   52.61   |  55.12 | 53.83 |   83.47   |  83.47 | 83.47 |   70.00   |  0.00  | 87.83 |  72.06 |  0.00 | 93.29 | 71.01 |  0.00 | 90.48 |
| CABASC       |   83.12  |   52.33   |  54.63 | 53.44 |   83.12   |  83.12 | 83.12 |   69.57   |  0.00  | 87.42 |  70.59 |  0.00 | 93.29 | 70.07 |  0.00 | 90.26 |

### The results of Twitter
|              | Accuracy |   Macro   |        |       |   Micro   |        |       | Precision |       |       | Recall |       |       |   F1  |       |       |
|--------------|:--------:|:---------:|:------:|:-----:|:---------:|:------:|:-----:|:---------:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|              |          | Precision | Recall |   F1  | Precision | Recall |   F1  |    Neg.   |  Neu. |  Pos. |  Neg.  |  Neu. |  Pos. |  Neg. |  Neu. |  Pos. |
| ContextAvg   |   68.35  |   68.69   |  64.26 | 65.82 |   68.35   |  68.35 | 68.35 |   70.15   | 67.88 | 68.03 |  54.34 | 80.64 | 57.80 | 61.24 | 73.71 | 62.50 |
| AEContextAvg |   69.94  |   69.57   |  66.57 | 67.75 |   69.94   |  69.94 | 69.94 |   67.11   | 70.66 | 70.95 |  58.96 | 80.06 | 60.69 | 62.77 | 75.07 | 65.42 |
| LSTM         |   69.22  |   69.64   |  65.13 | 66.52 |   69.22   |  69.22 | 69.22 |   66.46   | 69.12 | 73.33 |  63.01 | 81.50 | 50.87 | 64.69 | 74.80 | 60.07 |
| GRU          |   68.79  |   67.37   |  68.11 | 67.71 |   68.79   |  68.79 | 68.79 |   64.32   | 73.80 | 64.00 |  68.79 | 70.81 | 64.74 | 66.48 | 72.27 | 64.37 |
| BiGRU        |   67.20  |   67.63   |  62.14 | 63.68 |   67.20   |  67.20 | 67.20 |   67.11   | 66.74 | 69.03 |  58.96 | 82.37 | 45.09 | 62.77 | 73.74 | 54.55 |
| BiLSTM       |   68.21  |   67.75   |  64.84 | 65.98 |   68.21   |  68.21 | 68.21 |   69.18   | 69.13 | 64.94 |  58.38 | 78.32 | 57.80 | 63.32 | 73.44 | 61.16 |
| TD-LSTM      |   71.82  |   72.21   |  68.11 | 68.67 |   71.82   |  71.82 | 71.82 |   65.15   | 73.59 | 77.88 |  74.57 | 82.95 | 46.82 | 69.54 | 77.99 | 58.48 |
| TC-LSTM      |   72.69  |   72.76   |  69.65 | 70.90 |   72.69   |  72.69 | 72.69 |   74.00   | 72.56 | 71.71 |  64.16 | 81.79 | 63.01 | 68.73 | 76.90 | 67.08 |
| AT-LSTM      |   70.95  |   69.94   |  69.17 | 69.52 |   70.95   |  70.95 | 70.95 |   69.01   | 73.54 | 67.28 |  68.21 | 76.30 | 63.01 | 68.60 | 74.89 | 65.07 |
| AT-GRU       |   70.66  |   71.21   |  66.47 | 67.97 |   70.66   |  70.66 | 70.66 |   70.00   | 70.07 | 73.55 |  64.74 | 83.24 | 51.45 | 67.27 | 76.09 | 60.54 |
| AT-BiGRU     |   71.97  |   75.33   |  67.73 | 69.62 |   71.97   |  71.97 | 71.97 |   89.00   | 69.93 | 67.05 |  51.45 | 84.68 | 67.05 | 65.20 | 76.60 | 67.05 |
| AT-BiLSTM    |   69.80  |   68.93   |  67.73 | 68.14 |   69.80   |  69.80 | 69.80 |   70.55   | 72.65 | 63.59 |  59.54 | 76.01 | 67.63 | 64.58 | 74.29 | 65.55 |
| ATAE-GRU     |   69.94  |   70.11   |  65.51 | 67.11 |   69.94   |  69.94 | 69.94 |   68.97   | 69.73 | 71.64 |  57.80 | 83.24 | 55.49 | 62.89 | 75.89 | 62.54 |
| ATAE-LSTM    |   68.64  |   68.86   |  65.22 | 66.60 |   68.64   |  68.64 | 68.64 |   69.54   | 68.25 | 68.79 |  60.69 | 78.90 | 56.07 | 64.81 | 73.19 | 61.78 |
| ATAE-BiGRU   |   70.23  |   71.31   |  66.28 | 68.07 |   70.23   |  70.23 | 70.23 |   72.99   | 68.60 | 72.34 |  57.80 | 82.08 | 58.96 | 64.52 | 74.74 | 64.97 |
| ATAE-BiLSTM  |   70.95  |   72.77   |  66.38 | 68.38 |   70.95   |  70.95 | 70.95 |   80.34   | 69.10 | 68.87 |  54.34 | 84.68 | 60.12 | 64.83 | 76.10 | 64.20 |
| IAN          |   71.82  |   73.00   |  67.15 | 69.11 |   71.82   |  71.82 | 71.82 |   76.52   | 70.21 | 72.26 |  58.38 | 85.84 | 57.23 | 66.23 | 77.24 | 63.87 |
| LCRS         |   68.06  |   67.63   |  64.93 | 65.96 |   68.06   |  68.06 | 68.06 |   70.00   | 69.25 | 63.64 |  56.65 | 77.46 | 60.69 | 62.62 | 73.12 | 62.13 |
| CNN          |   67.77  |   66.41   |  64.26 | 65.02 |   67.77   |  67.77 | 67.77 |   66.67   | 70.94 | 61.63 |  53.18 | 78.32 | 61.27 | 59.16 | 74.45 | 61.45 |
| GCAE         |   72.11  |   72.12   |  70.04 | 70.85 |   72.11   |  72.11 | 72.11 |   75.69   | 72.65 | 68.00 |  63.01 | 78.32 | 68.79 | 68.77 | 75.38 | 68.39 |
| MemNet       |   69.65  |   69.09   |  66.76 | 67.68 |   69.65   |  69.65 | 69.65 |   71.17   | 70.57 | 65.52 |  67.05 | 78.32 | 54.91 | 69.05 | 74.25 | 59.75 |
| RAM          |   70.09  |   71.32   |  64.93 | 66.48 |   70.09   |  70.09 | 70.09 |   70.62   | 68.84 | 74.51 |  65.32 | 85.55 | 43.93 | 67.87 | 76.29 | 55.27 |
| CABASC       |   68.64  |   69.74   |  64.64 | 66.44 |   68.64   |  68.64 | 68.64 |   75.00   | 67.07 | 67.14 |  58.96 | 80.64 | 54.34 | 66.02 | 73.23 | 60.06 |


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

## Node
Thanks for [code](https://github.com/songyouwei/ABSA-PyTorch) shared by Song, Youwei, we conduct our model based on it.

# SemEval 2019 - Task 6: OffensEval: Identifying and Categorizing Offensive Language in Social Media
In this work we propose four deep recurrent architectures to tackle the task of offensive tweet detection as well as further categorization into targeting and subject of said targeting. Our architectures are based on LSTMs and GRUs, we present a simple bidirectional LSTM as a baseline system and then further increase the complexity of the models by adding convolutional layers and implementing a split-process-merge architecture with LSTM and GRU as processors. Multiple pre-processing techniques were also investigated. The validation F1-score results from each model are presented for the three subtasks as well as the final F1-score performance on the private competition test set. It was found that model complexity did not necessarily yield better results. Our best-performing model was also the simplest, a bidirectional LSTM; closely followed by a two-branch bidirectional LSTM and GRU architecture.

### Requisites
- [Keras](https://keras.io/)
- [TensorFlow-gpu](https://www.tensorflow.org/)
- [NLTK](https://www.nltk.org/)
- [Symspell](https://github.com/mammothb/symspellpy)
- [GloVe Twitter 27B Embeddings](https://nlp.stanford.edu/projects/glove/)

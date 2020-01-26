## **seq2seq** 

- The Sequence-to-Sequence model (seq2seq) converts a given sequence of text of fixed length into another sequence of fixed length, which we can easily relate to machine translation. But Seq2seq is not just limited to translation, in fact, it is quite efficient in tasks that require text generation.

- The model uses an encoder-decoder architecture and has been very successful in machine translation and question answering tasks. It uses a stack of Long Short Term Memory(LSTM) networks or Gated Recurrent Units(GRU) in encoders and decoders.

**Here is a simple demonstration of Seq2Seq model**:

- One major drawback of the Seq2Seq model comes from the limitation of its underlying RNNs. Though LSTMs are meant to deal with long term dependencies between the word vectors, the performance drops as the distance increases. The model also restricts parallelization.

- The transformer model introduces an architecture that is solely based on attention mechanism and does not use any Recurrent Networks but yet produces results superior in quality to Seq2Seq models.It addresses the long term dependency problem of the Seq2Seq model. The transformer architecture is also parallelizable and the training process is considerably faster.

**Let’s take a look at some of the important features** :

- Encoder: The encoder has 6 identical layers in which each layer consists of a multi-head self-attention mechanism and a fully connected feed-forward network. The multi-head attention system and feed-forward network both have a residual connection and a normalization layer. 

- Decoder: The decoder also consists of 6 identical layers with an additional sublayer in each of the 6 layers. The additional sublayer performs multi-head attention over the output of the encoder stack.

Attention Mechanism: 

- Attention is the mapping of a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The attention mechanism allows the model to understand the context of a text. 

- The transformer architecture is a breakthrough in the NLP spectrum, giving rise to many state-of-the-art algorithms such as Google’s BERT, RoBERTa, OpenGPT and many others.
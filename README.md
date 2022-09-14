# Seq2Seq-Bahdanau-Attention-based-Encoder-Decoder-Language-Translator

A language translator based on a very simple NLP Transformer model, backed by an encoder, decoder and a Bahdanau Attention Layer in between, implemented on TensorFlow.

## Dataset 
We use a random dataset of integers from `1 to vocab_size` to train and test our model. The library has built-in function `translator.AdditiveAttentionTranslator.generate_random_data()` to train our model and graph convergence.

## Abstract
Encoder-Decoder models for sequence-to-sequence transformation jobs like - translation, Machine Comprehension, Association tasks, summarization, etc often lack accuracy and face convergence/mode collapse issues. The reduction of a variable sentence into a fixed-size context vector `c` causes a loss of information before reaching the decoder. A basic remedy to the problem was introduced by [D, Bahdanau, et al. ICLR 2015], by using multiple hidden vectors (one for each word) (often called keys in transformer language as introduced by [Vaswani et al. Attention is all you need. 2017]) and using one attention layer that weighs each key and adds up them to form a new context vector each time after generating each word from the decoder and this way, generating one new context vector from keys each time after generating one new word. The solution led to improvement in the translation task and the model was able to translate long sequences without any problem. We stick to the basic framework as laid by [D, Bahdanau el al. ICLR 2015] and reproduce the model with significantly similar accuracy. The model takes in a fixed-size sentence (by padding the sentence with zeros for example to get the fixed-sized sentence) and generates fixed size values from the query (by padding the sentence with zeros to resize them to `max_width` as specified to the model during building). The usual model uses default `tf.keras.losses.CategoricalCrossentropy(with_logits = True)` as loss function and `tf.keras.optimizers.Adam()` for applying gradients, which can be altered and changed later on. The whole model is built on TensorFlow from scratch and is easily integrable to any future projects that require translation.

## Results

![Loss history](https://user-images.githubusercontent.com/31654395/190199136-1bc4fbe2-d388-40f2-820f-8d03728db967.png)

(*Training the model on Random data generated which mimicks encoding of a natural language.*)

## References
1. Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).
2. Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
3. Cho, Kyunghyun, et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation." arXiv preprint arXiv:1406.1078 (2014).
4. Radford, Alec, et al. "Improving language understanding by generative pre-training." (2018).
5. Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "Effective approaches to attention-based neural machine translation." arXiv preprint arXiv:1508.04025 (2015).
6. Chorowski, Jan K., et al. "Attention-based models for speech recognition." Advances in neural information processing systems 28 (2015).

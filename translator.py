import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size = 1000, embedding_size = 128):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
    def build(self, input_shapes):
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.gru = tf.keras.layers.GRU(self.embedding_size, return_sequences = True, return_state = True)
        print()
    def call(self, inputs):
        words = inputs
        embeddings = self.embedding_layer(words)
        output, state = self.gru(embeddings)
        return (output, state)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, words = 20, embedding_size = 128):
        super(BahdanauAttention, self).__init__()
        self.words = words
        self.embedding_size = embedding_size
    def build(self, input_shapes):
        self.W1 = self.add_weight(shape = (1, self.embedding_size), initializer = "random_uniform")
        self.W2 = self.add_weight(shape = (self.words, self.embedding_size), initializer = "random_uniform")
        self.W3 = self.add_weight(shape = (self.words, self.embedding_size), initializer = "random_uniform")
        self.W4 = self.add_weight(shape = (self.words, self.embedding_size), initializer = "random_uniform")
        print()
    def call(self, inputs):
        query, value = inputs
        
        regressed_query = tf.einsum("bi,ci -> bi", query, self.W1)
        regressed_value = tf.einsum("bij, ij -> bij", value, self.W2)
        
        sum_query_value = tf.einsum("bi, bji -> bji", regressed_query, regressed_value)
        sum_of_query_value = tf.nn.tanh(sum_query_value)
        
        a = tf.einsum("bij, ij -> bij", sum_of_query_value, self.W3)
        a = tf.math.reduce_sum(a, axis = -1)
        a = tf.nn.softmax(a)
        
        context = tf.einsum("bi, bij -> bij", a, value)
        context = tf.reduce_sum(context, axis = 1)
        
        
        return context

class Decoder(tf.keras.layers.Layer):
    def __init__(self, embedding_size = 128, vocab_size = 1000, words = 20):
        super(Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.words = words
    def build(self, input_shapes):
        self.attention = BahdanauAttention(words = self.words, embedding_size = self.embedding_size)
        self.gru = tf.keras.layers.GRU(self.embedding_size)
        self.op1 = tf.keras.layers.Dense(self.embedding_size * 10, activation = 'tanh')
        self.op2 = tf.keras.layers.Dense(self.embedding_size * 10, activation = 'tanh')
        self.op3 = tf.keras.layers.Dense(self.vocab_size, activation = 'softmax')
        print()
    def call(self, inputs):
        y, state, encode = inputs
        
        context = self.attention((state, encode))
        
        state_expanded = tf.expand_dims(state, axis = 1)
        context_expanded = tf.expand_dims(context, axis = 1)
        y_expanded = tf.expand_dims(y, axis = 1)
        
        gru1_input = tf.concat([state_expanded, context_expanded], axis = 1)
        gru1_input2 = tf.concat([gru1_input, y_expanded], axis = 1)
        
        new_state = self.gru(gru1_input2)
        
        g_input = tf.concat([tf.concat([y, context], axis = -1), new_state], axis = -1)
        g_output = self.op3(self.op2(self.op1(g_input)))
        
        return g_output, new_state

class AdditiveAttentionTranslator:
    encoder_input_words = 20
    vocab_size = 1000
    embedding_size = 128
    epochs = 30
    batch_size = 200
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
    loss_history = []
    
    
    def get_enc_dec(self):
        x_encoder_input = tf.keras.layers.Input(self.encoder_input_words)
        
        encode = encode = Encoder(vocab_size = self.vocab_size, embedding_size = self.embedding_size)(x_encoder_input)
        self.encoder = tf.keras.Model(inputs=x_encoder_input, outputs=encode)
        
        x_decoder_input = tf.keras.layers.Input(1)
        x_decoder = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)(x_decoder_input)
        x_state_input = tf.keras.layers.Input(self.embedding_size)
        x_states_input = tf.keras.layers.Input((self.encoder_input_words, self.embedding_size))
        
        decode = Decoder(embedding_size = self.embedding_size, vocab_size = self.vocab_size, words = self.encoder_input_words)((x_decoder[:,0], x_state_input, x_states_input))
        self.decoder = tf.keras.Model(inputs=[x_decoder_input, x_state_input, x_states_input], outputs = decode)
        return self.encoder.summary(), self.decoder.summary()
    
    def generate_random_data(self, instances = 1000, decoder_words = 10):
        X1, X2 = np.random.randint(self.vocab_size, size=(instances, self.encoder_input_words)), np.random.randint(self.vocab_size, size=(instances, decoder_words))
        Y = Y = np.eye(self.vocab_size)[np.random.choice(self.vocab_size, instances * decoder_words)].reshape(instances, decoder_words, self.vocab_size)
        self.X1, self.X2, self.Y = X1, X2, Y
        return X1, X2, Y
        
    def train_translator(self):
        tf.get_logger().setLevel('ERROR')
        
        optimizer, loss_fn = self.optimizer, self.loss_fn
        
        epochs, batch_size = self.epochs, self.batch_size
        total_instances = tf.shape(self.Y)[0]
        
        X1, X2, Y = self.X1, self.X2, self.Y
        
        
        self.loss_history = []
        
        for epoch in range(epochs):
            batch_loss = tf.constant(0.0)
            for batch in tqdm(range(0, total_instances, batch_size)):
                
                with tf.GradientTape() as tape:
                    loss_count = tf.constant(0.0)
                    x1_train = X1[batch : batch + batch_size]
                    x2_train = X2[batch : batch + batch_size]
                    y_train = Y[batch : batch + batch_size]
                    
                    H, state = self.encoder(x1_train)
                    
                    for query_number in range(x2_train.shape[-1]):
                        
                        output, state = self.decoder((x2_train[:, query_number], state, H))
                        loss_count = loss_count + loss_fn(y_train[:, query_number], output)
                grads = tape.gradient(loss_count, self.encoder.trainable_weights + self.decoder.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights + self.decoder.trainable_weights))
                batch_loss = batch_loss + loss_count
            print("Epoch: " + str(epoch + 1) + "/" + str(epochs) + " : Error " + str(batch_loss.numpy()))
            self.loss_history.append(batch_loss.numpy())
    
    def translate_sentence(self, keys, query_start, query_size = None):
        if query_size == None:
            query_size = self.X2.shape[-1]
        H, state = self.encoder(keys)
        
        value = []
        state_steps = []
        value.append(int(query_start[0][0]))
        
        
        for query_number in range(query_size):
            output, state = self.decoder((query_start, state, H))
            query_start = np.argmax(output.numpy(), axis = -1)
            value.append(query_start[0])
            state_steps.append(state)
        
        return value, state_steps
            
        
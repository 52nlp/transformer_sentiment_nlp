import pandas as pd
import numpy as np
# !pip install tensorflow==2.3.0
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_words = 20000
maxlen = 100

## Code setting
num_tests = 3
epochs = 5

vocab_size = num_words  # Only consider the top 20k words
embed_dim = 32  # Dimension of Embedding layers

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)




class baseTestModel():
    """The base class for testing model
    """

    def __init__(self):
        self.model = None
        self.stats = {}
        self.accuracies = []
        self.history = []

    def compile_and_test(self, num_test=3, batch_size=64, epochs=2, validation_split=0.2, verbose=2):
        # compile the model
        self.model.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        # get the weights
        self.model.save_weights('pretrain_weights.h5')
        # Perform test
        for random_seed in range(0, num_test):
            # reset the weights
            self.model.load_weights('pretrain_weights.h5')
            # Set seed before training
            tf.random.set_seed(random_seed)
            # Training
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                     validation_split=validation_split, verbose=verbose)
            self.history.append(history)
            # eval on test dataset
            test_scores = self.model.evaluate(x_test, y_test, verbose=2)
            # store the weights
            self.accuracies.append(test_scores[1])
            # print
            print(f"Accuracy: {test_scores[1]}; Loss: {test_scores[0]}")
        # create stats of the accuracies
        self.stats = {
            'model_name': [self.model.name],
            'num_tests': [num_test],
            'epochs': [epochs],
            'avg_accuracy': [np.mean(self.accuracies)],
            'std': [np.std(self.accuracies)],
            'max_accuracy': [max(self.accuracies)],
            'min_accuracy': [min(self.accuracies)]
        }
        # save stats
        self.saveStats()
        # print stats
        print(self.stats)

    def summary(self):
        print(self.model.summary())

    def plot(self):
        keras.utils.plot_model(self.model)

    def saveStats(self):
        self.stats['count_params'] = [self.model.count_params()]
        pd.DataFrame(self.stats).to_csv("modelTrainingStats_e=5.csv", mode="a", index=False, header=False)

    def plotLoss(self, epoch_no=0):
        plt.plot(self.history[epoch_no].history['loss'])
        plt.plot(self.history[epoch_no].history['val_loss'])
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def plotAccuracy(self, epoch_no=0):
        plt.plot(self.history[epoch_no].history['accuracy'])
        plt.plot(self.history[epoch_no].history['val_accuracy'])
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()




class Tansformer(baseTestModel):
    def __init__(self):
        # inherit
        super().__init__()
        # build model

        embedding_inputs = keras.Input(shape=(maxlen,))
        word_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        position_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        positions = np.array(range(maxlen))
        positions = position_emb(positions)
        embedding_outputs = word_emb(embedding_inputs)
        embedding_outputs += positions

        print('embedding_inputs', embedding_inputs)
        print('embedding_outputs', embedding_outputs)

        Embedding = keras.Model(inputs=embedding_inputs,
                                outputs=embedding_outputs)

        num_heads = 2  # Number of heads for multi-head attention model
        projection_dim = embed_dim // num_heads  # 32//2 = 16 in our case

        encoder_input = keras.Input(shape=(maxlen, embed_dim))
        query_dense = layers.Dense(embed_dim)  # defines 2 W_Q
        key_dense = layers.Dense(embed_dim)  # defines 2 W_K
        value_dense = layers.Dense(embed_dim)  # defines 2 W_V

        # Query, Key and Value matrices
        Q = query_dense(encoder_input)
        K = key_dense(encoder_input)
        V = value_dense(encoder_input)

        print('Q', Q)
        print('K', K)
        print('V', V)

        # split multi-heads
        Q = tf.reshape(Q, (-1, maxlen, num_heads, projection_dim))
        K = tf.reshape(K, (-1, maxlen, num_heads, projection_dim))
        V = tf.reshape(V, (-1, maxlen, num_heads, projection_dim))

        print('Q', Q)
        print('K', K)
        print('V', V)

        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        print('Q', Q)
        print('K', K)
        print('V', V)

        dimension_k = tf.cast(K.shape[-1], tf.float32)
        score = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dimension_k)
        print('score', score)

        weights = Activation('softmax', name='self-attention')(score)
        print('weights', weights)

        attention = tf.matmul(weights, V)
        print('attention', attention)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        print('attention', attention)

        concat_attention = tf.reshape(attention, (-1, maxlen, num_heads * projection_dim))
        print('concat_attention', concat_attention)

        combine_heads = layers.Dense(embed_dim)
        multiHead_output = combine_heads(concat_attention)
        print('multiHead_output', multiHead_output)

        MultiHeadSelfAttention = keras.Model(inputs=encoder_input,
                                             outputs=multiHead_output)

        hidden_units = 32
        dropout_rate = 0.1

        encoder_inputs = keras.Input(shape=(maxlen,))
        embeded_inputs = Embedding(encoder_inputs)
        multiHead_output = MultiHeadSelfAttention(embeded_inputs)
        out1 = layers.Dropout(dropout_rate)(multiHead_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(out1 + embeded_inputs)
        feed_forword_nn = keras.Sequential(
            [layers.Dense(hidden_units, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        feed_forword_output = feed_forword_nn(out1)
        out2 = layers.Dropout(dropout_rate)(feed_forword_output)
        out2 = layers.LayerNormalization(epsilon=1e-6)(out2 + out1)

        Encoder = keras.Model(inputs=encoder_inputs,
                              outputs=out2)
        inputs = keras.Input(shape=(maxlen,))
        x = Encoder(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name="TRANSFORMER")



class LSTM_DENSE_CON(baseTestModel):
  def __init__(self):
    # inherit
    super().__init__()
    # build model
    inputs = keras.Input(shape=(maxlen, ))
    embedding = layers.Embedding(num_words, embed_dim)(inputs)
    r1 = layers.LSTM(128, return_sequences=True)(embedding)
    r2 = layers.LSTM(128, return_sequences=True)(r1)
    concat12 = tf.keras.layers.Concatenate()([r1, r2])

    r3 = layers.LSTM(128, return_sequences=True)(concat12)
    concat123 = tf.keras.layers.Concatenate()([concat12, r3])


    avg = tf.keras.layers.AveragePooling1D(pool_size=maxlen)(r3)
    dense = layers.Dense(64)(avg)
    output = layers.Dense(1, activation='sigmoid')(dense)
    self.model = keras.Model(inputs=inputs, outputs=output, name="LSTM_DENSE_CON")


# call
model = Tansformer()
model.compile_and_test(num_test=num_tests, epochs=epochs)

# call
model = LSTM_DENSE_CON()
model.compile_and_test(num_test=num_tests, epochs=epochs)

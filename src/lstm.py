import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding
from keras.optimizers import RMSprop
import random

import os

output_dir = 'result'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'gen.txt')

class TextGenerator:
    def __init__(self, file_path='C:\\Users\\HP\\source\\repos\\mod-lab07-lstm\\src\\input.txt'):
        self.file_path = file_path
        self.load_data()
        self.prepare_data()
        self.build_model()
        
    def load_data(self):
        with open(self.file_path, 'r', encoding='utf8') as f:
            self.text = f.read()
        self.words = self.text.split()
        self.vocab = sorted(list(set(self.words)))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        
    def prepare_data(self):
        self.seq_length = 10
        self.step = 1
        sequences = [self.word2idx[word] for word in self.words if word in self.word2idx]
        
        self.sentences = []
        self.next_words = []
        for i in range(0, len(sequences) - self.seq_length, self.step):
            self.sentences.append(sequences[i:i + self.seq_length])
            self.next_words.append(sequences[i + self.seq_length])
            
        X = np.zeros((len(self.sentences), self.seq_length), dtype=np.int32)
        y = np.zeros((len(self.sentences), len(self.vocab)), dtype=np.bool)
        
        for i, sentence in enumerate(self.sentences):
            for t, word in enumerate(sentence):
                X[i, t] = word
            y[i, self.next_words[i]] = 1
            
        self.X = X
        self.y = y
        
    def build_model(self):
        self.model = Sequential([
            Embedding(len(self.vocab), 100, input_length=self.seq_length),
            LSTM(128),
            Dense(len(self.vocab)),
            Activation('softmax')
        ])
        
        optimizer = RMSprop(learning_rate=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        
    def train(self, epochs=50, batch_size=128):
        self.model.fit(self.X, self.y, batch_size=batch_size, epochs=epochs)
        
    def generate(self, length=1000, diversity=0.2):
        start_idx = random.randint(0, len(self.sentences) - self.seq_length - 1)
        generated = []
        sentence = self.sentences[start_idx]
        generated.extend(sentence)
        
        for _ in range(length):
            x = np.zeros((1, self.seq_length))
            for t, word in enumerate(sentence):
                x[0, t] = word
                
            preds = self.model.predict(x, verbose=0)[0]
            next_idx = self._sample(preds, diversity)
            generated.append(next_idx)
            sentence = sentence[1:] + [next_idx]
            
        return ' '.join(self.idx2word[idx] for idx in generated)
        
    def _sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

if __name__ == "__main__":
    generator = TextGenerator()
    generator.train()
    result = generator.generate()
    print(result)

    output_dir = 'result'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'gen.txt')

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(result)

from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from transformers.optimization_tf import WarmUp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from random import choice
import numpy as np


class AQModel:
    """ Asking Questions Model """
    
    def __init__(self):
        self.tokenizer = None
        self.load_tokenizer()
        self.model = None
        self.load_model()
        
    def load_tokenizer(self):
        self.tokenizer = T5Tokenizer('models/tokenizer/sentencepiece.model')
        print(f'Tokenizer loaded.')

    def load_model(self):
        self.model = T5ForConditionalGeneration.from_pretrained('models/aq', from_tf=True)
        print(f'AQ-Model loaded.')

    def generate(self, context, choose_from=1):
        input_ids = self.tokenizer([context], return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, 
                                        max_length=128,
                                        num_beams=5,
                                        no_repeat_ngram_size=2, 
                                        num_return_sequences=choose_from, 
                                        early_stopping=True)

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        if choose_from > 1:
            print(f'Randomly choosing from: ')
            for d in decoded:
                print(f'{self.parse(d)}')
        
        return self.parse(choice(decoded))
    
    def parse(self, generated):
        try:
            q, a = generated.split('Answer:')
            return q.replace('Question:', '').split('?')[0].strip()+'?', a.strip()
        except:
            print('W: Not parsed.')
            return generated
    

class SCModel:
    """ Semantic Continuity Model """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        self.model, self.embedder, self.dist_model = None, None, None
        self.load_models()

    def load_models(self):
        self.model = load_model('models/sc/model', custom_objects={'WarmUp': WarmUp, 'siamese_loss': None})
        self.embedder = self.model.get_layer('embedder')
        self.dist_model = self.model.get_layer('dist_model')
        print('Models loaded.')

    def vectorize_seq(self, seq, max_len):
        seq = self.tokenizer.tokenize(seq)
        input_ids = [self.cls_token_id, *self.tokenizer.convert_tokens_to_ids(seq), self.sep_token_id]
        input_ids = pad_sequences([input_ids], maxlen=max_len, dtype=np.int32, padding='post', value=self.pad_token_id)
        return input_ids
        
    def score(self, prompt, reaction):
        _, pemb = self.embedder.predict(self.vectorize_seq(prompt, max_len=32))
        remb, _ = self.embedder.predict(self.vectorize_seq(reaction, max_len=64))
        dist, prob = self.dist_model.predict([pemb, remb])
        return float(dist)


class SSModel:
    """ Semantic Similarity Model """
    
    def __init__(self):
        self.bert_model = SentenceTransformer('all-distilroberta-v1')

    def similarity(self, a, b):
        return np.linalg.norm(a-b)
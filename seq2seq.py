import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time


def seq2seq():
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text):
        # tokenize German text and reverse it
        return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

    def tokenize_en(text):
        # tokenize English text
        return [tok.text for tok in spacy_en.tokenizer(text)]


    # English
    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    # German
    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(SRC, TRG))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)


if __name__ == '__main__':
    seq2seq()

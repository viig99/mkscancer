import pandas as pd
import numpy as np
import re
import glob
import spacy
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
import pickle
import csv
import string
import itertools
import os.path
import gensim
from tqdm import tqdm
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from gensim.models.keyedvectors import KeyedVectors
import pickle
from keras import regularizers
from keras import layers as ll
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, Model
from keras.layers.core import Flatten, RepeatVector, Reshape, Dropout, Masking
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot, multiply, concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.optimizers import TFOptimizer, Adam
from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.utils import plot_model, normalize

tqdm.pandas()
nlp = spacy.load('en')
table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))


def removePunctuation(s):
    return s.translate(table)


def getTranslatedText(s):
    tokens = nlp(s, entity=False)
    list_of_tags = ['CD', 'FW', 'JJ', 'JJR', 'JJS',
                    'NN', 'NNP', 'NNPS', 'NNS', 'RB', 'RBR', 'RBS']
    list_of_adj_adv = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    list_of_nouns = ['CD', 'FW', 'NN', 'NNP', 'NNPS', 'NNS']
    return {
        'all_text': getTextFromTokens(tokens, list_of_tags),
        'nouns': getTextFromTokens(tokens, list_of_nouns),
        'adjVer': getTextFromTokens(tokens, list_of_adj_adv)
    }


def getTextFromTokens(tokens, list_of_tags):
    tags = [(token.lemma_, token.tag_) for token in tokens]
    filtered_list = filter(lambda x: x[1] in list_of_tags, tags)
    file_filtered_string = " ".join(map(lambda x: x[0], filtered_list)).lower()
    return file_filtered_string


def preprocessAndFormColumns(s):
    obj = getTranslatedText(removePunctuation(s))
    return pd.Series([obj['all_text'], obj['nouns'], obj['adjVer']])


def preprocessText(s):
    return getTranslatedText(removePunctuation(s))


def formEmbeddingWeights(max_words):
    print("Loading Pubmed word2vec weights")
    word_vectors = KeyedVectors.load_word2vec_format('data/bio_nlp_vec/PubMed-and-PMC-w2v.bin', binary=True)
    # word_vectors = KeyedVectors.load_word2vec_format(
    #     'data/bio_nlp_vec/PubMed-shuffle-win-30.bin', binary=True)
    print("Loaded Pubmed word2vec weights")
    lowerToWVDict = {v.lower(): v for v in word_vectors.vocab.keys()}
    embedding_matrix = np.random.uniform(-0.05,
                                         0.05, (max_words, word_vectors.vector_size or 200))
    textEncoder = pickle.load(open('data/textEncoder.pk', 'rb'))
    word_index = textEncoder.word_index
    vocab = lowerToWVDict.keys()
    for word, i in word_index.items():
        if word in vocab:
            embedding_matrix[i] = word_vectors.word_vec(lowerToWVDict[word])
    return embedding_matrix


def get_all_data(path):
    print("Reading Files Into Memory")
    train_variant = pd.read_csv("data/training_variants")
    test_variant = pd.read_csv("data/test_variants")
    train_text = pd.read_csv("data/training_text", sep="\|\|",
                             engine='python', header=None, skiprows=1, names=["ID", "Text"])
    test_text = pd.read_csv("data/test_text", sep="\|\|",
                            engine='python', header=None, skiprows=1, names=["ID", "Text"])
    print("Files Read Into Memory")

    print("Begin Forming Input and Output Numpy Matrices.")
    train = pd.merge(train_variant, train_text, how='left', on='ID')
    train['Class'] = train['Class'] - 1
    train_y = train['Class'].values
    train_x = train.drop('Class', axis=1)
    print("Numy Matrix formation Done.")
    # number of train data : 3321

    test_x = pd.merge(test_variant, test_text, how='left', on='ID')
    test_index = test_x['ID'].values
    print("Test Stuff Done!")
    # number of test data : 5668

    if os.path.exists(path):
        all_data = pd.read_csv(path)
        print("Read CSV FILE!")
    else:
        all_data = np.concatenate((train_x, test_x), axis=0)
        all_data = pd.DataFrame(all_data)
        all_data.columns = ["ID", "Gene", "Variation", "Text"]
        all_data[["Clinical_Data", "Nouns", "AdjVerbs"]] = all_data.progress_apply(
            lambda x: preprocessAndFormColumns(x['Text']), axis=1)
        all_data.to_csv(path, index=False, header=True)
    return all_data, train_x, train_y, test_x


def training_generator(path="train", dir_path="./", batch_size=32, maxlen=8000, test_size=0.1):
    all_data, train_x, train_y, test_x = get_all_data(
        'data/data_with_clinical_text_py3_next.csv')

    print("All Data Loaded")

    oncokb = pd.read_csv('data/validationOncoKB_next.csv')

    print("OncoKB Loaded")

    # textEncoder = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
    # texts = itertools.chain(list(all_data['Clinical_Data'].values), list(
    #     all_data['Gene'].values), list(all_data['Variation'].values))
    # textEncoder.fit_on_texts(texts)
    textEncoder = pickle.load(open('data/textEncoder.pk', 'rb'))
    gene_sequence = textEncoder.texts_to_sequences(
        list(all_data["Gene"].values))
    variants_sequence = pad_sequences(textEncoder.texts_to_sequences(
        list(all_data["Variation"].values)), 6)
    nouns_sequence = pad_sequences(textEncoder.texts_to_sequences(
        list(all_data["Nouns"].values.astype(str))), maxlen=maxlen)
    adjver_sequence = pad_sequences(textEncoder.texts_to_sequences(
        list(all_data["AdjVerbs"].values.astype(str))), maxlen=1500)

    print("Sequences Formed")

    gene_sequence_onco = textEncoder.texts_to_sequences(
        list(oncokb["Gene"].values))
    variants_sequence_onco = pad_sequences(textEncoder.texts_to_sequences(
        list(oncokb["Variation"].values)), 6)
    nouns_sequence_onco = pad_sequences(textEncoder.texts_to_sequences(
        list(oncokb["Nouns"].values.astype(str))), maxlen=maxlen)
    adjver_sequence_onco = pad_sequences(textEncoder.texts_to_sequences(
        list(oncokb["AdjVerbs"].values.astype(str))), maxlen=1500)

    gene_sequence_train = gene_sequence[:len(train_x)]
    variants_sequence_train = variants_sequence[:len(train_x)]
    nouns_sequence_train = nouns_sequence[:len(train_x)]
    adjver_sequence_train = adjver_sequence[:len(train_x)]
    y_sequence_train = to_categorical(train_y)
    y_sequence_onco = to_categorical(oncokb['Class'].values)

    if path == "train":
        genes = np.array(gene_sequence_train).reshape(-1, 1)
        variants = np.array(variants_sequence_train)
        nouns = np.array(nouns_sequence_train)
        verbs = np.array(adjver_sequence_train)
        y = np.array(y_sequence_train)
    else:
        genes = np.array(gene_sequence_onco).reshape(-1, 1)
        variants = np.array(variants_sequence_onco)
        nouns = np.array(nouns_sequence_onco)
        verbs = np.array(adjver_sequence_onco)
        y = np.array(y_sequence_onco)

    if os.path.exists('data/textEncoder.pk') != True:
        pickle.dump(textEncoder, open('data/textEncoder.pk', 'wb'))

    while True:
        idx = np.random.random_integers(0, y.shape[0] - 1, (batch_size,))
        yield ({'genes': genes[idx, ...], 'variants': variants[idx, ...], 'noun_data': nouns[idx, ...], 'verb_data': verbs[idx, ...]}, [y[idx, ...]])


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv1D(filters1, 1, name=conv_name_base + '2a')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)

    x = ll.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv1D(filters1, 1, strides=strides,
               name=conv_name_base + '2a')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)

    shortcut = BatchNormalization(name=bn_name_base + '1')(input_tensor)
    shortcut = Conv1D(filters3, 1, strides=strides,
                      name=conv_name_base + '1')(shortcut)

    x = ll.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_block(input_tensor, final_layer_output=220, append='n'):
    x = Conv1D(
        64, 7, strides=2, padding='same', name='conv1' + append)(input_tensor)
    x = BatchNormalization(name='bn_conv1' + append)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)
    x = conv_block(x, 3, [64, 64, 256],
                   stage=2, block='a' + append, strides=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b' + append)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c' + append)
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d' + append)
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='g' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='h' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='i' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='j' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='k' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='l' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='m' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='n' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='o' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='p' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='q' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='r' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='s' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='t' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='u' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='v' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='w' + append)
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a' + append)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b' + append)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c' + append)
    x = AveragePooling1D(final_layer_output, name='avg_pool' + append)(x)
    x = Flatten()(x)
    return x


def getMedCanModel(num_words, latent_dim, sequence_length, final_output_length, embedding_matrix):
    # Input Tensors
    # clinical_data_input = Input((sequence_length,), name="clinical_data")
    nouns_data_input = Input((sequence_length,), name="noun_data")
    verbs_data_input = Input((1500,), name="verb_data")
    genes_input = Input((1,), name="genes")
    variants_input = Input((6,), name="variants")

    shared_layers = Embedding(
        num_words, latent_dim, weights=[embedding_matrix], name='text_embedding', trainable=True)
    nouns_data_embedding_layer = shared_layers(nouns_data_input)
    verbs_data_embedding_layer = shared_layers(verbs_data_input)

    gene_embedding_layers = Masking(mask_value=0)(genes_input)
    gene_embedding_layers = shared_layers(gene_embedding_layers)
    gene_embedding_layers = GlobalMaxPooling1D()(gene_embedding_layers)

    variants_embedding_layers = Masking(mask_value=0)(variants_input)
    variants_embedding_layers = shared_layers(variants_embedding_layers)
    variants_embedding_layers = GlobalMaxPooling1D()(variants_embedding_layers)

    nouns_resnet_layers = resnet_block(nouns_data_embedding_layer, 220, 'n')
    verbs_resnet_layers = resnet_block(verbs_data_embedding_layer, 47, 'v')

    embedding_layers = add([gene_embedding_layers, variants_embedding_layers])
    layers = concatenate(
        [nouns_resnet_layers, verbs_resnet_layers, embedding_layers], name='final')
    layers = Dense(final_output_length, activation='softmax')(layers)
    model = Model(
        inputs=[nouns_data_input, verbs_data_input,
                genes_input, variants_input],
        outputs=[layers])
    # model.load_weights("model_weights_resnet_nouns_pubmed.h5")
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01, epsilon=0.1),
                  metrics=['categorical_accuracy'])
    return model

if __name__ == '__main__':
    MAX_WORDS = 263965
    MAX_LENGTH = 7040
    OUTPUT_DIM = 9
    BATCH_SIZE = 4

    embedding_matrix = formEmbeddingWeights(MAX_WORDS)
    EMBEDDING_DIMS = embedding_matrix.shape[1]

    model = getMedCanModel(MAX_WORDS, EMBEDDING_DIMS,
                           MAX_LENGTH, OUTPUT_DIM, embedding_matrix)
    model.summary()
    plot_model(model, to_file='medcan_model.png', show_shapes=True)

    training_gen = training_generator(
        'train', batch_size=BATCH_SIZE, maxlen=MAX_LENGTH)
    test_gen = training_generator(
        'valid', batch_size=BATCH_SIZE, maxlen=MAX_LENGTH)

    checkpointer = ModelCheckpoint(
        filepath='model_weights_resnet_nouns_pubmed_pmc.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True, save_weights_only=True, period=5)
    model.fit_generator(generator=training_gen, steps_per_epoch=200, initial_epoch=0,
                        epochs=150, verbose=1, validation_data=test_gen, validation_steps=150, max_q_size=10, workers=1, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=12, verbose=1, mode='auto'), ReduceLROnPlateau(monitor='val_loss', patience=7, verbose=1, min_lr=0.00001, epsilon=0.0001, factor=0.1), checkpointer])
    model.save_weights("model_weights_resnet_nouns_pubmed_pmc.h5")
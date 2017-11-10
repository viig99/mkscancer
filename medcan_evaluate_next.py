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
from tqdm import tqdm
from scipy.sparse import load_npz
from keras.models import load_model
from sklearn.metrics import log_loss
from joblib import Parallel, delayed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import TFOptimizer, Adam
from keras.models import model_from_json
from keras import regularizers
from keras import layers as ll
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, Model
from keras.layers.core import Flatten, RepeatVector, Reshape, Dropout, Masking
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot, multiply, concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.optimizers import TFOptimizer, Adam
from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.utils import plot_model, normalize


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


def getMedCanModel(num_words, latent_dim, sequence_length, final_output_length):
    # Input Tensors
    nouns_data_input = Input((sequence_length,), name="noun_data")
    verbs_data_input = Input((1500,), name="verb_data")
    genes_input = Input((1,), name="genes")
    variants_input = Input((6,), name="variants")

    shared_layers = Embedding(
        num_words, latent_dim, name='text_embedding')
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
    model.load_weights("model_weights_resnet_nouns_pubmed_pmc.49-0.98.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01, epsilon=0.1),
                  metrics=['categorical_accuracy'])
    return model


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
    # number of test data : 5668

    if os.path.exists(path):
        all_data = pd.read_csv(path)
    else:
        all_data = np.concatenate((train_x, test_x), axis=0)
        all_data = pd.DataFrame(all_data)
        all_data.columns = ["ID", "Gene", "Variation", "Text"]
        all_data["Clinical_Data"] = all_data.progress_apply(
            lambda x: preprocessText(x['Text']), axis=1)
        all_data.to_csv(path, index=False, header=True)
    return all_data, train_x, train_y, test_x, test_index


def testResults():
    all_data, train_x, train_y, test_x, test_index = get_all_data(
        'data/data_with_clinical_text_py3_next.csv')
    textEncoder = pickle.load(open('data/textEncoder.pk', 'rb'))

    gene_sequence = textEncoder.texts_to_sequences(
        list(test_x["Gene"].values))
    variants_sequence = pad_sequences(textEncoder.texts_to_sequences(
        list(test_x["Variation"].values)), 6)
    text_sequence = pad_sequences(textEncoder.texts_to_sequences(
        list(all_data.iloc[len(train_y):]["Nouns"].values.astype(str))), maxlen=7040)
    verb_sequence = pad_sequences(textEncoder.texts_to_sequences(
        list(all_data.iloc[len(train_y):]["AdjVerbs"].values.astype(str))), maxlen=1500)
    genes = np.array(gene_sequence)
    variants = np.array(variants_sequence)
    text = np.array(text_sequence)
    verbs = np.array(verb_sequence)
    inputDict = {'genes': genes, 'variants': variants,
                 'noun_data': text, 'verb_data': verbs}
    return inputDict, test_index

if __name__ == '__main__':
    print("Loading Model")
    # model = load_model('medcan_resnet_gene_variant')

    # load json and create model
    model = getMedCanModel(263965, 200, 7040, 9)
    # load weights into new model
    print("Model Loaded")

    print("Collecting Input")
    inputDict, test_index = testResults()
    print("Done")

    print("Starting Prediction")
    probs = model.predict(inputDict, batch_size=4, verbose=1)
    print("Form Submission File")
    submission = pd.DataFrame(probs)
    submission.columns = ['class1', 'class2', 'class3',
                          'class4', 'class5', 'class6', 'class7', 'class8', 'class9']
    submission['id'] = test_index
    submission.to_csv("submission_pubmed_pmc.csv", index=False)

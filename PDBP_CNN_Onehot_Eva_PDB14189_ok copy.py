# -*- coding: utf-8 -*-
import os
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score , accuracy_score, precision_score,f1_score,matthews_corrcoef
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten,Activation
from keras.layers.convolutional import Convolution1D, MaxPooling1D, SeparableConv1D
from keras.layers import LSTM, Bidirectional,Dense,Embedding,Input,concatenate
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
import numpy as np
import utils.tools as utils
import encode_schema as tool
import cnn_models as cnnmodel
from datetime import datetime


# ================gpu动态占用====================#
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
# A "Best-fit with coalescing" algorithm, simplified from
#   a version of dlmalloc.
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))
# =================gpu动态占用====================#


def get_CNN_model(number=4,conv_width=7):   
    # 构建网络
    inputs = layers.Input(shape=(800, 20), name="input")

    for index in range(number):
        if index==0:
            x = Convolution1D(64,
                      conv_width,
                      strides=1,
                      padding='same',
                      kernel_initializer='random_uniform',
                      activation="relu",
                      name="conv1d_1"+str(index))(inputs)
        else:
            x = Convolution1D(64,
                      conv_width,
                      strides=1,
                      padding='same',
                      kernel_initializer='random_uniform',
                      activation="relu",
                      name="conv1d_1"+str(index))(x)
        x = layers.MaxPooling1D(pool_size=2, name="max_pooling1d_1"+str(index))(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2, name="dropout_1"+str(index))(x)

    x = layers.Flatten(name="flatten_6")(x)
    x = layers.Dense(128, activation='relu', name="dense_11")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name="dropout_3")(x)
    x = layers.Dense(2, activation='softmax', name="dense_12")(x)

    model = models.Model(inputs, x, name='DBP_CNN_model')
    return model

if __name__ == "__main__":
    path = "result/DNA/"
    if not os.path.exists(path):
        os.makedirs(path)


    model_array = [2,3,4]
    filters = 64
    conv_width = [7,8,9]
    conv_stride = 1
    max_polling_width1= 3
    max_polling_width2= 2
    lstm_size = 32
    dense_size1 = 128
    dense_size2 = 2
    drop_out = [0.2,0.3]
    uselstm=[0,1]
    batchsiz=[64,32]
    # for model_layers in model_array[:-1]:
    for batch_s in batchsiz[:-1]:
        for model_layers in model_array[:-1]:
            for use_blstm in uselstm[:-1]:
                start = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                fw_perf = open(path + 'PDB14189_CNN_' +str(model_layers)+ '_['+str(7)+']'+start, 'w')
                fw_perf.writelines('model_layers'+str(model_layers)+ '\n')
                fw_perf.writelines('conv_width'+str(7)+ '\n')
                fw_perf.writelines('uselsem'+str(use_blstm)+ '\n')
                fw_perf.writelines('batch_s'+str(batch_s)+ '\n')
                fw_perf.writelines('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------'+ '\n')
                fw_perf.write('          acc,           precision,              npv,          sensitivity,      specificity,          mcc,                 ppv,             auc,                pr' + '\n')
                fw_perf.writelines('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------'+ '\n')

                scores = []
                k_fold = 5
                for index in range(5):
                    dataset = []
                    dataset = tool.read_seq_onehot("data/DNA_Pading_800_PDB14189")
                    # print(dataset.shape[0])
                    label = np.loadtxt("data/class_PDB14189")
                    # print(label.shape())
                    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=1024)
                    for ((train_index, test_index), k) in zip(skf.split(dataset, label),
                                                range(k_fold)):

                        X_train, X_test = dataset[train_index], dataset[test_index]
                        y_train, y_test = label[train_index], label[test_index]

                        
                        # create the model
                        model = get_CNN_model(number=model_layers,conv_width=7)

                        # 指定回调函数
                        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.1,
                                                    patience=3,
                                                    verbose=1)
                        early_stopping = EarlyStopping(monitor='val_loss',
                                                    min_delta=0,
                                                    patience=5,
                                                    verbose=1)

                        print(model.summary())
                        model.compile(loss='binary_crossentropy',
                                    optimizer='adam',
                                    metrics=['accuracy'])
                        y_train = utils.to_categorical(y_train)

                        model.fit(X_train,
                                y_train,
                                validation_split=0.1,
                                batch_size=batch_s,
                                epochs=50,
                                verbose=1,
                                callbacks=[reduce_lr, early_stopping])

                        # prediction probability
                        predictions = model.predict(X_test)
                        predictions_prob = predictions[:, 1]
   
                        y_test = utils.to_categorical(y_test)
                        y_class = utils.categorical_probas_to_classes(predictions)
                        true_y = utils.categorical_probas_to_classes(y_test)


                        auc_ = roc_auc_score(true_y, predictions_prob)
                        pr = average_precision_score(true_y, predictions_prob)

                        (
                            acc,
                            precision,
                            npv,
                            sensitivity,
                            specificity,
                            mcc,
                            f1
                        ) = utils.calculate_performace(len(y_class), y_class, true_y)
                        print("======================")

                        print(str(k + 1) + " of " + str(k_fold) +
                            "cv:")
                        print(
                            '\tacc=%0.4f,pre=%0.4f,npv=%0.4f,sn=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f'
                            % (acc, precision, npv, sensitivity, specificity, mcc, f1))
                        print('\tauc=%0.4f,pr=%0.4f' % (auc_, pr))

                        fw_perf.write(
                            str(acc) + ',' + str(precision) + ',' + str(npv) + ',' +
                            str(sensitivity) + ',' + str(specificity) + ',' + str(mcc) +
                            ',' + str(f1) + ',' + str(auc_) + ',' + str(pr) + '\n')

                        scores.append([
                            acc, precision, npv, sensitivity, specificity, mcc, f1, auc_,
                            pr
                        ])

                scores = np.array(scores)
                print(len(scores))
                print("acc=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100))
                print("precision=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100))
                print("npv=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100))
                print("sensitivity=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100))
                print("specificity=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100))
                print("mcc=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100))
                print("f1=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100))
                print("roc_auc=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100))
                print("roc_pr=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[8] * 100, np.std(scores, axis=0)[8] * 100))

                end = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                print('start: %s' % start)
                print('end: %s' % end)

                fw_perf.write(
                    "acc=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100) +
                    '\n')
                fw_perf.write(
                    "precision=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100) +
                    '\n')
                fw_perf.write(
                    "npv=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100) +
                    '\n')
                fw_perf.write(
                    "sensitivity=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100) +
                    '\n')
                fw_perf.write(
                    "specificity=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100) +
                    '\n')
                fw_perf.write(
                    "mcc=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100) +
                    '\n')
                fw_perf.write(
                    "f1=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100) +
                    '\n')
                fw_perf.write(
                    "roc_auc=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100) +
                    '\n')
                fw_perf.write(
                    "roc_pr=%.2f%% (+/- %.2f%%)" %
                    (np.mean(scores, axis=0)[8] * 100, np.std(scores, axis=0)[8] * 100) +
                    '\n')
                fw_perf.write('start: %s' % start + '\n')
                fw_perf.write('end: %s' % end)

                fw_perf.close()

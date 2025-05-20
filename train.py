import time
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, LambdaCallback
from keras.models import load_model
from DenseNet_aux_tf import DenseNet
from ResNet_aux_tf import ResnetBuilder
from WideResNet_aux_tf import WideResidualNetwork
from helper import *
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np
from sklearn.metrics import classification_report

from load_RSI_CB128_data import load_RSI_CB128_data

def print_per_class_accuracy(model, x_test, y_test, class_names):
    y_pred = model.predict(x_test, batch_size=256)
    if isinstance(y_pred, list):
        y_pred = y_pred[0]  # 主输出在第一个
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print("\nPer-class classification report:\n")
    print(classification_report(y_true_labels, y_pred_labels, target_names=class_names, digits=4))

def train(**kwargs):
    params = {'with_aux': False, 'aux_weight': 0.3, 'aux_depth': [], 'batch_size': 256, 'epochs': 1,
              'learning_rate': 0.01, 'weight_decay': None, 'aux_weight_decay': None,
              'momentum': 0.95, 'num_coarse_classes': 20, 'exp_combination_factor': 1.0,
              'save_model': False, 'use_pretrained': None, 'max_epoch': 10,
              'grouping': 'default', 'optimize': False, 'network': None, 'aux_layouts': None,
              'wide_width': 10, 'wide_depth': 28, 'se_net': False, 'dense_depth': 100, 'dense_growth': 12,
              'nesterov': False, 'mean_std_norm': False, 'label_smoothing': False
              }
    params.update(kwargs)

    grouping = params['grouping']
    if not isinstance(grouping, list):
        grouping = [grouping]

    hyper = params
    aux_depth = params['aux_depth']
    if isinstance(aux_depth, int):
        aux_depth = [aux_depth]
    aux_depth = aux_depth.copy()
    for i in range(len(aux_depth)):
        if not isinstance(aux_depth[i], list):
            aux_depth[i] = (int(aux_depth[i]), -1)

    with_aux = hyper['with_aux']
    num_auxs = len(aux_depth)

    max_epoch = params['max_epoch']

    batch_size = hyper['batch_size']
    epochs = hyper['epochs']
    aux_weight = hyper['aux_weight']
    if not isinstance(aux_weight, list):
        aux_weight = [aux_weight]

    aux_weight_decay = params['aux_weight_decay']
    if aux_weight_decay is None:
        aux_weight_decay = params['weight_decay']

    learning_rate = hyper['learning_rate']

    num_coarse_classes = hyper['num_coarse_classes']
    if not isinstance(num_coarse_classes, list):
        num_coarse_classes = [num_coarse_classes]

    aux_layouts = hyper['aux_layouts']

    if with_aux:
        if not isinstance(aux_layouts[0][0], list):
            raise TypeError('Bad aux_layouts format. Expect list with 2-element list for each SSAL branch')
        if aux_layouts is not None:
            if len(aux_layouts) < len(aux_depth):
                while len(aux_layouts) < len(aux_depth):
                    aux_layouts.append(aux_layouts[-1])

    num_classes = 45

    data_dir = 'data'
    x_train, y_train = load_RSI_CB128_data(data_dir)

    if params['optimize']:
        len_val = round(0.85 * len(y_train))
        x_test = x_train[len_val:]
        x_train = x_train[:len_val]
        y_test = y_train[len_val:]
        y_train = y_train[:len_val]
    else:
        x_test, y_test = x_train, y_train

    print('x_train:', x_train.shape)
    print('y_train:', y_train.shape)
    print('x_test:', x_test.shape)
    print('y_test:', y_test.shape)

    steps_per_epoch = int(len(x_train) / batch_size)
    combined_accuracy_log = []

    cats = []
    y_train_c = []
    y_test_c = []
    if with_aux:
        cats, y_train_c, y_test_c = create_coarse_data(y_train, y_test, 'cifar100', grouping)

    if not params['label_smoothing']:
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        if with_aux:
            for i in range(0, num_auxs):
                y_train_c[i] = tf.keras.utils.to_categorical(y_train_c[i], num_coarse_classes[i])
                y_test_c[i] = tf.keras.utils.to_categorical(y_test_c[i], num_coarse_classes[i])
    else:
        y_train = to_categorical_smooth(y_train, num_classes, temperature=0.1)
        y_test = to_categorical_smooth(y_test, num_classes, temperature=0.1)
        if with_aux:
            for i in range(0, num_auxs):
                y_train_c[i] = to_categorical_smooth(y_train_c[i], num_coarse_classes[i], temperature=0.1)
                y_test_c[i] = to_categorical_smooth(y_test_c[i], num_coarse_classes[i], temperature=0.1)

    if not params['mean_std_norm']:
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
    else:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        avg = np.average(x_train, axis=(0, 1, 2))
        stddev = np.std(x_train, axis=(0, 1, 2))
        x_train = (x_train - avg) / stddev
        x_test = (x_test - avg) / stddev

    weight_decay = params['weight_decay']
    use_pretrained = params['use_pretrained']

    if use_pretrained is not None:
        # load .h5 model
        model = load_model(use_pretrained)
    else:
        if params['network'] == 'resnet50':
            if with_aux:
                model = ResnetBuilder.build_resnet_50(x_train.shape[1:], num_classes, aux_depth=aux_depth,
                                                      num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts)
            else:
                model = ResnetBuilder.build_resnet_50(x_train.shape[1:], num_classes)
        elif params['network'] == 'WRN':
            depth = params['wide_depth']
            width = params['wide_width']
            if with_aux:
                model = WideResidualNetwork(depth=depth, width=width, input_shape=x_train.shape[1:],
                                            classes=num_classes, activation='softmax', aux_depth=aux_depth,
                                            num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
                                            aux_weight_decay=aux_weight_decay, se_net=params['se_net'],
                                            aux_init='he_normal')
            else:
                model = WideResidualNetwork(depth=depth, width=width, input_shape=x_train.shape[1:],
                                            classes=num_classes, activation='softmax', aux_depth=[],
                                            num_coarse_classes=[], aux_layouts=[], se_net=params['se_net'])

            if weight_decay is not None:
                # manually add weight decay as loss
                for layer in model.layers:
                    if len(layer.losses) == 0:
                        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                            layer.add_loss(tf.keras.regularizers.l2(weight_decay)(layer.kernel))
                        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                            layer.add_loss(tf.keras.regularizers.l2(weight_decay)(layer.bias))
        elif params['network'] == 'DenseNet':
            depth = params['dense_depth']
            growth = params['dense_growth']

            decay = weight_decay if weight_decay is not None else 0
            # 100-12, 250-24, 190-40
            if with_aux:
                model = DenseNet(input_shape=x_train.shape[1:],
                                 depth=depth,  # L parameter / Depth parameter
                                 growth_rate=growth,  # k parameter
                                 bottleneck=True,  # True for DenseNet-B
                                 reduction=0.5,  # 1-theta (1-compression), >0.0 for DenseNet-C
                                 subsample_initial_block=False,  # Keep false for CIFAR
                                 weight_decay=decay,
                                 classes=num_classes, aux_depth=aux_depth,
                                 num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
                                 initialization='orthogonal',
                                 aux_init='he_normal')

            else:
                model = DenseNet(input_shape=x_train.shape[1:],
                                 depth=depth,  # L parameter / Depth parameter
                                 growth_rate=growth,  # k parameter
                                 bottleneck=True,  # True for DenseNet-B
                                 reduction=0.5,  # 1-theta (1-compression), >0.0 for DenseNet-C
                                 subsample_initial_block=False,  # Keep false for CIFAR
                                 weight_decay=decay,
                                 classes=num_classes, aux_depth=[],
                                 num_coarse_classes=[], aux_layouts=[], initialization='orthogonal')
        else:
            raise NotImplementedError("Unknown Model: " + str(params['network']))

        # stochastic gradient descent

        print('Free parameters:', model.count_params())
        update_lr = LearningRateScheduler(lambda epoch, lr: lr_scheduler(epoch, epochs, learning_rate, max_epoch))

        
        '''
        if with_aux:
            model.compile(optimizer='sgd', loss='categorical_crossentropy', loss_weights=[1.0] + aux_weight,
                          metrics=['accuracy'])

        else:
            model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        '''    
        
        if with_aux:
            model.compile(optimizer='adam', loss='categorical_crossentropy', loss_weights=[1.0] + aux_weight, metrics=['accuracy'])
        else:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
        
        

        log_call = LambdaCallback(
            on_epoch_end=lambda epoch, logs: log_combined_acc(model, x_test, y_test, cats, num_classes,
                                                              combined_accuracy_log, params['exp_combination_factor']),
            on_train_end=lambda logs: classification_analysis(model, x_test, y_test, cats,
                                                              num_classes))  # called during training after each epoch

        start_time = time.time()

        datagen = ImageDataGenerator(width_shift_range=5, height_shift_range=5, horizontal_flip=True,
                                     fill_mode='reflect')

        # train
        if with_aux:
            for epoch in range(epochs):
                print(f'Epoch {epoch + 1}/{epochs}', end=' ')
                model.fit(
                    generator_for_multi_outputs(datagen, x_train, [y_train] + y_train_c, batch_size=batch_size),
                    epochs=1, validation_data=(x_test, [y_test] + y_test_c), callbacks=[update_lr, log_call],
                    verbose=1, steps_per_epoch=steps_per_epoch, shuffle=True)
                print(f'\rduration {time.time() - start_time:.2f} s', end=' ')

        else:
            for epoch in range(epochs):
                print(f'Epoch {epoch + 1}/{epochs}', end=' ')
                model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=1, shuffle=True,
                          validation_data=(x_test, y_test), callbacks=[update_lr], verbose=1,
                          steps_per_epoch=steps_per_epoch)
                print(f'\rduration {time.time() - start_time:.2f} s', end=' ')

        end_time = time.time()
        print('duration', end_time - start_time, 's')  # 输出训练持续时间
        if params['save_model']:
            model.save('../models/cifar100_model.h5')


    if with_aux:
        score = model.evaluate(x_test, [y_test] + y_test_c, batch_size=batch_size)
        print('Test acc (fine):', score[2 + num_auxs])
        for i in range(0, num_auxs):
            print('Test acc (SSAL ' + str(i) + '):', score[3 + num_auxs + i])
        log_combined_acc(model, x_test, y_test, cats, num_classes, [],
                         exp_combination_factor=params['exp_combination_factor'])
        
        class_names = ['airport_runway', 'artificial_grassland', 'avenue', 'bare_land', 'bridge', 'city_avenue', 'city_building', 'city_green_tree', 'city_road', 'coastline', 'container', 'crossroads', 'dam', 'desert', 'dry_farm', 'forest', 'fork_road', 'grave', 'green_farmland', 'highway', 'hirst', 'lakeshore', 'mangrove', 'marina', 'mountain', 'mountain_road', 'natural_grassland', 'overpass', 'parkinglot', 'pipeline', 'rail', 'residents', 'river', 'river_protection_forest', 'sandbeach', 'sapling', 'sea', 'shrubwood', 'snow_mountain', 'sparse_forest', 'storage_room', 'stream', 'tower', 'town', 'turning_circle']

        print_per_class_accuracy(model, x_test, y_test, class_names)
    else:
        score = model.evaluate(x_test, y_test, batch_size=batch_size)
        print('Test acc:', score[1])

        class_names = ['airport_runway', 'artificial_grassland', 'avenue', 'bare_land', 'bridge', 'city_avenue', 'city_building', 'city_green_tree', 'city_road', 'coastline', 'container', 'crossroads', 'dam', 'desert', 'dry_farm', 'forest', 'fork_road', 'grave', 'green_farmland', 'highway', 'hirst', 'lakeshore', 'mangrove', 'marina', 'mountain', 'mountain_road', 'natural_grassland', 'overpass', 'parkinglot', 'pipeline', 'rail', 'residents', 'river', 'river_protection_forest', 'sandbeach', 'sapling', 'sea', 'shrubwood', 'snow_mountain', 'sparse_forest', 'storage_room', 'stream', 'tower', 'town', 'turning_circle']

        print_per_class_accuracy(model, x_test, y_test, class_names)

    del model

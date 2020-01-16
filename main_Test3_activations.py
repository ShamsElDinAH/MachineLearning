from ClassDataSet import *
from ClassModel import *
from Doplots import DoPlots_activations
import pickle
import time
import numpy as np
import seaborn as sns


data_set_1 = DataSet(data_set='german', number_of_labels=10, number_of_images=200, grayscale=True, normalize=True, contrast=True)

epochs = 20
batch_size= 10
lr = 0.0001
optimizer = Adam(lr=0.0001)
loss_function = 'categorical_crossentropy'

# parameters to test
activation_list = ['relu', 'tanh', 'sigmoid']
pooling_list = ['maxpooling', 'averagepooling']

start_time_total = time.time()
for activation in activation_list:
    for pooling in pooling_list:
        print('activation: '+activation + ' pooling: ' + pooling)
        lr = 0.0001
        batch_size = 10
        print('batch_size: ' + str(batch_size) + ' lr: '+ str(lr))
        start_time = time.time()

        # creat model
        output_num = data_set_1.number_of_labels
        my_model = ClassModel(conv_num=3, kernel_size=(3, 3), filter_number=16, dense_num=5, conv_dropout=0.2,
                              dense_dropout=0.3, activation=activation, pooling=pooling, pool_size=(2, 2),
                              hidden_num_units=600, output_layer_num=output_num)

        my_model.model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

        current_model = my_model.model

        train_x = data_set_1.training_images
        train_y = data_set_1.training_labels

        val_x = data_set_1.validation_images
        val_y = data_set_1.validation_labels

        trained_model_history = current_model.fit(train_x.reshape(-1, 64, 64, 3), train_y, epochs=epochs, batch_size=batch_size,
                                                  validation_data=(val_x, val_y))

        # current_model.save("out" + str(epochs) + "epoch_" + str(activation) + '_activation_' +str(pooling)+"_pooling_.h5")

        total_time = time.time() - start_time

        time_file = str(epochs) + 'epoch_' + str(activation) + '_activation_' + str(pooling) + '_pooling__time.npy'
        np.save(time_file, total_time)

        history_file = open(str(epochs) + 'epoch_' + str(activation) + '_activation_' + str(pooling) + '_pooling_history.pickl', 'wb')
        pickle.dump(trained_model_history.history, history_file)
        history_file.close()


DoPlots_activations(epochs, activation_list, pooling_list)

run_time_total = (time.time() - start_time_total) /60

print(run_time_total + 'min')
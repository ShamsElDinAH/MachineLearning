from ClassDataSet import *
from ClassModel import *
from Doplots import DoPlots_optimizers
import pickle
import time
import numpy as np
from keras import backend as K
import seaborn as sns



# Reset Keras Session

data_set_1 = DataSet(data_set='german', number_of_labels=10, number_of_images=200, grayscale=True, normalize=True, contrast=True)

epochs = 20
batch_size= 10
lr = 0.01
activation = 'relu'
pooling = 'averagepooling'

# parameters to test
# optimizer_list =  [Adam(lr=lr), adagrad(lr=lr), sgd(lr=lr)]
optimizer_list = [sgd(lr=lr)]
optimizer_name_list = ['Adam', 'adagrad', 'sgd']
i = 2

loss_function_list = ['mean_absolute_error', 'mean_squared_error', 'categorical_crossentropy', 'squared_hinge']
start_time_total = time.time()
# i = 0
for optimizer in optimizer_list:

    optimizer_name = optimizer_name_list[i]

    for loss_function in loss_function_list:
        print('optimizer: '+ optimizer_name + ' loss_function: ' + loss_function)
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

        time_file = str(epochs) + 'epoch_' + optimizer_name + '_optimizer_' + loss_function + '_loss_function_time.npy'
        np.save(time_file, total_time)

        history_file = open(str(epochs) + 'epoch_' + optimizer_name + '_optimizer_' + loss_function + '_loss_function_history.pickl', 'wb')
        pickle.dump(trained_model_history.history, history_file)
        history_file.close()

    # i = i+1


DoPlots_optimizers(epochs, optimizer_name_list, loss_function_list)

run_time_total = (time.time() - start_time_total) /60

print('Total running time : ' + str(round(run_time_total, 2))+ 'min')
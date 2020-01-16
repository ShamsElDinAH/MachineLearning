from ClassDataSet import *
from ClassModel import *
from Doplots import DoPlots
import pickle
import time
import numpy as np
import seaborn as sns


data_set_1 = DataSet(data_set='german', number_of_labels=10, number_of_images=200, grayscale=True, normalize=True, contrast=True)

epochs = 21
# epochs = 20

activation = 'relu'
pooling = 'averagepooling'
# optimizer = Adam(lr=0.0001)
loss_function = 'categorical_crossentropy'

# parameters to test
# batch_size_list = [5, 10, 20]
batch_size_list = [10]
lr_list = [0.1, 0.01, 0.001]

start_time_total = time.time()
for batch_size in batch_size_list:
    for lr in lr_list:
        print('activation: '+activation + ' pooling: ' + pooling)
        print('batch_size: ' + str(batch_size) + ' lr: '+ str(lr))

        start_time = time.time()

        # creat model
        output_num = data_set_1.number_of_labels
        my_model = ClassModel(conv_num=3, kernel_size=(3, 3), filter_number=16, dense_num=5, conv_dropout=0.2,
                              dense_dropout=0.3, pooling=pooling, activation=activation, pool_size=(2, 2),
                              hidden_num_units=600, output_layer_num=output_num)

        my_model.model.compile(loss=loss_function, optimizer=sgd(lr=lr), metrics=['accuracy'])

        current_model = my_model.model

        train_x = data_set_1.training_images
        train_y = data_set_1.training_labels

        val_x = data_set_1.validation_images
        val_y = data_set_1.validation_labels

        trained_model_history = current_model.fit(train_x.reshape(-1, 64, 64, 3), train_y, epochs=epochs, batch_size=batch_size,
                                                  validation_data=(val_x, val_y))

        current_model.save("out" + str(epochs) + "epoch_" + str(batch_size) +'batch_'+str(lr)+"lr_.h5")

        total_time = time.time() - start_time

        time_file = str(epochs) + 'epoch_' + str(batch_size) + 'batch_' + str(lr) + 'lr_time.npy'
        np.save(time_file, total_time)

        history_file = open(str(epochs) + 'epoch_' + str(batch_size) + 'batch_' + str(lr) + 'lr_history.pickl', 'wb')
        pickle.dump(trained_model_history.history, history_file)
        history_file.close()


DoPlots(epochs, batch_size_list, lr_list)

run_time_total = (time.time() - start_time_total) /60

print('total time: ' + str(run_time_total) + 'min')
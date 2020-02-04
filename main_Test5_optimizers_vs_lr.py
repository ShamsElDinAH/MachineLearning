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
activation = 'relu'
pooling = 'averagepooling'
loss_function = 'categorical_crossentropy'

# parameters to test
# optimizer_list =  [Adam(lr=lr), adagrad(lr=lr), sgd(lr=lr)] These must be tested one by one
# optimizer_name_list = ['Adam', 'adagrad', 'sgd']
optimizer_name = 'sgd'
learning_rate_list = [0.1, 0.01, 0.001, 0.0001]

start_time_total = time.time()


for learning_rate in learning_rate_list:
    # # Adam optimizer
    # optimizer = Adam(lr=learning_rate)
    # optimizer_name = 'Adam'

    # adagrad optimizer
    optimizer = adagrad(lr=learning_rate)
    optimizer_name = 'adagrad'

    # sgd
    # optimizer = sgd(lr=learning_rate)
    # optimizer_name = 'sgd'

    print('optimizer: ' + optimizer_name + ' learning_rate: ' + str(learning_rate))
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

    time_file = str(epochs) + 'epoch_' + optimizer_name + '_optimizer_' + str(learning_rate) + '_lr_time.npy'
    np.save(time_file, total_time)

    history_file = open(str(epochs) + 'epoch_' + optimizer_name + '_optimizer_' + str(learning_rate) + '_lr_history.pickl', 'wb')
    pickle.dump(trained_model_history.history, history_file)
    history_file.close()

    # i = i+1


# DoPlots_optimizers(epochs, optimizer_name_list, loss_function_list)

fig1 = plt
fig2 = plt
fig3 = plt

i = 1

for learning_rate in learning_rate_list:
    history_file = open(str(epochs) + 'epoch_' + optimizer_name + '_optimizer_' + str(learning_rate) + '_lr_history.pickl', 'rb')
    current_model_history = pickle.load(history_file)
    history_file.close()

    time = np.load(str(epochs) + 'epoch_' + optimizer_name + '_optimizer_' + str(learning_rate) + '_lr_time.npy')


    plt.figure(0)
    plt.subplot(311)
    fig1.plot(current_model_history['val_loss'], label='optimizer='+optimizer_name+' lr='+str(learning_rate))

    # plt.figure(2)
    plt.subplot(312)
    fig2.plot(current_model_history['val_accuracy'])

    plt.subplot(313)
    fig3.plot(i, time, marker='o')
    i = i + 1

plt.figure(0)
plt.subplot(311)
fig1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',  borderaxespad=0.)
fig1.title('Validation loss')
fig1.xlabel('Epochs')
fig1.ylabel('loss')
# fig1.axis([0, 10, 0, 1])

plt.subplot(312)
# fig2.legend()
fig2.title('validation accuracy')
fig2.xlabel('Epochs')
fig2.ylabel('accuracy')

plt.subplot(313)
# fig3.legend()
fig3.title('Time to train model')
fig3.xlabel('Test run')
fig3.ylabel('Time')

plt.tight_layout()
plt.savefig('out_comparison_optimizer_vs_lr'+optimizer_name+'.jpg')
plt.show()

run_time_total = (time.time() - start_time_total) /60

print('Total running time : ' + str(round(run_time_total, 2))+ 'min')
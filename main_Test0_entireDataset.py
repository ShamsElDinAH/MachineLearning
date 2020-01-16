from ClassDataSet import *
from ClassModel import *
from Doplots import DoPlots_activations
import pickle
import time
import numpy as np
import seaborn as sns

test_name = '43_lables_2000_images'

start_time_total = time.time()

data_set_1 = DataSet(data_set='german', number_of_labels=4, number_of_images=1000, augment_dataset=True, grayscale=True, normalize=True, contrast=True)

data_set_list = [data_set_1, data_set_1]

epochs = 20
lr = 0.01
batch_size = 10

activation = 'relu'
optimizer = sgd(lr=lr)
pooling = 'averagepooling'

print('activation: '+ activation + ' pooling: ' + pooling)
print('batch_size: ' + str(batch_size) + ' lr: '+ str(lr))
start_time = time.time()

# create model
output_num = data_set_1.number_of_labels
my_model = ClassModel(conv_num=3, kernel_size=(3, 3), filter_number=16, dense_num=5, conv_dropout=0.2,
                      dense_dropout=0.3, activation=activation, pooling=pooling, pool_size=(2, 2),
                      hidden_num_units=600, output_layer_num=output_num)

my_model.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

current_model = my_model.model

train_x = data_set_1.training_images
train_y = data_set_1.training_labels

val_x = data_set_1.validation_images
val_y = data_set_1.validation_labels

trained_model_history = current_model.fit(train_x.reshape(-1, 64, 64, 3), train_y, epochs=epochs, batch_size=batch_size,
                                          validation_data=(val_x, val_y))

current_model.save(test_name+"_.h5")

total_time = time.time() - start_time

time_file = test_name + '_time.npy'
np.save(time_file, total_time)

history_file = open(test_name + '_history.pickl', 'wb')
pickle.dump(trained_model_history.history, history_file)
history_file.close()

run_time_total = (time.time() - start_time_total) /60

print('Total running time : ' + str(round(run_time_total, 2))+ 'min')
# ------------------ plotting
fig1 = plt
fig2 = plt
fig3 = plt

history_file = open(test_name + '_history.pickl', 'rb')
current_model_history = pickle.load(history_file)
history_file.close()

time = np.load(time_file)

plt.figure(0)
plt.subplot(311)
fig1.plot(current_model_history['val_loss'], label='batch='+str(batch_size)+' lr='+str(lr))

# plt.figure(2)
plt.subplot(312)
fig2.plot(current_model_history['val_accuracy'], label='batch='+str(batch_size)+' lr='+str(lr))

plt.subplot(313)
fig3.plot(1, time, marker='o', label='batch='+str(batch_size)+' lr='+str(lr))

plt.figure(0)
plt.subplot(311)
# fig1.legend()
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
fig3.legend()
fig3.title('Time to train model')
fig3.xlabel('Test run')
fig3.ylabel('Time')

plt.tight_layout()
plt.savefig('out_comparison.jpg')
plt.show()

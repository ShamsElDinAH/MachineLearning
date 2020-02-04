from ClassDataSet import *
from ClassModel import *
from Doplots import DoPlots_activations
import pickle
import time
import numpy as np
import seaborn as sns



start_time_total = time.time()

number_of_labels = 43
number_of_images = 2000
#
# data_set_1 = DataSet(data_set='german', number_of_labels=number_of_labels, number_of_images=number_of_images, augment_dataset=True, grayscale=True, normalize=True, contrast=True)
#
test_name = str(number_of_labels)+'_lables_' + str(number_of_images) + '_images'

# data_set_list = [data_set_1, data_set_1]

dataset_file = open(test_name+'_dataset.pickl', 'rb')
data_set_1 = pickle.load(dataset_file)
dataset_file.close()

epochs = 20
lr = 0.01
batch_size = 10

activation = 'relu'
optimizer = sgd(lr=lr)
pooling = 'averagepooling'
loss = 'categorical_crossentropy'

print('activation: '+ activation + ' pooling: ' + pooling)
print('batch_size: ' + str(batch_size) + ' lr: '+ str(lr))
start_time = time.time()

# create model
output_num = data_set_1.number_of_labels
my_model = ClassModel(conv_num=3, kernel_size=(3, 3), filter_number=16, dense_num=5, conv_dropout=0.2,
                      dense_dropout=0.3, activation=activation, pooling=pooling, pool_size=(2, 2),
                      hidden_num_units=600, output_layer_num=output_num)

my_model.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

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

### Prdicting the class
pred_classes = current_model.predict_classes(data_set_1.test_images)
print(pred_classes)

prediction_file = open(test_name + '_predictions.pickl', 'wb')
pickle.dump(pred_classes, prediction_file)
prediction_file.close()

### Evaluating the model
model_evaluation = current_model.evaluate(data_set_1.test_images, data_set_1.test_labels)
print(model_evaluation)

evaluation_file = open(test_name + '_evaluation.pickl', 'wb')
pickle.dump(model_evaluation, evaluation_file)
evaluation_file.close()


# ### Prdicting the class
# pred_classes = model.predict_classes(test_x)
# print(pred_classes)
# ### Evaluating the model
# model_evaluation = model.evaluate(test_x, test_y)
# print(model_evaluation)

print('Total running time : ' + str(round(run_time_total, 2))+ 'min')
# ------------------ plotting

fig1 = plt
fig2 = plt

history_file = open(test_name + '_history.pickl', 'rb')
current_model_history = pickle.load(history_file)
history_file.close()

wow = range(1, 21)
plt.figure(0, figsize=(10, 8))
plt.subplot(211)
fig1.plot(wow, current_model_history['val_loss'], label='Validation')
fig1.plot(wow, current_model_history['loss'], label='Training')

# plt.figure(2)
plt.subplot(212)
fig2.plot(wow, current_model_history['val_accuracy'])
fig2.plot(wow, current_model_history['accuracy'], label='Training')

plt.figure(0)
# plt.title('43 Label 2000 Images Result')
plt.subplot(211)
fig1.legend()
fig1.title('43 Label 2000 Images Result \n loss' )
fig1.xlabel('Epochs')
fig1.ylabel('loss')
# fig1.suptitle('43 Label 2000 Images Result')
# fig1.axis([1, 20, 0, 1])
fig1.xticks(range(1, 21))
plt.subplot(212)
# fig2.legend()
fig2.title('accuracy')
fig2.xlabel('Epochs')
fig2.ylabel('accuracy')
fig2.xticks(range(1,21))

plt.tight_layout()
# plt.savefig('out_comparison.jpg')
plt.show()

# fig1 = plt
# fig2 = plt
# fig3 = plt
#
# history_file = open(test_name + '_history.pickl', 'rb')
# current_model_history = pickle.load(history_file)
# history_file.close()
#
# time = np.load(time_file)
# wow = range(1, 21)
# plt.figure(0, figsize=(10, 8))
# plt.subplot(311)
# fig1.plot(wow, current_model_history['val_loss'], label='Validation')
# fig1.plot(wow, current_model_history['loss'], label='Training')
#
# # plt.figure(2)
# plt.subplot(312)
# fig2.plot(wow, current_model_history['val_accuracy'])
# fig2.plot(wow, current_model_history['accuracy'], label='Training')
#
# plt.subplot(313)
# fig3.plot(1, time, marker='o')
#
# plt.figure(0)
# # plt.title('43 Label 2000 Images Result')
# plt.subplot(311)
# fig1.legend()
# fig1.title('43 Label 2000 Images Result \n loss' )
# fig1.xlabel('Epochs')
# fig1.ylabel('loss')
# # fig1.suptitle('43 Label 2000 Images Result')
# # fig1.axis([1, 20, 0, 1])
# fig1.xticks(range(1, 21))
# plt.subplot(312)
# # fig2.legend()
# fig2.title('accuracy')
# fig2.xlabel('Epochs')
# fig2.ylabel('accuracy')
# fig2.xticks(range(1,21))
#
# plt.subplot(313)
# # fig3.legend()
# fig3.title('Time to train model')
# fig3.xlabel('Test run')
# fig3.ylabel('Time')
#
# plt.tight_layout()
# # plt.savefig('out_comparison.jpg')
# plt.show()

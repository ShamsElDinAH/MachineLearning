import pickle
import matplotlib.pyplot as plt
import numpy as np

def DoPlots(epochs, batch_size_list, lr_list):

    fig1 = plt
    fig2 = plt
    fig3 = plt

    i = 1

    for batch_size in batch_size_list:
        for lr in lr_list:
            history_file = open(str(epochs) + 'epoch_' + str(batch_size) + 'batch_' + str(lr) + 'lr_history.pickl', 'rb')
            current_model_history = pickle.load(history_file)
            history_file.close()

            time = np.load(str(epochs) + 'epoch_' + str(batch_size) + 'batch_' + str(lr) + 'lr_time.npy')


            plt.figure(0)
            plt.subplot(311)
            fig1.plot(current_model_history['val_loss'], label='batch='+str(batch_size)+' lr='+str(lr))

            # plt.figure(2)
            plt.subplot(312)
            fig2.plot(current_model_history['val_accuracy'], label='batch='+str(batch_size)+' lr='+str(lr))

            plt.subplot(313)
            fig3.plot(i, time, marker='o', label='batch='+str(batch_size)+' lr='+str(lr))
            i = i + 1

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


def DoPlots_activations(epochs, activation_list, pooling_list):

    fig1 = plt
    fig2 = plt
    fig3 = plt

    i = 1

    for activation in activation_list:
        for pooling in pooling_list:
            history_file = open(str(epochs) + 'epoch_' + str(activation) + '_activation_' + str(pooling) + '_pooling_history.pickl', 'rb')
            current_model_history = pickle.load(history_file)
            history_file.close()

            time = np.load(str(epochs) + 'epoch_' + str(activation) + '_activation_' + str(pooling) + '_pooling__time.npy')


            plt.figure(0)
            plt.subplot(311)
            fig1.plot(current_model_history['val_loss'])

            # plt.figure(2)
            plt.subplot(312)
            fig2.plot(current_model_history['val_accuracy'])

            plt.subplot(313)
            fig3.plot(i, time, marker='o', label='activation='+str(activation)+' pooling='+str(pooling))
            i = i + 1

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
    plt.savefig('out_comparison_activation.jpg')
    plt.show()
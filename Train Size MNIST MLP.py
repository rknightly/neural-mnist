from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np

# Prepare Data
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'Total train samples')
print(x_test.shape[0], 'Total test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def test_model(sample_count):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    x_train_small = x_train[:][:sample_count]
    y_train_small = y_train[:][:sample_count]
    print('Training Shape:', x_train_small.shape)

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    training = model.fit(x_train_small, y_train_small,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))


    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Convert to percent
    training.history['acc'] = 100.0 * np.array(training.history['acc']) 
    training.history['val_acc'] = 100.0 * np.array(training.history['val_acc']) 
    
    # Start the accuracy at 10% before training started
    training.history['acc'] = np.insert(training.history['acc'], 0, 10.0)
    training.history['val_acc'] = np.insert(training.history['val_acc'], 0, 10.0)
    
    return training

sixty = test_model(60)
six_hundred = test_model(600)
six_thousand = test_model(6000)
sixty_thousand = test_model(60000)

# Plot Results
histories = [sixty.history, six_hundred.history, six_thousand.history, sixty_thousand.history]

# Bar Graphs
plt.figure(0)
plt.title('Training Data Size vs. Training Accuracy')
plt.ylabel('Training Accuracy %')
plt.grid(axis='y', linestyle='dashed')
bar_titles = ('60 Samples', '600 Samples', '6,000 Samples', '60,000 Samples')
accuracy = [history['acc'][-1] for history in histories]
x_nums = np.arange(len(bar_titles))
plt.bar(x_nums, accuracy, align='center')
plt.xticks(x_nums, bar_titles)

plt.figure(1)
plt.title('Training Data Size vs. Validation Accuracy')
plt.ylabel('Training Accuracy %')
plt.grid(axis='y', linestyle='dashed')
bar_titles = ('60 Samples', '600 Samples', '6,000 Samples', '60,000 Samples')
accuracy = [history['val_acc'][-1] for history in histories]
x_nums = np.arange(len(bar_titles))
plt.bar(x_nums, accuracy, align='center')
plt.xticks(x_nums, bar_titles)

# Line graphs
x_data = np.arange(0, epochs+1)

plt.figure(2)
plt.title("Train Accuracy Vs. Epoch")
plt.ylabel('Accuracy')  
plt.xlabel('Epoch')
plt.grid(linestyle='dashed')
plt.xlim(1, epochs)
plt.xticks(np.arange(0, epochs+1, step=2.0))
for history in histories:
    plt.plot(x_data, history['acc'])
plt.legend(['60 Samples', '600 Samples', '6,000 Samples', '60,000 Samples'], loc='lower right') 

plt.figure(3)
plt.title("Validation Accuracy Vs. Epoch")
plt.ylabel('Accuracy')  
plt.xlabel('Epoch')
plt.grid(linestyle='dashed')
plt.xlim(1, epochs)
plt.xticks(np.arange(0, epochs+1, step=2.0))
for history in histories:
    plt.plot(x_data, history['val_acc'])
plt.legend(['60 Samples', '600 Samples', '6,000 Samples', '60,000 Samples'], loc='lower right') 


import numpy as np
from tensorflow import keras

# load
with open('train.csv', 'r') as file:
    train_lines = file.readlines()

# process data
X_train = []
y_train = []
for line in train_lines[1:]:
    line = line.strip().split(';')

    X_train.append(line[:-1])
    y_train.append(line[-1])

# convert to numpy
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train)

# encode the class labels
class_mapping = {'classA': 0, 'classB': 1}
y_train_encoded = np.array([class_mapping[label] for label in y_train])


# load the test data
with open('test.csv', 'r') as file:
    test_lines = file.readlines()

# process
X_test = []
y_test = []
for line in test_lines[1:]:
    line = line.strip().split(';')

    X_test.append(line[:-1])
    y_test.append(line[-1])

# convert to numpy
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)

# encode the class labels
y_test_encoded = np.array([class_mapping[label] for label in y_test])


# initializing
classifier = keras.models.Sequential()

# adding layers
classifier.add(keras.layers.Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))

classifier.add(keras.layers.Dense(units=6, kernel_initializer='uniform', activation='relu'))

classifier.add(keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# compiling
classifier.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# train
classifier.fit(X_train, y_train_encoded, batch_size=10, epochs=100)

# evaluating the model on the training set
train_loss, train_accuracy = classifier.evaluate(X_train, y_train_encoded)
print("Training Set - Loss:", train_loss, "- Accuracy:", train_accuracy)

# evaluating the model on the test set
test_loss, test_accuracy = classifier.evaluate(X_test, y_test_encoded)
print("Test Set - Loss:", test_loss, "- Accuracy:", test_accuracy)

# save
classifier.save('trained_model_test.hdf5')

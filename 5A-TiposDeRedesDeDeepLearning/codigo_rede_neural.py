from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

## dado que o range de valores possível para um pixel vai de 0-255
## escalonamos os valores entre 0-1
## esse processo torna nosso modelo menos variante a pequenas alterações
x_train = x_train / 255
x_test = x_test / 255

model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (5,5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
## usada na camada de saída do classificador, onde realmente estamos tentando
## gerar as probabilidades para definir a classe de cada entrada

optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

# reduz o parâmetro de learning rate se não houver
# melhoras em determinado número de epocas
# útil para encontrar o mínimo global
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
	patiente=3,
	verbose=1,
	factor=0.5,
	min_lr=0.00001
	)

batch_size = 32
epochs = 10

history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.2, verbose = 1, callbacks=[learning_rate_reduction])
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
range_epochs = range(1, len(acc) + 1)

plt.style.use('default')
accuracy_val = plt.plot(range_epochs, val_acc, label='Acurácia no conjunto de validação')
accuracy_train = plt.plot(range_epochs, acc, label='Acurácia no conjunto de treinamento', color='r')
plt.setp(accuracy_val, linewidth=2.0)
plt.setp(accuracy_train, linewidth=2.0)
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.show()

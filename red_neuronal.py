from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomUniform

# Inicializando la Red Neuronal
neural_network = Sequential()

# kernel_initializer Define la forma como se asignará los Pesos iniciales Wi
initial_weights = RandomUniform(minval = -1.0, maxval = 1.0)
num_neuronas_entrada = X_train.shape[1]

# Agregado la Capa de entrada y la primera capa oculta
# 8 Neuronas en la capa de entrada y 5 Neuronas en la primera capa oculta
neural_network.add(Dense(units = 5, kernel_initializer = initial_weights, activation = 'sigmoid', input_dim = num_neuronas_entrada))

# Agregando capa oculta
neural_network.add(Dense(units = 4, kernel_initializer = initial_weights, activation = 'relu'))

# Agregando capa oculta
neural_network.add(Dense(units = 3, kernel_initializer = initial_weights, activation = 'relu'))

# Agregando capa de salida
neural_network.add(Dense(units = 1, kernel_initializer = initial_weights, activation = 'sigmoid'))


# Imprimir Arquitectura de la Red
neural_network.summary()


# Compilando la Red Neuronal
# optimizer: Algoritmo de optimización | binary_crossentropy = 2 Classes
# loss: es el error que da el modelo, -valor + eficiente es el modelo
# accuracy: clasificación correcta que realiza el modelo

neural_network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenamiento
# batch_size = numero de datos que se introducen en la red para que entre nuestro modelo
# epoch= ciclos que duran el entrenamiento

history = neural_network.fit(X_train, y_train, batch_size = 16, epochs = 400)


# Haciendo predicción de los resultados del Test
y_pred = neural_network.predict(X_test)
y_pred_norm = (y_pred > 0.5)

y_pred_norm = y_pred_norm.astype(int)
y_test = y_test.astype(int)

# 20 primeros resultados a comparar
print("\nPredicciones (20 primeros):")
print("\n\tReal", "\t", "Predicción(N)","\t", "Predicción(O)")
for i in range(20):
    print(i, '\t', y_test[i], '\t ', y_pred_norm[i], '\t \t', y_pred[i])


# Métricas:
# Aplicando la Matriz de Confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_norm)
print ("\nMatriz de Confusión: \n", cm)

TP = cm[1,1]
FP = cm[0,1]
TN = cm[0,0]
FN = cm[1,0]

# Accuracy
accuracy = (TP+TN)/(TP+TN+FP+FN)
# Sensitivity/Recall
sensitivity = TP/(TP+FN)


# Accuracy: representa el porcentaje total de valores correctamente clasificados
print("Accuracy: ","({:.2%})".format(accuracy))
# Sensibilidad: Representa la fracción de verdaderos positivos
print("Sensitivity:","({:.2%})".format(sensitivity))

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding

if __name__ == "__main__":
    model = Sequential()
    embedding = Embedding(100, 10, input_length=1)  # Embedding(5, 1, input_length=5)

    model.add(Embedding(5, 2, input_length=5))

    input_array = np.random.randint(5, size=(1, 5))

    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)
    print(input_array)
    print(output_array)

from loadLibraries import *
# fix random seed for reproducibility


def load_dataset():
    # load pima indians dataset
    ds = numpy.loadtxt('data/pima-indians-diabetes.csv', delimiter=",")
    # split into input (X) and output (Y) variables
    X = ds[:, 0:8]
    Y = ds[:, 8]
    return (ds, X, Y)


def define_model():
    # We create a Sequential model and add layers one at a time
    # until we are happy with our network topology.
    # We use a fullt connected (dense) 3-layer network
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def compile_model():
    # Compile model
    # Compiling the model uses the efficient numerical libraries
    # under the covers (the so-called backend) such as Theano or
    # TensorFlow. The backend automatically chooses the best way to
    # represent the network for training and making predictions to
    # run on your hardware, such as CPU or GPU or even distributed
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def fit_model():
    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10)


def evaluate_model():
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def make_prediction():
    # calculate predictions
    predictions = model.predict(X)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    print(rounded)


if __name__ == '__main__':
    print('Revving up ...')
    numpy.random.seed(7)

    print('Loading dataset ... ')
    dataset, X, Y = load_dataset()

    print('Defining model ...')
    model = define_model()

    print('And now compiling model ...')
    compile_model()

    print("Let's see if we can fi it ...")
    fit_model()

    print('Time to evaluate ...')
    evaluate_model()

    print('Making predictions ... finally')
    make_prediction()

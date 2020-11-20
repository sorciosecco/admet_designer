
#import tensorflow as tf
#from tensorflow import keras

    
    # TensorFlow (TF)
    if model_type=="TF":
        tf.set_random_seed(666)
        np.random.seed(666)
        X = np.array(self.X_train)
        inp = keras.Input(shape=(X.shape[1],))
        
        x = keras.layers.Dense(80, activation=tf.nn.relu)(inp)
        x = keras.layers.Dense(50, activation=tf.nn.relu)(x)
        #x = keras.layers.Dense(50, activation=tf.nn.relu)(x)
        
        outputs=[keras.layers.Dense(2, activation=tf.nn.softmax)(x)]
        model=keras.Model(inp, outputs)
        model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

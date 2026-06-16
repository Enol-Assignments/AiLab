import tensorflow as tf


def Train(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=64):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(x_test, y_test))


def Test(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[Test] Accuracy on {len(x_test)} test images: {accuracy * 100:.2f}%")

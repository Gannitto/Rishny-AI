import tensorflow as tf
try:
    loaded_model = tf.keras.models.load_model('model.keras')
    print("Model loaded successfully!")
except:
    print("Error - retry saving.")
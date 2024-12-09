import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy("mixed_float16")

# Load your model
model = tf.keras.models.load_model("model/Llama-3.2-1B-Instruct-f16.gguf")

# Save the model in FP16
model.save("/model")

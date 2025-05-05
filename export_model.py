# export_model.py
import tensorflow as tf

model = tf.keras.models.load_model('best_sports_classifier.keras')
version = 1
export_path = f"models/sports_classifier/{version}"
model.save(export_path, save_format="tf")
print(f"SavedModel exported to {export_path}")

import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 40  # must match training

# Load trained model
model = tf.keras.models.load_model("lstm_model")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_sarcasm(text: str) -> bool:
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    prediction = model.predict(padded)[0][0]
    return prediction > 0.5

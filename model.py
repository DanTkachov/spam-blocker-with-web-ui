import random
import pickle
import tensorflow as tf


def dummy_predict():
    return random.choice([True, False])


def load_model(model_path: str = 'model.pkl'):
    try:
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
            return loaded_model
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
        print(f"Error loading the model: {str(e)}")
        return None


def model_predict(model, email, vectorizer=None) -> str:
    if model is None:
        print("Model failed to load; check model.pkl file")
    elif vectorizer is None:
        print("Vectorizer not set or failed to load; check vectorizer.pkl file")
    else:
        try:
            print(email, type(email))
            email_tfidf = vectorizer.transform([email])
            print("Transformed email shape:", email_tfidf.shape)
            pred = model.predict(email_tfidf)[0]
            print("(model.py) Prediction: ", pred)
            return "Spam" if pred >= 0.4 else "Not Spam"
        except (tf.errors.InvalidArgumentError) as e:
            return f"Error: no words or phrases in the input were present in the training data. Try again.\n\nFull error: {e}"


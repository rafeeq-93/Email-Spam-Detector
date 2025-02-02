import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os


class SpamDetector:
    def __init__(self, model_path='spam_model.pkl', vectorizer_path='vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer = None
        self.model = None

        print("Initializing SpamDetector...")
        self.load_model()

    def load_data(self):
        """Load the dataset from a CSV file."""
        try:
            print("Loading dataset...")
            df = pd.read_csv('email.csv', encoding='ISO-8859-1')  # Adjust encoding if necessary
            print("Dataset loaded successfully.")
            print("Columns in the dataset:", df.columns)
            print("Data types:\n", df.dtypes)
            return df
        except FileNotFoundError:
            print("Error: Dataset 'email.csv' not found.")
            raise
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def train(self):
        """Train the spam detection model."""
        try:
            spam_df = self.load_data()
            print("Training the model...")

            # Vectorize the text data using the correct column names
            self.vectorizer = CountVectorizer()
            X = self.vectorizer.fit_transform(spam_df['Text'])  # Corrected column name
            y = spam_df['Spam']  # Corrected column name

            # Train the Naive Bayes classifier
            self.model = MultinomialNB()
            self.model.fit(X, y)

            # Save the model and vectorizer to disk
            self.save_model()
            print("Training completed and model saved.")
        except Exception as e:
            print(f"Error during training: {e}")

    def save_model(self):
        """Save the trained model and vectorizer to disk."""
        try:
            print("Saving model and vectorizer...")
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)
            print(f"Model saved to {self.model_path}")
            print(f"Vectorizer saved to {self.vectorizer_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Load the model and vectorizer if they exist."""
        print("Loading model if available...")
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            try:
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                print("Model and vectorizer loaded successfully.")
            except (FileNotFoundError, IOError) as e:
                print(f"Error loading model: {e}")
                self.train()  # Train if loading fails
        else:
            print("Model files not found, training a new model...")
            self.train()

    def predict(self, text):
        """Make a prediction based on input text."""
        if self.vectorizer is None or self.model is None:
            print("Model or vectorizer is not loaded. Cannot predict.")
            return None

        text_vector = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vector)
        return prediction[0]


# Example usage
if __name__ == "__main__":
    detector = SpamDetector()
    # Example prediction
    print("Predicting...")
    print(detector.predict("Congratulations! You've won a $1000 cash prize!"))
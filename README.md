 # Simple RNN for Sentiment Analysis on IMDB Dataset
### Overview
This project demonstrates how to build a simple Recurrent Neural Network (RNN) using TensorFlow/Keras to perform sentiment analysis on the IMDB movie review dataset. The model is trained to predict whether a given movie review is positive or negative.

### Dataset
The project uses the IMDB dataset, which contains 25,000 highly polarized movie reviews for training and 25,000 for testing. The reviews have been preprocessed, and each review is encoded as a sequence of word indices.

### Model Architecture
Embedding Layer: Transforms word indices into dense vectors of fixed size.
SimpleRNN Layer: Processes the input sequences and captures temporal dependencies.
Dense Layer: Outputs the probability that a review is positive.

1. Load and preprocess the IMDB dataset.
2. Build a simple RNN model using Keras.
3. Train the model on the training data.
4. Evaluate the model on the test data.
5. Predict sentiment for a custom review

After training, you can modify the script to predict the sentiment of custom movie reviews. The script includes a function to preprocess the text and predict its sentiment using the trained model.

### Save the trained model:

Ensure that the model is saved after training so it can be loaded by the Streamlit app.

### Streamlit App Overview

User Input: Enter a movie review in a text box.
Prediction: The app displays whether the review is predicted to be positive or negative.
Real-time Feedback: The prediction updates in real-time as the user inputs different reviews.

# Import necessary libraries
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

import pytube
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import moviepy.editor as mp  # Import moviepy module
import speech_recognition as sr
# import tensorflow.compat.v1 as tf

# Function to extract features from video content
def extract_features(video_url):
    # Extract video ID from the URL
    video_id = None
    if 'youtube.com/watch?v=' in video_url:
        video_id = video_url.split('youtube.com/watch?v=')[1].split('&')[0]
    elif 'youtu.be/' in video_url:
        video_id = video_url.split('youtu.be/')[1].split('?')[0]

    if video_id is None:
        print("Invalid YouTube URL")
        return None

    # Download the video
    yt = pytube.YouTube("https://www.youtube.com/watch?v=" + video_id)
    stream = yt.streams.first()
    stream.download(filename='video.mp4')

    # Extract text from video using speech recognition
    video = mp.VideoFileClip('video.mp4')
    audio = video.audio
    audio.write_audiofile('audio.wav')
    r = sr.Recognizer()
    
    value = ""  # Initialize with a default value
    try:
        with sr.AudioFile('audio.wav') as source:
            audio_data = r.record(source)
            value = r.recognize_google(audio_data)

        print("\nGoogle Text: {}".format(value) + "\n")
       
    except sr.UnknownValueError:
        print("Oops! Didn't catch that\n")
        
    except sr.RequestError as e:
        print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e) + "\n")

    # try:
    #     with sr.AudioFile('audio.wav') as source:
    #         audio_data = r.record(source)
    #         value = r.recognize_sphinx(audio_data)

    #     print("Sphinx Text: {}".format(value) + "\n")
    
    # except sr.UnknownValueError:
    #     print("Sphinx could not understand audio\n")
    # except sr.RequestError as e:
    #     print("Sphinx error; {0}".format(e) + "\n")

    # try:
    #     with sr.AudioFile('audio.wav') as source:
    #         audio_data = r.record(source)
    #         value = r.recognize_tensorflow(audio_data)

    #     print("Tensorflow Text: {}".format(value) + "\n")
    
    # except sr.UnknownValueError:
    #     print("Tensorflow could not understand audio\n")
    # except sr.RequestError as e:
    #     print("Tensorflow error; {0}".format(e) + "\n")

    try:
        with sr.AudioFile('audio.wav') as source:
            audio_data = r.record(source)
            value = r.recognize_whisper(audio_data)

        print("Whisper Text: {}".format(value) + "\n")
    
    except sr.UnknownValueError:
        print("Whisper could not understand audio\n")
    except sr.RequestError as e:
        print("Whisper error; {0}".format(e) + "\n")

    return value

# Function to train a model to determine productivity
def train_model(features, labels):
    vectorizer = TfidfVectorizer()
    features_vectorized = vectorizer.fit_transform(features)
    model = LogisticRegression()
    model.fit(features_vectorized, labels)
    return model, vectorizer

# Function to predict productivity of a video
def predict_productivity(model, vectorizer, video_url):
    # Extract features from the video
    text = extract_features(video_url)
    # Vectorize the text using the same vectorizer used during training
    text_vectorized = vectorizer.transform([text])
    # Use the trained model to predict productivity
    prediction = model.predict(text_vectorized)

    print(prediction)

    return prediction

# Main function
def main():
    # Example dataset
    videos = [
        {"url": "https://www.youtube.com/watch?v=pDoALM0q1LA", "label": 1},  # Example of a productive video
        {"url": "https://www.youtube.com/watch?v=Bt48m9fZhmM", "label": 0},  # Example of a non-productive video
        # Add more videos as needed
    ]

    # Extract features and labels from the dataset
    features = []
    labels = []
    for video in videos:
        text = extract_features(video['url'])
        features.append(text)
        labels.append(video['label'])

    # Train your model
    model, vectorizer = train_model(features, labels)

    # Example usage:
    video_url = "https://www.youtube.com/watch?v=9e9D7ABgHpU"
    prediction = predict_productivity(model, vectorizer, video_url)
    if prediction == 1:
        print("Predicted Productivity: Productive")
    else:
        print("Predicted Productivity: Not Productive")

if __name__ == "__main__":
    main()
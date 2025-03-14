# Image Classifier Chatbot

This project is an interactive image classification system that allows users to upload images and chat with an AI about the classifications. Built using Streamlit, OpenAI's GPT, and a machine learning model for image recognition, this application enables users to receive AI-generated insights about their images in real time.

## Features
- Upload images for classification.
- View predicted labels for the uploaded image.
- Chat with the AI about the classification results.
- Interactive and user-friendly web interface powered by Streamlit.

## Installation

To set up and run the project locally, follow these steps:

1. Clone this repository:
   ```sh
   git clone https://github.com/Konyanj0278/COMP-680/tree/main
   cd COMP-680
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the following Python command for starting an SMTP server locally.
   ```sh
   python -m aiosmtpd -n
   ```
4. Run the application, using a new terminal:
   ```sh
   streamlit run app.py
   ```

## Dependencies
Ensure you have the following installed (managed via `requirements.txt`):
- Streamlit
- TensorFlow/Keras or another image classification model
- Any additional necessary Python libraries

## Usage
- Start the application using the command above.
- Upload an image to receive classification predictions.
- Use the chat interface to ask questions about the classification results.

## License
This project is licensed under the MIT License.


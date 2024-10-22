# SMS SPAM Detector
## Introduction
Welcome to the SMS Spam Detector project! This project is designed to automatically detect and classify SMS messages as either spam or non-spam (ham). The model is built using machine learning techniques, and the project includes a simple web application that allows users to input a message and receive a prediction.

The project was inspired by the increasing prevalence of malicious spam messages and the desire to create a tool that can help protect users from potential threats.

## Features
Spam Detection: Classifies SMS messages as either spam or non-spam.

Web Interface: Simple and intuitive web application for testing the model.

Machine Learning: Model trained using a dataset of SMS messages with a focus on accuracy and generalization.

## Dataset
The dataset used for training the model was provided by Ms. Sandhya Mishra and Dr. Devpriya Soni, containing labeled SMS messages as either spam or non-spam.

## Technologies and Libraries Used
Python: Core language used for building the model.

Pandas: Data manipulation and analysis.

NumPy: Numerical operations and handling arrays.

scikit-learn: Machine learning library used for model development, pipeline creation, and data preprocessing.

imblearn: Used for handling imbalanced datasets with the RandomOverSampler.

NLTK (Natural Language Toolkit): For text preprocessing, including tokenization and lemmatization.

Matplotlib: Used for plotting and visualizing data distributions.

Flask: Backend framework used for serving the model predictions.

React: Frontend library used for building the web interface.

Joblib: Used for saving and loading the trained model.

## Setup Instructions
### 1. Clone Repository
First, clone the repository to your local machine:

    git clone https://github.com/your-username/SMSSpamDetector.git
    cd SMSSpamDetector

### 2. Backend Setup
1. Create and activate a virtual environment:
  
       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Install the required Python packages:

       pip install pandas numpy scikit-learn imbalanced-learn nltk matplotlib Flask joblib

3. Run the Flask Server:

       cd backend
       python app.py

### 3. Frontend Setup
1. Navigate to the frontend directory:
      
       cd ../frontend
      
2. Install the required Node.js packages:

       npm install

3. Start the React development server:

       npm start

### 4. Acces the Application
Once both the Flask backend and React frontend are running, you can access the web application at:

       http://localhost:3000

## Usage
Enter an SMS message into the input box on the web interface.

Click "Find out!" to see whether the message is classified as spam or non-spam.

## Model Performance
The initial model achieved an accuracy of 94% on the training dataset. However, due to the imbalance in the dataset, additional work is needed to improve the model's performance on unseen data. The current model uses RandomOverSampler to balance the dataset during training.

## Future Work
Model Improvement: Enhance the model's ability to generalize to new, unseen messages.


Mobile Application: Develop a mobile app version of the detector.

Deployment: Deploy the web application on a cloud platform for public use.

Additional Features: Implement more sophisticated techniques for spam detection, such as deep learning models.

## Acknowledgments
Special thanks to Ms. Sandhya Mishra and Dr. Devpriya Soni for providing the SMS phishing dataset used in this project.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to suggest improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.



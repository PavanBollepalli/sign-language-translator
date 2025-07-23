# HandShake: AI-Powered Sign Language Translator

![ASL Translation](https://img.shields.io/badge/ASL-Translation-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4+-orange)
![React](https://img.shields.io/badge/React-18.3+-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)

HandShake is a web application that uses computer vision and machine learning to recognize and translate American Sign Language (ASL) gestures in real-time. The project consists of a React frontend and a Python Flask backend.

## 🚀 Performance Metrics

Our model achieves impressive performance on ASL gesture recognition:

- **Accuracy**: 96.88% (training) and 93.04% (validation)
- **Loss**: 0.2410 (training) and 0.2090 (validation)

## ✨ Features

- Real-time sign language detection and translation
- Webcam integration for capturing hand gestures
- Speech synthesis for vocalizing detected signs
- Adjustable confidence threshold for detection accuracy
- Visual feedback with detection confidence scores
- Cross-platform compatibility (works in modern browsers)

## 🏗️ Project Structure

```
HandShake/
├── backend/               # Python Flask backend
│   ├── api.py             # REST API endpoints
│   ├── data_loader.py     # Dataset loading utilities
│   ├── inference.py       # Model inference code
│   ├── model.py           # CNN model architecture
│   ├── requirements.txt   # Python dependencies
│   ├── train.py           # Model training script
│   └── webcam_demo.py     # Standalone webcam demo
├── public/                # Static assets
├── src/                   # React frontend
│   ├── pages/             # React page components
│   ├── services/          # API service layer
│   ├── App.tsx            # Main App component
│   ├── index.css          # Global styles
│   └── main.tsx           # Application entry point
├── index.html             # HTML entry point
├── package.json           # NPM dependencies
├── tsconfig.json          # TypeScript configuration
└── vite.config.ts         # Vite configuration
```

## 📋 Prerequisites

- Node.js (v18+)
- Python (v3.8+)
- Webcam
- TensorFlow/Keras (for model training)

## 🛠️ Setup and Installation

### Backend Setup

1. Install Python dependencies:

```bash
cd backend
pip install -r requirements.txt
```

2. Train the ASL detection model (if you don't have a pre-trained model):

```bash
python train.py --data_dir path/to/asl_dataset --epochs 20 --img_size 64
```

3. Start the Flask API server:

```bash
python api.py --port 5000
```

### Frontend Setup

1. Install Node.js dependencies:

```bash
npm install
```

2. Update browserslist database (to fix warnings):

```bash
npx update-browserslist-db@latest
```

3. Start the development server:

```bash
npm run dev
```

## 📱 Usage

1. Open your browser and navigate to `http://localhost:5173/`
2. Allow camera access when prompted
3. Position your hand in front of the camera and make ASL signs
4. The application will display the detected letters and build words as you sign
5. Adjust the confidence threshold slider if needed for better detection

## 🧠 Training Your Own Model

To train your own sign language detection model, you'll need a dataset of ASL hand gestures. The dataset should be organized with subdirectories for each letter/sign, each containing image examples:

```
asl_dataset/
├── A/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── B/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── ...
```

Then run the training script:

```bash
python backend/train.py --data_dir path/to/asl_dataset --epochs 20
```

## 🔍 Model Architecture

Our model uses a CNN architecture with:
- 3 convolutional blocks with increasing filter sizes (32 → 64 → 128)
- Batch normalization for faster training
- Dropout layers to prevent overfitting
- Data augmentation during training
- Early stopping to prevent overfitting

## 🐛 Troubleshooting

- **API Connection Errors**: Ensure the Flask server is running on port 5000
- **Webcam Issues**: Check browser permissions and ensure no other application is using the webcam
- **Model Training Problems**: Verify your dataset structure and ensure TensorFlow is properly installed
- **Low Detection Accuracy**: Try adjusting the confidence threshold or improving lighting conditions

## 📚 Resources

- [ASL Fingerspelling Alphabet](https://www.nidcd.nih.gov/health/american-sign-language)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [React Documentation](https://reactjs.org/docs/getting-started.html)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 📄 License

MIT

## 👏 Acknowledgements

- TensorFlow/Keras for machine learning capabilities
- React and Vite for frontend framework
- Flask for backend API

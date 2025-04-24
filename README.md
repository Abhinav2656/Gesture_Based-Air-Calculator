# 🍎 MathCam – Apple Calculator Using Computer Vision

MathCam is a computer vision-based calculator that lets you draw digits in the air using hand gestures, processes the input using AI, and performs mathematical calculations—all in real time.

<p align="center">
  <img src="https://img.shields.io/github/license/ayush-that/Apple-Calculator-Using-Computer-Vision?style=for-the-badge">
  <img src="https://img.shields.io/github/stars/ayush-that/Apple-Calculator-Using-Computer-Vision?style=for-the-badge">
</p>

---

## 🚀 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
  <img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white">
  <img src="https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white">
  <img src="https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white">
  <img src="https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E">
</p>

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ayush-that/Apple-Calculator-Using-Computer-Vision.git
cd Apple-Calculator-Using-Computer-Vision
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key

To use AI features, you need a **Gemini API key**.

- Generate your key via [Google AI Studio](https://makersuite.google.com/app/apikey).
- Replace `YOUR_API_KEY` in `app.py` with your own key.

> ⚠️ *Note: This project hasn't been tested with a `python-dotenv` file.*

### 5. Run the Application

```bash
python app.py
```

---

## 🧠 How It Works

- Uses computer vision to detect hand gestures.
- Recognizes drawn digits and symbols via a CNN-based model.
- Evaluates the mathematical expression using backend logic.
- Displays the result in real-time.

---

<p align="center">
  ⭐ If you like this project, give it a star!
</p>

# 🛡 Fake News Detection System

> ⚡ Combating misinformation, one article at a time.

![Repo Size](https://img.shields.io/github/repo-size/Shyam4849/DIGIBHEM)
![Issues](https://img.shields.io/github/issues/Shyam4849/DIGIBHEM)
![Forks](https://img.shields.io/github/forks/Shyam4849/DIGIBHEM?style=social)
![Stars](https://img.shields.io/github/stars/Shyam4849/DIGIBHEM?style=social)

---

## 🧠 Overview

*DIGIBHEM* (Digital BHEEM) is an AI-powered platform built during an internship project to detect *Fake News* using a fine-tuned *BERT* model. It provides a secure, easy-to-use web interface with real-time predictions and confidence scores.

---

## 🎯 Core Features

- 🧠 *BERT Model* – Detect fake vs. real news with high accuracy  
- 📊 *Confidence Scores* – View prediction probabilities in real-time  
- 🧾 *Text Analyzer* – Paste or type news to evaluate credibility  
- 🔐 *Secure & Offline-Ready* – Built with model loading via Git LFS  
- 🧬 *Clean UI* – Built with glassmorphism, animated results, and background visuals  
- 🗃 *History & Download* – View past predictions and export as CSV  
- 🎥 *Lottie Animations* – Engaging fake/real/neutral feedback

---

## 🚀 Tech Stack

### 🖼 Frontend

- *Streamlit* – Interactive UI
- *HTML/CSS-injected* – Glassmorphism + animation styling
- *LottieFiles* – Animated visual feedback

### 🧠 Backend / ML

- *BERT* (HuggingFace Transformers)  
- *TensorFlow* – Prediction backend  
- *Git LFS* – Handles large .bin model files  
- *Pandas* – For data handling and CSV export

---

## 🗂 Folder Structure

bash
DIGIBHEM/
├── bert_fakenews_model/     # Trained BERT model (via Git LFS)
├── .gitattributes           # Git LFS config
├── app.py                   # Main Streamlit App
├── error.json               # Lottie: Fake News
├── fake.csv                 # Fake News
├── neutral.json             # Lottie: Unsure Result
├── news_background.jpg      # Page Background
├── preprocess_data.py       # Cleaning Data
├── processed_news.csv       # Processed News 
├── requirements.txt         # Python Dependencies
├── success.json             # Lottie: Real News
├── train_bert.py            # Training File
├── True.csv                 # True News
└── README.md                # This file

---
## 🔧 Setup Instructions

bash
# Step 1: Clone the Repository
git clone https://github.com/Shyam4849/DIGIBHEM.git
cd DIGIBHEM

# Step 2: (Optional) Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Step 3: Install Required Packages
pip install -r requirements.txt

# Step 4: Run the App
streamlit run app.py

---

## 🧪 Future Features (Coming Soon!)
## 🌐 Real-time News URL classifier

- 🗣 Voice-to-text fake news detection

- 📊 Weekly fake news trend reports

- 📱 Mobile-friendly deployment

- 📤 Alerts & email summaries for flagged content

- 🧑‍⚖ Admin dashboard for moderators
---
## 📷 UI Preview
| 🧠 Input Screen | ✅ Real News |	❌ Fake News |	📜 History |
|-----------------|---------------|--------------|--------------|
|  Input_Screen   |   Real News   |	  Fake News  |	  History   |
|-----------------|---------------|--------------|--------------|
| ![Input_SCreen](https://raw.githubusercontent.com/Shyam4849/DIGIBHEM/main/Images/Input_Screen.jpg) | ![Real_News](https://raw.githubusercontent.com/Shyam4849/DIGIBHEM/main/Images/Real_News.jpg) | ![Fake_News](https://raw.githubusercontent.com/Shyam4849/DIGIBHEM/main/Images/Fake_News.jpg) | ![History](https://raw.githubusercontent.com/Shyam4849/DIGIBHEM/main/Images/History.jpg) |

---

## 🤝 Contributing
# We welcome contributors, interns, and AI enthusiasts to enhance this project.

- 1. 🍴 *Fork* the repository
- 2. 🌿 *Create* a feature branch
- 3. 💾 *Commit* your changes
- 4. 🚀 *Push* the branch to GitHub
- 5. ✅ *Open* a Pull Request

- We welcome *coders, **designers, and **wellness advocates* alike! 🙌
- Let's build something meaningful together 💙

---

## 🙏 Acknowledgements

- 🤖 [HuggingFace Transformers](https://huggingface.co/)
- 🎨 [LottieFiles](https://lottiefiles.com/)
- 🧠 [TensorFlow](https://www.tensorflow.org/)
- 🧩 Open Source Community
  
---

## 👨‍💻 Developed By  
*Shyam Kumar Soni*  
B.Tech CSE | Internship Project: Digital BHEM Fake News Detector  
🔗 [GitHub](https://github.com/Shyam4849) | 💼 [LinkedIn](https://www.linkedin.com/in/shyam-kumar-soni-4017ba28b/)

---

Built with ❤ by *[SHYAM KUMAR SONI]*

# Emotion-Aware Speech Transcription

Multimodal NLP pipeline that transcribes speech and detects emotion.

## Setup & Run

### 1. Clone the repo
git clone (https://github.com/Ayush-Saini-AI/multimodal-ser-asr)
cd NLP_project

### 2. Create virtual environment
python -m venv venv

### 3. Activate venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

### 4. Install dependencies
pip install -r requirements.txt

### 5. Install ffmpeg (Windows only)
Download from: https://github.com/BtbN/FFmpeg-Builds/releases
Get: ffmpeg-master-latest-win64-gpl.zip
Extract and add the bin folder to System PATH.

### 6. Run the app
streamlit run app.py

## Files
- app.py → Streamlit frontend
- requirements.txt → All dependencies
```

---

**Step 3 — Initialize Git in VS Code terminal**
```
git init
git add .
git commit -m "Initial commit - Review 1 pipeline"
```

---

**Step 4 — Push to GitHub**

1. Go to **github.com** → click **"New repository"**
2. Name it `NLP_project`
3. Keep it **Public** (so teammates can access)
4. **Don't** tick "Add README" — you already have one
5. Click **Create repository**
6. GitHub will show you commands — run these in your terminal:
```
git remote add origin https://github.com/yourusername/NLP_project.git
git branch -M main
git push -u origin main
```

Replace `yourusername` with your actual GitHub username.

---

**Step 5 — What your teammates do**

They just run these 6 commands on their machine:
```
git clone https://github.com/yourusername/NLP_project.git
cd NLP_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Plus they need to install ffmpeg separately if they're on Windows (same steps you did).

---

**Your project folder should look like this before pushing:**
```
NLP_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/          ← this will be ignored by git ✅

# 🧠 FetalBrain AI - Fetal Brain Abnormality Detection System

An AI-powered deep learning diagnostic web platform utilizing **Swin Transformer (Swin-Tiny)** for automated detection and multi-class classification of fetal brain abnormalities from ultrasound scans with clinical PDF reporting.

---

## ⚡ 1-Click Easy Start (For Friends & Reviewers)

After unzipping this project folder, follow these simple steps to run the application locally:

### 🪟 On Windows (Easiest):
1. **Double-click `run_app.bat`** in the project folder.
2. The script will automatically:
   - Check your Python installation.
   - Create a local virtual environment (`venv`).
   - Install all required dependencies (`PyTorch`, `timm`, `Flask`, `ReportLab`, etc.).
   - Launch the server and automatically open your default browser to **`http://127.0.0.1:5000`**.

---

### 🍎 On macOS / Linux:
1. Open Terminal in the project folder.
2. Run:
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```
3. The script will configure dependencies and launch the portal automatically at **`http://127.0.0.1:5000`**.

---

## 💻 Manual Step-by-Step Run (Terminal / Command Prompt)

If you prefer to run the commands manually:

### 1. Prerequisites
Make sure **Python 3.9, 3.10, or 3.11** is installed.
- Download Python: [python.org/downloads](https://www.python.org/downloads/)
- ⚠️ *Important*: Check the box **"Add Python to PATH"** during installation.

### 2. Open Terminal in the Project Directory
```bash
# Windows Command Prompt / PowerShell
python -m venv venv
venv\Scripts\activate

# macOS / Linux Terminal
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Web Server
```bash
python webapp/app.py
```

### 5. Open in Web Browser
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## 🛡️ System Administrator Portal & Credentials

The system includes a Master Admin Control Center for managing clinician approvals, monitoring scan activity, and running diagnostic scans.

| Role | Access URL | Login ID / Email | Password | Status |
| :--- | :--- | :--- | :--- | :--- |
| **System Administrator** | `http://127.0.0.1:5000/login` *(Admin Portal tab)* | `admin` *(or `admin@fetalbrain.org`)* | `admin123` | Master Access (Pre-configured) |
| **Doctor / Clinician** | `http://127.0.0.1:5000/login` *(Doctor Login tab)* | Registered Email | Chosen Password | Requires Admin Verification |

---

## 🩺 Clinician Registration & Approval Workflow

```text
[Doctor Registers at /register] 
          │
          ▼
[Account Status: "Pending Verification"]
          │
          ├──► Real-Time Notification Email sent to Admin (fetalbraingroup@gmail.com)
          │
          ▼
[Admin opens /admin/requests]
          │
   ┌──────┴──────┐
   ▼             ▼
[Approve ✓]   [Reject ✗]
   │             │
   ▼             ▼
Doctor can    Access
log in to     blocked
portal
```

1. **Doctor Registration**:
   - Clinicians sign up via the **Register** page (`/register`) with their Name, Email, Phone, and Hospital / Organization.
   - Upon submission, their account is placed in **Pending Verification** status.
2. **Admin Real-Time Notification**:
   - An automated alert email is instantly dispatched to the Administrator (`fetalbraingroup@gmail.com`) detailing the doctor's name, email, and hospital affiliation.
3. **Admin Approval**:
   - The Administrator logs in to the **Admin Portal** -> **Doctor Requests** (`/admin/requests`) and clicks **✓ Approve** or **✗ Reject**.
4. **Login & Diagnosis**:
   - Approved clinicians can sign in, upload fetal brain ultrasound images, view instant AI predictions, and download clinical PDF reports.

---

## 📁 Project Directory Overview

```text
fetal_brain_abnormalities_detection/
│
├── fetal_brain_swin_best.pth   # Swin Transformer PyTorch Model Checkpoint (98.7% Accuracy)
├── requirements.txt            # Python Dependencies
├── README.md                   # Full Documentation & Setup Guide
├── run_app.bat                 # 1-Click Startup Script for Windows
├── run_app.sh                  # 1-Click Startup Script for macOS/Linux
├── datasets/                   # Sample Ultrasound Scan Images
├── test.py                     # Standalone Desktop GUI (Tkinter)
├── train.ipynb                 # Swin Transformer Training Notebook
│
└── webapp/                     # Flask Web Application Directory
    ├── app.py                  # Main Server, Inference Pipeline & Admin Handlers
    ├── instance/               # SQLite Database (fetal_brain.db)
    ├── static/
    │   ├── css/style.css       # Dynamic Dark/Light Responsive UI Styling
    │   ├── js/main.js          # Ripple Effects & Interactions
    │   └── uploads/            # Saved Ultrasound Scans & Previews
    └── templates/              # Jinja2 HTML5 Templates
        ├── index.html          # Landing Page
        ├── about.html          # System & Model Architecture Info
        ├── contact.html        # Clinical Support Desk
        ├── login.html          # Role-based Tabbed Sign-In
        ├── register.html       # Doctor Registration
        ├── predict.html        # AI Diagnostic Scanning Interface
        ├── profile.html        # Doctor History & Logout Page
        ├── admin.html          # Admin Dashboard & Statistics
        ├── admin_requests.html # Doctor Verification Desk
        └── admin_doctor_detail.html # Doctor In-Depth Audit View
```

---

## 🖥️ Alternative: Standalone Desktop GUI (Tkinter)

To test the model in a local desktop window without the web browser:

```bash
python test.py
```
1. Click **"Select Ultrasound Image"**.
2. Select any scan from the `datasets/` folder.
3. View the classification result and confidence score.

---

## 🛠️ Troubleshooting & FAQ

#### 1. `Python is not recognized as an internal or external command`
- Reinstall Python from [python.org](https://www.python.org/) and make sure to check the box **"Add Python to PATH"**.

#### 2. `Address already in use` (Port 5000 busy)
- Open `webapp/app.py`, scroll to the bottom line, and change `port=5000` to `port=5050`.
- Then open `http://127.0.0.1:5050` in your browser.

#### 3. How to perform a fresh database reset?
- Log in as Administrator -> Click **"⚠️ Hard Reset System"** in the top header to wipe all test data cleanly.

---

## 📄 License & Medical Notice

Developed for Fetal Brain Abnormality Detection & Research. All AI findings serve as clinical decision support.

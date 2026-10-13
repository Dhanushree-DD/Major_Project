import os
import sqlite3
import uuid
import smtplib
import json
import urllib.request
import mimetypes
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta

import torch
import timm
from torchvision import transforms
from PIL import Image as PILImage
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, g, Response
)
import csv
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    HRFlowable, KeepTogether, Image as RLImage
)
from reportlab.pdfgen import canvas
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import numpy as np

from ai.segmentation import load_segmentation_model, predict_mask, make_segmentation_overlay, crop_to_mask
from ai.gradcam import generate_gradcam, overlay_cam

# ==========================================
# CONFIG
# ==========================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fetalbrain-ai-secret-key-change-in-production")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(BASE_DIR, "instance", "fetal_brain.db")
MODEL_PATH = os.path.join(PARENT_DIR, "fetal_brain_swin_best.pth")
SEGMENTATION_MODEL_PATH = os.environ.get("SEGMENTATION_MODEL_PATH", os.path.join(PARENT_DIR, "webapp", "models", "segmentation_unet.pth"))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
SEGMENTATION_DIR = os.path.join(BASE_DIR, "static", "segmentations")
EXPLANATION_DIR = os.path.join(BASE_DIR, "static", "explanations")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_DIR, exist_ok=True)
os.makedirs(EXPLANATION_DIR, exist_ok=True)

# ==========================================
# DATABASE
# ==========================================

# Admin Credentials
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db


def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        db = get_db()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                organisation TEXT,
                last_login TIMESTAMP,
                role TEXT DEFAULT 'doctor',
                status TEXT DEFAULT 'pending',
                approved_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                result_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                patient_name TEXT,
                patient_email TEXT,
                patient_phone TEXT,
                patient_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS support_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT,
                category TEXT,
                subject TEXT,
                message TEXT NOT NULL,
                attachment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS login_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        try:
            db.execute("ALTER TABLE users ADD COLUMN name TEXT")
            db.commit()
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("ALTER TABLE users ADD COLUMN organisation TEXT")
            db.commit()
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("ALTER TABLE users ADD COLUMN last_login TIMESTAMP")
            db.commit()
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'doctor'")
            db.commit()
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'pending'")
            db.commit()
            # Set legacy existing records to approved
            db.execute("UPDATE users SET status = 'approved' WHERE status IS NULL OR status = ''")
            db.commit()
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("ALTER TABLE users ADD COLUMN approved_at TIMESTAMP")
            db.commit()
        except sqlite3.OperationalError:
            pass
        for column in ["patient_name", "patient_email", "patient_phone", "patient_message"]:
            try:
                db.execute(f"ALTER TABLE predictions ADD COLUMN {column} TEXT")
                db.commit()
            except sqlite3.OperationalError:
                pass
        db.commit()


app.teardown_appcontext(close_db)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")
try:
    # The repository uses Git LFS for the 110 MB checkpoint. If only the
    # pointer file is present, fail gracefully instead of crashing Flask.
    with open(MODEL_PATH, "rb") as _f:
        _prefix = _f.read(80)
    if _prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError("Swin checkpoint is a Git-LFS pointer. Run `git lfs pull` or download the real .pth file.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=len(classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded. {len(classes)} classes. Acc: {checkpoint.get('accuracy', 'N/A')}")
except Exception as e:
    print("MODEL LOAD ERROR:", e)
    model = None
    classes = []

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Optional trained anatomical segmentation model. If no checkpoint exists,
# the UI uses a clearly-labelled demo ROI proposal so the integration can be
# demonstrated before pixel-level masks are collected and training is done.
try:
    segmentation_model = load_segmentation_model(SEGMENTATION_MODEL_PATH, "cpu")
    print("Segmentation model loaded:", SEGMENTATION_MODEL_PATH)
except Exception as e:
    print("SEGMENTATION MODEL LOAD ERROR:", e)
    segmentation_model = None

# ==========================================
# HELPERS & AUTH DECORATORS
# ==========================================

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            flash("Admin access required. Please sign in as administrator.", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def allowed_file(filename):
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS


def predict_image(filepath, classification_image=None):
    """Run Swin classification. If classification_image is supplied, it is the
    segmentation-guided ROI; otherwise the original scan is used."""
    image = classification_image or PILImage.open(filepath).convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        predicted_class = classes[pred.item()]
    return predicted_class.replace("_", " ").title(), round(confidence.item() * 100, 2), pred.item(), x


def build_explainability_outputs(filepath, stem, classification_image=None, pred_idx=None, input_tensor=None):
    """Create segmentation + Grad-CAM visuals for one uploaded scan."""
    image = PILImage.open(filepath).convert("RGB")
    mask, seg_mode = predict_mask(image, segmentation_model, "cpu")
    seg_overlay = make_segmentation_overlay(image, mask)
    seg_name = f"{stem}_segmentation.jpg"
    seg_path = os.path.join(SEGMENTATION_DIR, seg_name)
    seg_overlay.save(seg_path, quality=92)

    cam_name = f"{stem}_gradcam.jpg"
    cam_path = os.path.join(EXPLANATION_DIR, cam_name)
    cam_available = False
    if model is not None and input_tensor is not None and pred_idx is not None:
        try:
            cam, _ = generate_gradcam(model, input_tensor, pred_idx)
            overlay_cam(classification_image or image, cam, mask=mask).save(cam_path, quality=92)
            cam_available = True
        except Exception as exc:
            print("Grad-CAM generation error:", exc)

    return {
        "segmentation": seg_name,
        "gradcam": cam_name if cam_available else None,
        "segmentation_mode": seg_mode,
        "segmentation_trained": seg_mode == "trained_unet",
        "gradcam_available": cam_available,
    }


# Web3Forms API Configuration
WEB3FORMS_KEY = os.environ.get("WEB3FORMS_ACCESS_KEY", "a5c0c025-e91e-4122-b10a-9f4a14a1af3f")

def send_web3forms_ticket(name, email, phone, category, subject, message):
    access_key = os.environ.get("WEB3FORMS_ACCESS_KEY", WEB3FORMS_KEY)
    if not access_key:
        print("[Web3Forms Note] Set WEB3FORMS_ACCESS_KEY to activate Web3Forms email delivery.")
        return False

    try:
        url = "https://api.web3forms.com/submit"
        full_message = f"Doctor/User Name: {name}\nDoctor Email: {email}\nPhone Number: {phone or 'Not Provided'}\nCategory: {category or 'General Support'}\n\nsubject: {subject or 'No Subject'}\n\nIssue Details:\n{message}"

        payload = {
            "access_key": access_key,
            "name": name,
            "email": email,
            "category": category or "General Support",
            "subject": f"[FetalBrain Support - {category}] {subject or 'New Support Ticket'}",
            "message": full_message,
            "from_name": "FetalBrain AI Support Desk"
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "FetalBrainAI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            res = json.loads(response.read().decode("utf-8"))
            print("[Web3Forms API Response]:", res)
            if res.get("success"):
                print("[Web3Forms] SUCCESS! Email Delivered to fetalbraingroup@gmail.com!")
                return True
            else:
                print("[Web3Forms API Error Message]:", res.get("message"))
    except Exception as e:
        print("[Web3Forms Exception]:", str(e))
    return False

def send_support_email(name, email, phone, category, subject, message):
    receiver_email = os.environ.get("RECIPIENT_EMAIL", "fetalbraingroup@gmail.com")
    smtp_server = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("MAIL_PORT", 587))
    smtp_user = os.environ.get("MAIL_USERNAME")
    smtp_password = os.environ.get("MAIL_PASSWORD")

    email_body = f"""
==================================================
NEW CLINICAL / TECHNICAL SUPPORT TICKET SUBMITTED
==================================================
Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Doctor/User Name: {name}
Doctor Email: {email}
Phone Number: {phone or 'Not Provided'}
Category: {category or 'General Support'}
Subject: {subject or 'No Subject'}

MESSAGE / ISSUE DESCRIPTION:
--------------------------------------------------
{message}
"""

    print("--------------------------------------------------")
    print("SUPPORT TICKET DISPATCH LOG:")
    print(email_body)
    print("--------------------------------------------------")

    web3_sent = send_web3forms_ticket(name, email, phone, category, subject, message)

    if not web3_sent and smtp_user and smtp_password:
        try:
            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = receiver_email
            msg["Subject"] = f"[FetalBrain Support - {category}] {subject or 'New Support Ticket'}"
            msg.attach(MIMEText(email_body, "plain"))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, receiver_email, msg.as_string())
            server.quit()
            print("Support Ticket Email sent successfully via SMTP!")
        except Exception as e:
            print("SMTP Send Error:", str(e))


def send_admin_new_doctor_alert(name, email, phone, organisation):
    """Sends an automated email notification to the Admin (fetalbraingroup@gmail.com)
    alerting that a new doctor has registered and needs review/approval."""
    receiver_email = os.environ.get("RECIPIENT_EMAIL", "fetalbraingroup@gmail.com")
    subject = f"[FetalBrain Admin Alert] New Doctor Registration Request: Dr. {name}"
    
    email_body = f"""==================================================
NEW DOCTOR REGISTRATION REQUEST AWAITING REVIEW
==================================================
Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CLINICIAN DETAILS:
--------------------------------------------------
Doctor Name: Dr. {name}
Doctor Email: {email}
Phone Number: {phone or 'Not Provided'}
Hospital / Organisation: {organisation or 'Clinical Diagnostic Centre'}
Account Status: PENDING VERIFICATION

ADMIN ACTION REQUIRED:
--------------------------------------------------
A new doctor has registered and requested system access.
Please review their details and approve or reject the request:
Admin Review Link: http://127.0.0.1:5000/admin/requests

=================================================="""

    print("--------------------------------------------------")
    print("ADMIN NOTIFICATION DISPATCH LOG (NEW DOCTOR REGISTRATION):")
    print(email_body)
    print("--------------------------------------------------")

    # Send via Web3Forms API to Admin inbox (fetalbraingroup@gmail.com)
    access_key = os.environ.get("WEB3FORMS_ACCESS_KEY", WEB3FORMS_KEY)
    if access_key:
        try:
            url = "https://api.web3forms.com/submit"
            payload = {
                "access_key": access_key,
                "name": f"New Doctor Request - Dr. {name}",
                "email": email,
                "subject": subject,
                "message": email_body,
                "from_name": "FetalBrain Registration Alert"
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "FetalBrainAI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res = json.loads(response.read().decode("utf-8"))
                print("[Web3Forms Admin Alert Response]:", res)
                if res.get("success"):
                    print("[Web3Forms] SUCCESS! New doctor registration alert delivered to fetalbraingroup@gmail.com!")
        except Exception as e:
            print("[Web3Forms Admin Alert Exception]:", str(e))

    # Send via SMTP if configured
    smtp_user = os.environ.get("MAIL_USERNAME")
    smtp_password = os.environ.get("MAIL_PASSWORD")
    smtp_server = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("MAIL_PORT", 587))
    if smtp_user and smtp_password:
        try:
            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = receiver_email
            msg["Subject"] = subject
            msg.attach(MIMEText(email_body, "plain"))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, receiver_email, msg.as_string())
            server.quit()
            print(f"[SMTP] Registration alert email sent to {receiver_email} successfully!")
        except Exception as e:
            print(f"[SMTP Alert Error]: {e}")


def send_doctor_status_email(name, email, status, organisation=None):
    """Sends an automated email directly to the doctor via SMTP when approved or rejected."""
    if status == "approved":
        subject = "[FetalBrain AI] Your Clinician Account has been APPROVED"
        body = f"""Dear Dr. {name},

Congratulations! Your clinical registration for FetalBrain AI Fetal Neurosonogram Diagnostic System has been APPROVED by the Administrator.

Account Details:
- Doctor Name: Dr. {name}
- Registered Email: {email}
- Affiliation: {organisation or 'Clinical Unit'}
- Account Status: Active / Approved

You can now log in to the portal and start analyzing fetal brain ultrasound scans:
Login Portal: http://127.0.0.1:5000/login

Best regards,
FetalBrain AI Diagnostic & Research Team
AIT Institute of Technology & Health Sciences
Support Email: fetalbraingroup@gmail.com
"""
    else:
        subject = "[FetalBrain AI] Update regarding your Clinician Registration Request"
        body = f"""Dear Dr. {name},

Thank you for your interest in the FetalBrain AI Diagnostic System.

We regret to inform you that your clinical registration request for {email} could not be approved by the Administrator at this time.

If you believe this decision was made in error or would like to submit updated clinical affiliation credentials, please contact our administrative desk at fetalbraingroup@gmail.com.

Best regards,
FetalBrain AI Administration Team
AIT Institute of Technology & Health Sciences
"""

    print("==================================================")
    print(f"AUTOMATED CLINICIAN NOTIFICATION LOG ({status.upper()}):")
    print(f"Recipient: Dr. {name} <{email}>")
    print(f"Subject: {subject}")
    print("--------------------------------------------------")
    print(body)
    print("==================================================")

    # Attempt direct SMTP delivery to doctor if configured
    smtp_user = os.environ.get("MAIL_USERNAME")
    smtp_password = os.environ.get("MAIL_PASSWORD")
    smtp_server = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("MAIL_PORT", 587))
    if smtp_user and smtp_password:
        try:
            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, email, msg.as_string())
            server.quit()
            print(f"[SMTP] Status notification sent directly to Dr. {name} at {email} successfully!")
        except Exception as e:
            print(f"[SMTP Status Email Error]: {e}")


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        category = request.form.get("category", "General Support").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()

        if not name or not email or not message:
            flash("Please fill in your name, email address, and issue description.", "danger")
            return redirect(url_for("contact"))

        db = get_db()
        db.execute(
            """INSERT INTO support_tickets
               (name, email, phone, category, subject, message)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, email, phone, category, subject, message)
        )
        db.commit()

        send_support_email(name, email, phone, category, subject, message)

        contact_info = f"at {email}" + (f" or {phone}" if phone else "")
        flash(
            f"Thank you, Dr. {name}! Your support ticket has been submitted to our engineering team. "
            f"We will review your request and contact you {contact_info} shortly.",
            "success"
        )
        return redirect(url_for("contact"))

    return render_template("contact.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        organisation = request.form.get("organisation", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not name or not email or not phone or not password:
            flash("All required fields must be completed.", "danger")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if len(password) < 4:
            flash("Password must be at least 4 characters.", "danger")
            return render_template("register.html")

        db = get_db()
        if db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone():
            flash("Email already registered. Please sign in instead.", "danger")
            return render_template("register.html")

        pw_hash = generate_password_hash(password)
        org_name = organisation or "Hospital / Diagnostic Clinic"
        db.execute(
            """INSERT INTO users (name, email, phone, organisation, password_hash, status)
               VALUES (?, ?, ?, ?, ?, 'pending')""",
            (name, email, phone, org_name, pw_hash)
        )
        db.commit()

        # Send alert email to Administrator (fetalbraingroup@gmail.com)
        send_admin_new_doctor_alert(name, email, phone, org_name)

        flash(
            f"Registration submitted, Dr. {name}! Your account is currently PENDING VERIFICATION by the Administrator. "
            "You will be able to log in as soon as the administrator approves your request.",
            "warning"
        )
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # Check for Fixed Admin Credentials
        admin_aliases = {
            ADMIN_USERNAME.lower(),
            f"{ADMIN_USERNAME.lower()}@fetalbrain.org",
            "admin",
            "admin@fetalbrain.org"
        }
        if identifier in admin_aliases and password == ADMIN_PASSWORD:
            session.clear()
            session.permanent = True
            session["user_id"] = "admin"
            session["is_admin"] = True
            session["name"] = "System Administrator"
            session["email"] = f"{ADMIN_USERNAME}@fetalbrain.org"
            session["role"] = "admin"
            flash("Welcome to Admin Control Center, Administrator.", "success")
            return redirect(url_for("admin_dashboard"))

        # Check Doctor/User Database Credentials
        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE email = ?", (identifier,)
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            # Check Account Approval Status
            status = user["status"] if "status" in user.keys() and user["status"] else "approved"
            if status == "pending":
                flash(
                    "Your doctor account is currently PENDING VERIFICATION by the Administrator. "
                    "Access is restricted until your registration is approved.",
                    "warning"
                )
                return render_template("login.html")
            elif status == "rejected":
                flash(
                    "Your registration request has been REJECTED by the Administrator. Access denied.",
                    "danger"
                )
                return render_template("login.html")

            session.clear()
            session.permanent = True
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["email"] = user["email"]
            session["role"] = "doctor"
            session["is_admin"] = False

            # Track doctor login
            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
            user_agent = request.headers.get("User-Agent", "Unknown Device")[:120]
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            db.execute(
                "INSERT INTO login_logs (user_id, ip_address, user_agent, login_time) VALUES (?, ?, ?, ?)",
                (user["id"], client_ip, user_agent, now_ts)
            )
            db.execute("UPDATE users SET last_login = ? WHERE id = ?", (now_ts, user["id"]))
            db.commit()

            flash(f"Welcome back, Dr. {user['name']}!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid email/login ID or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been safely logged out.", "info")
    return redirect(url_for("index"))


@app.route("/profile")
@login_required
def profile():
    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    if not user:
        session.clear()
        flash("Session expired or doctor record not found. Please log in again.", "warning")
        return redirect(url_for("login"))
    predictions = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
        (session["user_id"],)
    ).fetchall()
    return render_template("profile.html", user=user, predictions=predictions)


# ==========================================
# ADMIN PORTAL & DOCTOR TRACKING
# ==========================================

@app.route("/admin")
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    db = get_db()
    total_doctors = db.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
    pending_requests_count = db.execute("SELECT COUNT(*) as c FROM users WHERE status = 'pending'").fetchone()["c"]
    approved_doctors_count = db.execute("SELECT COUNT(*) as c FROM users WHERE status = 'approved'").fetchone()["c"]
    total_scans = db.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
    total_anomalies = db.execute("SELECT COUNT(*) as c FROM predictions WHERE LOWER(result_class) != 'normal'").fetchone()["c"]
    total_normal = db.execute("SELECT COUNT(*) as c FROM predictions WHERE LOWER(result_class) == 'normal'").fetchone()["c"]
    total_tickets = db.execute("SELECT COUNT(*) as c FROM support_tickets").fetchone()["c"]

    # Today's scan metrics
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_scans_total = db.execute("SELECT COUNT(*) as c FROM predictions WHERE created_at LIKE ?", (f"{today_str}%",)).fetchone()["c"]
    today_admin_scans = db.execute("SELECT COUNT(*) as c FROM predictions WHERE (user_id = 0 OR user_id = 'admin') AND created_at LIKE ?", (f"{today_str}%",)).fetchone()["c"]
    today_doctor_scans = db.execute("SELECT COUNT(*) as c FROM predictions WHERE (user_id != 0 AND user_id != 'admin') AND created_at LIKE ?", (f"{today_str}%",)).fetchone()["c"]

    # All doctors with scan count, verification status and last scan timestamp
    doctors = db.execute("""
        SELECT u.id, u.name, u.email, u.phone, u.organisation, u.created_at, u.last_login, u.status, u.approved_at,
               COUNT(p.id) as scan_count,
               MAX(p.created_at) as last_scan_date
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()

    # Admin Scanned Patients (Scans uploaded directly by Admin)
    admin_scans = db.execute("""
        SELECT * FROM predictions
        WHERE user_id = 0 OR user_id = 'admin'
        ORDER BY created_at DESC
    """).fetchall()

    # Recent doctor logins audit trail
    recent_logins = db.execute("""
        SELECT l.id, l.user_id, l.ip_address, l.user_agent, l.login_time,
               u.name as doctor_name, u.email as doctor_email, u.organisation
        FROM login_logs l
        JOIN users u ON l.user_id = u.id
        ORDER BY l.login_time DESC
        LIMIT 25
    """).fetchall()

    # Recent scans across all users
    recent_scans = db.execute("""
        SELECT p.*,
               CASE WHEN p.user_id = 0 OR p.user_id = 'admin' THEN 'Administrator' ELSE u.name END as doctor_name,
               CASE WHEN p.user_id = 0 OR p.user_id = 'admin' THEN 'admin@fetalbrain.org' ELSE u.email END as doctor_email
        FROM predictions p
        LEFT JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
        LIMIT 15
    """).fetchall()

    return render_template(
        "admin.html",
        total_doctors=total_doctors,
        pending_requests_count=pending_requests_count,
        approved_doctors_count=approved_doctors_count,
        total_scans=total_scans,
        total_anomalies=total_anomalies,
        total_normal=total_normal,
        total_tickets=total_tickets,
        today_scans_total=today_scans_total,
        today_admin_scans=today_admin_scans,
        today_doctor_scans=today_doctor_scans,
        doctors=doctors,
        admin_scans=admin_scans,
        recent_logins=recent_logins,
        recent_scans=recent_scans
    )


@app.route("/admin/requests")
@admin_required
def admin_requests():
    db = get_db()
    pending_requests = db.execute("SELECT * FROM users WHERE status = 'pending' ORDER BY created_at DESC").fetchall()
    all_requests = db.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    pending_count = len(pending_requests)

    return render_template(
        "admin_requests.html",
        pending_requests=pending_requests,
        all_requests=all_requests,
        pending_count=pending_count
    )


@app.route("/admin/request/<int:user_id>/approve", methods=["POST"])
@admin_required
def admin_approve_doctor(user_id):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        flash("Doctor request not found.", "danger")
        return redirect(url_for("admin_requests"))

    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute("UPDATE users SET status = 'approved', approved_at = ? WHERE id = ?", (now_ts, user_id))
    db.commit()

    # Dispatch automated email notification
    send_doctor_status_email(user["name"], user["email"], "approved", user["organisation"] if "organisation" in user.keys() else None)

    flash(f"Doctor account for Dr. {user['name']} ({user['email']}) has been APPROVED! An automated notification email was dispatched.", "success")
    return redirect(request.referrer or url_for("admin_requests"))


@app.route("/admin/request/<int:user_id>/reject", methods=["POST"])
@admin_required
def admin_reject_doctor(user_id):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        flash("Doctor request not found.", "danger")
        return redirect(url_for("admin_requests"))

    db.execute("UPDATE users SET status = 'rejected' WHERE id = ?", (user_id,))
    db.commit()

    # Dispatch automated email notification
    send_doctor_status_email(user["name"], user["email"], "rejected", user["organisation"] if "organisation" in user.keys() else None)

    flash(f"Doctor request for Dr. {user['name']} ({user['email']}) has been REJECTED. Notification dispatched.", "warning")
    return redirect(request.referrer or url_for("admin_requests"))


@app.route("/admin/system/hard-reset", methods=["POST"])
@admin_required
def admin_hard_reset():
    db = get_db()
    try:
        # Delete all uploaded scan files
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as err:
                        print(f"Error removing file {file_path}: {err}")

        # Wipe database tables
        db.execute("DELETE FROM predictions")
        db.execute("DELETE FROM login_logs")
        db.execute("DELETE FROM support_tickets")
        db.execute("DELETE FROM users")
        try:
            db.execute("DELETE FROM sqlite_sequence WHERE name IN ('predictions', 'login_logs', 'support_tickets', 'users')")
        except sqlite3.OperationalError:
            pass
        db.commit()

        flash("COMPLETE SYSTEM HARD RESET PERFORMED! All doctors, scan records, login logs, and uploaded ultrasound files have been permanently erased.", "warning")
    except Exception as e:
        flash(f"Error executing hard reset: {str(e)}", "danger")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/doctor/<int:doctor_id>")
@admin_required
def admin_doctor_detail(doctor_id):
    db = get_db()
    doctor = db.execute("SELECT * FROM users WHERE id = ?", (doctor_id,)).fetchone()
    if not doctor:
        flash("Doctor record not found.", "danger")
        return redirect(url_for("admin_dashboard"))

    predictions = db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
        (doctor_id,)
    ).fetchall()

    login_history = db.execute(
        "SELECT * FROM login_logs WHERE user_id = ? ORDER BY login_time DESC LIMIT 100",
        (doctor_id,)
    ).fetchall()

    anomaly_count = sum(1 for p in predictions if p["result_class"].lower() != "normal")

    return render_template(
        "admin_doctor_detail.html",
        doctor=doctor,
        predictions=predictions,
        login_history=login_history,
        anomaly_count=anomaly_count
    )


@app.route("/admin/doctor/<int:doctor_id>/delete", methods=["POST"])
@admin_required
def admin_delete_doctor(doctor_id):
    db = get_db()
    doctor = db.execute("SELECT * FROM users WHERE id = ?", (doctor_id,)).fetchone()
    if not doctor:
        flash("Doctor record not found.", "danger")
        return redirect(url_for("admin_dashboard"))

    # Remove scan image files uploaded by this doctor
    preds = db.execute("SELECT filename FROM predictions WHERE user_id = ?", (doctor_id,)).fetchall()
    for p in preds:
        if p["filename"]:
            fpath = os.path.join(UPLOAD_DIR, p["filename"])
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    print(f"Error removing scan file {fpath}: {e}")

    # Remove from database
    db.execute("DELETE FROM predictions WHERE user_id = ?", (doctor_id,))
    db.execute("DELETE FROM login_logs WHERE user_id = ?", (doctor_id,))
    db.execute("DELETE FROM users WHERE id = ?", (doctor_id,))
    db.commit()

    flash(f"Doctor account 'Dr. {doctor['name']}' ({doctor['email']}) and all associated scan data were successfully deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/prediction/<int:prediction_id>/delete", methods=["POST"])
@admin_required
def admin_delete_prediction(prediction_id):
    db = get_db()
    pred = db.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    if not pred:
        flash("Scan record not found.", "danger")
        return redirect(url_for("admin_dashboard"))

    if pred["filename"]:
        fpath = os.path.join(UPLOAD_DIR, pred["filename"])
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass

    db.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    db.commit()
    flash("Scan record removed from system database.", "info")
    return redirect(request.referrer or url_for("admin_dashboard"))



@app.route("/admin/export/doctors/csv")
@admin_required
def admin_export_doctors():
    db = get_db()
    doctors = db.execute("""
        SELECT u.id, u.name, u.email, u.phone, u.organisation, u.created_at, u.last_login,
               COUNT(p.id) as scan_count
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Doctor Name", "Email", "Phone", "Organisation / Hospital", "Registered Date", "Last Login", "Total Scans"])
    for d in doctors:
        writer.writerow([
            d["id"],
            d["name"],
            d["email"],
            d["phone"],
            d["organisation"] or "N/A",
            d["created_at"],
            d["last_login"] or "Never",
            d["scan_count"]
        ])

    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=doctors_directory_report.csv"}
    )



def get_user_predictions():
    db = get_db()
    return db.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
        (session["user_id"],)
    ).fetchall()


def get_prediction_owned(prediction_id):
    db = get_db()
    if session.get("is_admin"):
        return db.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    return db.execute(
        "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, session.get("user_id"))
    ).fetchone()


def build_csv_response(predictions):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["#", "Patient Name", "Email", "Phone", "Message", "Result", "Confidence (%)", "Date"])
    for i, p in enumerate(predictions, 1):
        writer.writerow([
            i,
            p["patient_name"] or "",
            p["patient_email"] or "",
            p["patient_phone"] or "",
            p["patient_message"] or "",
            p["result_class"],
            p["confidence"],
            p["created_at"]
        ])

    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=fetal_brain_report.csv"}
    )


# ==========================================
# PROFESSIONAL PDF REPORT CANVAS
# ==========================================

class ProfessionalReportCanvas(canvas.Canvas):
    """Custom canvas that draws decorative page borders, teal header accent,
    and an institution footer with page numbering on every page."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self._draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def _draw_page_decorations(self, total_pages):
        self.saveState()
        page_width, page_height = A4

        # Outer teal border
        margin = 10 * mm
        self.setStrokeColor(colors.HexColor("#0f766e"))
        self.setLineWidth(1.5)
        self.rect(margin, margin, page_width - 2 * margin, page_height - 2 * margin)

        # Inner grey border
        inner_margin = 12 * mm
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.rect(inner_margin, inner_margin, page_width - 2 * inner_margin, page_height - 2 * inner_margin)

        # Top teal accent bar (drawn cleanly inside the inner border at y = 283mm .. 285mm)
        self.setFillColor(colors.HexColor("#0f766e"))
        self.rect(inner_margin, page_height - 14 * mm,
                  page_width - 2 * inner_margin, 2 * mm, fill=True, stroke=False)

        # Footer separator line
        footer_y = 15 * mm
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(inner_margin, footer_y, page_width - inner_margin, footer_y)

        # Footer institution text
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(
            16 * mm, 11 * mm,
            "AIT Institute of Technology • Chikkamagaluru, Karnataka • fetalbraingroup@gmail.com"
        )
        # Footer page number
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(page_width - 16 * mm, 11 * mm, page_str)
        self.restoreState()


# ==========================================
# PROFESSIONAL PDF RESPONSE BUILDER
# ==========================================

def _build_single_prediction_story(p, styles_map):
    """Build the reportlab story elements for a single prediction record."""
    hospital_header_style = styles_map["hospital_header"]
    dept_style = styles_map["dept"]
    sub_address_style = styles_map["sub_address"]
    doc_title_style = styles_map["doc_title"]
    meta_style = styles_map["meta"]
    section_heading = styles_map["section_heading"]
    label_style = styles_map["label"]
    value_style = styles_map["value"]
    body_style = styles_map["body"]
    disclaimer_style = styles_map["disclaimer"]

    story = []

    # ── 1. Header (Total printable width = 178mm) ───────────────────────────
    scan_date = str(p["created_at"])[:19] if p["created_at"] else "N/A"
    report_id = f"AIT-FBD-{p['id']:06d}"

    header_left = [
        Paragraph("AIT INSTITUTE OF TECHNOLOGY", hospital_header_style),
        Spacer(1, 0.4 * mm),
        Paragraph("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING", dept_style),
        Paragraph("Chikkamagaluru, Karnataka 577102 • fetalbraingroup@gmail.com", sub_address_style),
    ]
    header_right = [
        Paragraph("FETAL BRAIN DETECTION REPORT", doc_title_style),
        Spacer(1, 0.4 * mm),
        Paragraph(f"<b>Report ID:</b> {report_id}", meta_style),
        Paragraph(f"<b>Scan Date:</b> {scan_date}", meta_style),
        Paragraph("<b>Status:</b> Complete", meta_style),
    ]
    header_table = Table([[header_left, header_right]], colWidths=[110 * mm, 68 * mm])
    header_table.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(HRFlowable(
        width="100%", thickness=0.8,
        color=colors.HexColor("#0f766e"),
        spaceAfter=2 * mm, spaceBefore=1 * mm
    ))

    # ── 2. Patient Demographics & Record Details (178mm) ─────────────────────
    story.append(Paragraph("PATIENT RECORD & EXAM DETAILS", section_heading))

    pat_name  = p["patient_name"]    or "—"
    pat_email = p["patient_email"]   or "—"
    pat_phone = p["patient_phone"]   or "—"
    pat_msg   = p["patient_message"] or "Routine Fetal Ultrasound Scan"

    demo_data = [
        [
            Paragraph("Patient Name:",    label_style), Paragraph(pat_name,  value_style),
            Paragraph("Report ID:",       label_style), Paragraph(report_id, value_style),
        ],
        [
            Paragraph("Contact Phone:",   label_style), Paragraph(pat_phone, value_style),
            Paragraph("Email Address:",   label_style), Paragraph(pat_email, value_style),
        ],
        [
            Paragraph("Project Unit:",    label_style), Paragraph("Fetal Brain Research Project", value_style),
            Paragraph("Examination Note:", label_style), Paragraph(pat_msg,   value_style),
        ],
    ]
    demo_table = Table(demo_data, colWidths=[26 * mm, 63 * mm, 26 * mm, 63 * mm])
    demo_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 3.5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 2 * mm))

    # ── 3. Scan Image + Detection Findings (50mm + 128mm = 178mm) ────────────
    story.append(Paragraph("FETAL BRAIN SCAN & DETECTION FINDINGS", section_heading))

    img_path = os.path.join(UPLOAD_DIR, p["filename"]) if p["filename"] else None
    if img_path and os.path.exists(img_path):
        img_flowable = RLImage(img_path, width=46 * mm, height=31 * mm)
    else:
        img_flowable = Paragraph("<i>[ Fetal Brain Scan Image ]</i>", value_style)

    img_box = Table([[img_flowable]], colWidths=[50 * mm], rowHeights=[33 * mm])
    img_box.setStyle(TableStyle([
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))

    result_class = p["result_class"]
    confidence   = p["confidence"]

    if result_class.lower() == "normal":
        badge_fg    = "#15803d"
        status_text = "NORMAL SCAN"
    else:
        badge_fg    = "#b91c1c"
        status_text = "ANOMALY DETECTED"

    ai_badge_p = Paragraph(
        f'<font color="{badge_fg}"><b>{status_text} • {result_class.upper()}</b></font>',
        ParagraphStyle("Badge", fontName="Helvetica-Bold", fontSize=8, leading=10)
    )
    ai_details = [
        [Paragraph("Detected Condition:", label_style), Paragraph(f"<b>{result_class}</b>",                 value_style)],
        [Paragraph("Detection Confidence:", label_style), Paragraph(f"<b>{confidence}%</b> (High Certainty)", value_style)],
        [Paragraph("Model Architecture:",  label_style), Paragraph("Swin Transformer",                       value_style)],
        [Paragraph("Scan Classification:",  label_style), ai_badge_p],
    ]
    ai_table = Table(ai_details, colWidths=[36 * mm, 92 * mm])
    ai_table.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 3.5),
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
    ]))

    scan_ai_layout = Table([[img_box, ai_table]], colWidths=[50 * mm, 128 * mm])
    scan_ai_layout.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    story.append(scan_ai_layout)
    story.append(Spacer(1, 2 * mm))

    # ── 4. Attention Heatmap & Anomaly Segmentation (89mm + 89mm = 178mm) ─────
    story.append(Paragraph("ATTENTION HEATMAP & ANOMALY SEGMENTATION", section_heading))

    stem = os.path.splitext(p["filename"])[0] if p["filename"] else ""
    cam_file = os.path.join(EXPLANATION_DIR, f"{stem}_gradcam.jpg")
    seg_file = os.path.join(SEGMENTATION_DIR, f"{stem}_segmentation.jpg")

    if not (os.path.exists(cam_file) and os.path.exists(seg_file)) and img_path and os.path.exists(img_path):
        try:
            build_explainability_outputs(img_path, stem)
        except Exception as e:
            print("[PDF Generation Note] Auto-building explainability outputs:", e)

    caption_style = ParagraphStyle(
        "XaiCaption",
        fontName="Helvetica",
        fontSize=6.8,
        leading=8.2,
        textColor=colors.HexColor("#334155"),
        alignment=TA_CENTER
    )

    cam_cell = []
    if os.path.exists(cam_file):
        cam_cell.append(RLImage(cam_file, width=70 * mm, height=29 * mm))
    else:
        cam_cell.append(Paragraph("<i>[ Heatmap Unavailable ]</i>", value_style))
    cam_cell.append(Spacer(1, 0.5 * mm))
    cam_cell.append(Paragraph("<b>Grad-CAM Saliency Heatmap</b> (Attention Localization)", caption_style))

    seg_cell = []
    if os.path.exists(seg_file):
        seg_cell.append(RLImage(seg_file, width=70 * mm, height=29 * mm))
    else:
        seg_cell.append(Paragraph("<i>[ Segmentation Unavailable ]</i>", value_style))
    seg_cell.append(Spacer(1, 0.5 * mm))
    seg_cell.append(Paragraph("<b>Anomaly Pattern Segmentation</b> (Lesion Delineation)", caption_style))

    xai_table = Table([[cam_cell, seg_cell]], colWidths=[89 * mm, 89 * mm])
    xai_table.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(xai_table)
    story.append(Spacer(1, 2 * mm))

    # ── 5. Observations & Recommendations (178mm) ────────────────────────────
    story.append(Paragraph("OBSERVATION SUMMARY & RECOMMENDATIONS", section_heading))

    if result_class.lower() == "normal":
        clinical_desc = (
            "No structural anomalies detected in the fetal brain scan. Anatomy appears within normal limits for reported gestational age."
        )
        next_steps = (
            "• Continue standard antenatal monitoring.<br/>"
            "• Repeat scan as scheduled by the healthcare provider."
        )
    else:
        clinical_desc = (
            f"The research model identified features consistent with <b>{result_class}</b> (confidence: <b>{confidence}%</b>). "
            "Findings should be correlated with comprehensive sonographic assessment."
        )
        next_steps = (
            "• Detailed ultrasound target scan recommended for confirmation.<br/>"
            "• Multi-disciplinary clinical evaluation and specialist counselling."
        )

    clinical_text = (
        f"<b>Summary:</b> {clinical_desc}<br/>"
        f"<b>Recommendations:</b><br/>{next_steps}"
    )
    obs_box = Table([[Paragraph(clinical_text, body_style)]], colWidths=[178 * mm])
    obs_box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(obs_box)
    story.append(Spacer(1, 2 * mm))

    # ── 6. Verification & Project Review (96mm + 82mm = 178mm) ───────────────
    story.append(Paragraph("VERIFICATION & RESEARCH REVIEW", section_heading))

    ver_code = f"VER-AIT-{p['id']:06d}-X"
    sig_left = [
        Paragraph("<b>System:</b> Fetal Brain Detection System", value_style),
        Paragraph(f"<b>Verification Code:</b> {ver_code}",             value_style),
        Spacer(1, 0.5 * mm),
        Paragraph("<i>Digitally generated by Fetal Brain Detection System</i>", sub_address_style),
    ]
    sig_right = [
        Paragraph(
            "<b>Project Research Group</b>",
            ParagraphStyle("DocSig", fontName="Helvetica-Bold", fontSize=7.5, textColor=colors.HexColor("#0f766e"), alignment=2)
        ),
        Paragraph("AIT Institute of Technology, Chikkamagaluru", meta_style),
        Paragraph("Email: fetalbraingroup@gmail.com",             meta_style),
    ]
    sig_table = Table([[sig_left, sig_right]], colWidths=[96 * mm, 82 * mm])
    sig_table.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 1.8 * mm))

    # ── 7. Academic & Research Disclaimer (178mm) ─────────────────────────────
    disclaimer_text = (
        "<b>ACADEMIC & RESEARCH DISCLAIMER:</b> This report is generated for academic and research evaluation purposes only. "
        "It is not an official medical diagnosis. The findings and confidence scores must be independently reviewed and "
        "verified by qualified medical specialists."
    )
    disc_box = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[178 * mm])
    disc_box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX",           (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(disc_box)

    return story


def build_professional_pdf_response(predictions, filename="fetal_brain_report.pdf"):
    """Generate a clean, single-page A4 diagnostic report PDF for one or more predictions."""
    from reportlab.platypus import PageBreak
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Fetal Brain Detection Report - AIT Chikkamagaluru"
    )

    # ── Shared paragraph styles ───────────────────────────────────────────────
    styles_map = {
        "hospital_header": ParagraphStyle(
            "HospitalHeader", fontName="Helvetica-Bold", fontSize=11,
            leading=13, textColor=colors.HexColor("#0f766e")
        ),
        "dept": ParagraphStyle(
            "DeptStyle", fontName="Helvetica-Bold", fontSize=7.5,
            leading=9.5, textColor=colors.HexColor("#334155")
        ),
        "sub_address": ParagraphStyle(
            "SubAddress", fontName="Helvetica", fontSize=7,
            leading=8.5, textColor=colors.HexColor("#64748b")
        ),
        "doc_title": ParagraphStyle(
            "DocTitle", fontName="Helvetica-Bold", fontSize=10.5,
            leading=12.5, textColor=colors.HexColor("#1e293b"), alignment=2
        ),
        "meta": ParagraphStyle(
            "MetaText", fontName="Helvetica", fontSize=7,
            leading=9, textColor=colors.HexColor("#475569"), alignment=2
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading", fontName="Helvetica-Bold", fontSize=8,
            leading=10, textColor=colors.HexColor("#0f766e"), spaceAfter=1.5
        ),
        "label": ParagraphStyle(
            "LabelStyle", fontName="Helvetica-Bold", fontSize=7,
            leading=9, textColor=colors.HexColor("#475569")
        ),
        "value": ParagraphStyle(
            "ValueStyle", fontName="Helvetica", fontSize=7,
            leading=9, textColor=colors.HexColor("#0f172a")
        ),
        "body": ParagraphStyle(
            "BodyStyle", fontName="Helvetica", fontSize=7,
            leading=9, textColor=colors.HexColor("#1e293b")
        ),
        "disclaimer": ParagraphStyle(
            "DisclaimerStyle", fontName="Helvetica-Oblique", fontSize=6,
            leading=7.5, textColor=colors.HexColor("#64748b")
        ),
    }

    full_story = []
    for i, p in enumerate(predictions):
        full_story.extend(_build_single_prediction_story(p, styles_map))
        if i < len(predictions) - 1:
            full_story.append(PageBreak())

    doc.build(full_story, canvasmaker=ProfessionalReportCanvas)
    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.route("/profile/export/csv")
@login_required
def export_csv():
    return build_csv_response(get_user_predictions())


@app.route("/profile/export/pdf")
@login_required
def export_pdf():
    return build_professional_pdf_response(
        get_user_predictions(),
        filename="fetal_brain_full_report.pdf"
    )


@app.route("/profile/export/csv/<int:prediction_id>")
@login_required
def export_single_csv(prediction_id):
    p = get_prediction_owned(prediction_id)
    if not p:
        flash("Prediction not found.", "danger")
        return redirect(url_for("profile"))
    return build_csv_response([p])


@app.route("/download_report/<int:prediction_id>")
@app.route("/profile/export/pdf/<int:prediction_id>")
@login_required
def export_single_pdf(prediction_id):
    p = get_prediction_owned(prediction_id)
    if not p:
        flash("Prediction not found.", "danger")
        return redirect(url_for("profile"))
    return build_professional_pdf_response([p], filename=f"fetal_report_{prediction_id}.pdf")


@app.route("/analyze")
@login_required
def analyze():
    result = session.pop("last_result", None)
    return render_template(
        "predict.html",
        prediction=result["class"] if result else None,
        confidence=result["confidence"] if result else None,
        image_file=result["image"] if result else None,
        segmentation_file=result.get("segmentation") if result else None,
        gradcam_file=result.get("gradcam") if result else None,
        segmentation_mode=result.get("segmentation_mode") if result else None,
        segmentation_trained=result.get("segmentation_trained") if result else False,
        gradcam_available=result.get("gradcam_available") if result else False,
        prediction_id=result.get("prediction_id") if result else None,
        patient_name=result.get("patient_name") if result else None,
        patient_email=result.get("patient_email") if result else None,
        patient_phone=result.get("patient_phone") if result else None,
        patient_message=result.get("patient_message") if result else None
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        flash("Model not loaded. Contact admin.", "danger")
        return redirect(url_for("analyze"))

    if "file" not in request.files:
        flash("No file selected.", "danger")
        return redirect(url_for("analyze"))

    file = request.files["file"]
    if not file.filename:
        flash("No file selected.", "danger")
        return redirect(url_for("analyze"))

    if not allowed_file(file.filename):
        flash("Allowed file types: JPG, JPEG, PNG, BMP.", "danger")
        return redirect(url_for("analyze"))

    try:
        ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        file.save(save_path)

        # Segmentation first. A trained U-Net produces the ROI used for the
        # classifier. The demo fallback is visualization-only and does not
        # change the existing classifier prediction.
        uploaded_image = PILImage.open(save_path).convert("RGB")
        mask, seg_mode = predict_mask(uploaded_image, segmentation_model, "cpu")
        roi_image = crop_to_mask(uploaded_image, mask) if seg_mode == "trained_unet" else uploaded_image
        result_class, confidence, pred_idx, input_tensor = predict_image(save_path, roi_image)
        explain = build_explainability_outputs(
            save_path,
            os.path.splitext(unique_name)[0],
            classification_image=roi_image,
            pred_idx=pred_idx,
            input_tensor=input_tensor
        )

        patient_name = request.form.get("patient_name", "").strip()
        patient_email = request.form.get("patient_email", "").strip()
        patient_phone = request.form.get("patient_phone", "").strip()
        patient_message = request.form.get("patient_message", "").strip()

        db = get_db()
        current_uid = 0 if (session.get("is_admin") or session.get("user_id") == "admin") else session["user_id"]
        cursor = db.execute(
            """INSERT INTO predictions
               (user_id, filename, original_name, result_class, confidence,
                patient_name, patient_email, patient_phone, patient_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (current_uid, unique_name, file.filename, result_class, confidence,
             patient_name, patient_email, patient_phone, patient_message)
        )
        db.commit()
        last_id = cursor.lastrowid

        session["last_result"] = {
            "prediction_id": last_id,
            "class": result_class,
            "confidence": confidence,
            "image": unique_name,
            "segmentation": explain["segmentation"],
            "gradcam": explain["gradcam"],
            "segmentation_mode": explain["segmentation_mode"],
            "segmentation_trained": explain["segmentation_trained"],
            "gradcam_available": explain["gradcam_available"],
            "patient_name": patient_name,
            "patient_email": patient_email,
            "patient_phone": patient_phone,
            "patient_message": patient_message
        }
        return redirect(url_for("analyze"))

    except Exception as e:
        flash(f"Prediction error: {str(e)}", "danger")
        return redirect(url_for("analyze"))


# ==========================================
# ENTRY
# ==========================================

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)

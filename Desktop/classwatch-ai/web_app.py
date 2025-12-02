#!/usr/bin/env python3
"""
Classroom Attention Monitor - Reconnaissance Faciale CORRIGÉE
- Vérifie le visage à CHAQUE frame
- Pas de nouveaux IDs quand sortie/retour du cadre
- ✅ SÉCURISÉ: Aucune clé API en dur
"""

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
from ultralytics import YOLO
import mediapipe as mp
import time
from collections import defaultdict, deque
import threading
import requests
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import shutil

# ========================================
# CONFIGURATION ENVIRONNEMENT (SÉCURISÉE)
# ========================================
load_dotenv()

# Variables d'environnement - AUCUNE CLÉ EN DUR
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Vérification des clés disponibles au démarrage
print("🔐 VÉRIFICATION SÉCURITÉ DES CLÉS API:")
print(f"   - GOOGLE_API_KEY: {'✅ Présente' if GOOGLE_API_KEY else '❌ Absente'}")
print(f"   - GROQ_API_KEY: {'✅ Présente' if GROQ_API_KEY else '❌ Absente'}")

if not GOOGLE_API_KEY and not GROQ_API_KEY:
    print("\n⚠️  ATTENTION: Aucune clé API configurée!")
    print("   📋 Actions requises:")
    print("      1. Créez un fichier .env à la racine du projet")
    print("      2. Ajoutez: GOOGLE_API_KEY=votre_clé_google")
    print("      3. Ou ajoutez: GROQ_API_KEY=votre_clé_groq")
    print("      4. Les fonctions IA seront désactivées sans clés\n")

# ========================================
# CLIENT LLM GRATUITS ÉDUCATION
# ========================================
import sys
import traceback
sys.path.append('/workspace')

class EducationLLMClient:
    """Client LLM pour usage éducatif avec limites élevées"""
    
    def __init__(self, provider="google_gemini"):
        self.llm = None
        self.provider = provider
        self.configured = False
        
        # Configuration des clés API depuis les variables d'environnement (SÉCURISÉ)
        if provider == "google_gemini" and GOOGLE_API_KEY:
            try:
                self.client = GoogleGeminiClient(GOOGLE_API_KEY)
                self.configured = True
                print(f"✅ Google Gemini configuré - 3M tokens/jour")
            except Exception as e:
                print(f"⚠️ Google Gemini erreur: {str(e)}")
                self.configured = False
        
        elif provider == "groq" and GROQ_API_KEY:
            try:
                self.client = GroqClient(GROQ_API_KEY)
                self.configured = True
                print(f"✅ Groq configuré - 500K tokens/jour")
            except Exception as e:
                print(f"⚠️ Groq erreur: {str(e)}")
                self.configured = False
        
        else:
            print(f"❌ Aucune clé API disponible pour {provider}")
            self.configured = False
    
    def generate(self, prompt, max_tokens=500, temperature=0.7):
        """Génère une réponse via le provider configuré"""
        if not self.configured:
            return "❌ Client LLM non configuré. Vérifiez les clés API dans .env"
        
        try:
            if self.provider == "google_gemini":
                return self.client.generate(prompt, max_tokens, temperature)
            elif self.provider == "groq":
                return self.client.generate(prompt, max_tokens, temperature)
        except Exception as e:
            print(f"❌ Erreur génération: {str(e)}")
            return f"❌ Erreur: {str(e)}"

class GoogleGeminiClient:
    """Client Google Gemini - SÉCURISÉ"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-2.5-flash"
    
    def generate(self, prompt, max_tokens=500, temperature=0.7):
        """Génère une réponse avec Google Gemini"""
        try:
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature
                }
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "⚠️ Réponse Google Gemini vide"
            elif response.status_code == 401:
                return "❌ Clé Google API invalide"
            elif response.status_code == 429:
                return "❌ Limite Google API atteinte"
            else:
                return f"❌ Erreur Google API: {response.status_code}"
        
        except requests.exceptions.Timeout:
            return "❌ Timeout Google API"
        except Exception as e:
            return f"❌ Erreur Google Gemini: {str(e)}"

class GroqClient:
    """Client Groq - GRATUIT & SÉCURISÉ"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama3-70b-8192"
    
    def generate(self, prompt, max_tokens=500, temperature=0.7):
        """Génère une réponse avec Groq"""
        try:
            url = f"{self.base_url}/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
                else:
                    return "⚠️ Réponse Groq vide"
            elif response.status_code == 401:
                return "❌ Clé Groq API invalide"
            elif response.status_code == 429:
                return "❌ Limite Groq atteinte (500K tokens/jour)"
            else:
                return f"❌ Erreur Groq API: {response.status_code}"
        
        except requests.exceptions.Timeout:
            return "❌ Timeout Groq API"
        except Exception as e:
            return f"❌ Erreur Groq: {str(e)}"

print("\n🔧 INITIALISATION...")

# ========================================
# CONFIGURATION
# ========================================

SESSION_FILE = 'session_data.json'
STUDENTS_DB_FILE = 'students_database.json'
SCREENSHOTS_DIR = 'screenshots'
SCREENSHOT_CATEGORIES = ['new_student', 'orange_distraction', 'red_distraction', 'conversation', 'proximity']

# Créer le dossier screenshots avec sous-dossiers par catégorie
def setup_screenshots_folders():
    """Crée la structure de dossiers pour les captures d'écran"""
    if not os.path.exists(SCREENSHOTS_DIR):
        os.makedirs(SCREENSHOTS_DIR)
    
    for category in SCREENSHOT_CATEGORIES:
        category_dir = os.path.join(SCREENSHOTS_DIR, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
    
    print(f"📁 Dossiers de captures créés dans: {SCREENSHOTS_DIR}")

def take_screenshot_with_overlay(frame, face_id, category, additional_info=None):
    """Prend une capture d'écran avec overlay d'informations"""
    try:
        # Créer une copie de la frame avec overlay
        screenshot_frame = frame.copy()
        
        # Ajouter un overlay d'informations
        height, width = screenshot_frame.shape[:2]
        
        # Fond semi-transparent pour le texte
        overlay = screenshot_frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, screenshot_frame, 0.3, 0, screenshot_frame)
        
        # Informations à afficher
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        student_name = tracker.face_recognizer.known_faces.get(face_id, {}).get('name', face_id)
        
        # Couleur basée sur la catégorie
        color_map = {
            'new_student': (0, 255, 0),      # Vert
            'orange_distraction': (0, 165, 255),  # Orange
            'red_distraction': (0, 0, 255),       # Rouge
            'conversation': (255, 0, 255),        # Magenta
            'proximity': (255, 0, 0)              # Rouge vif
        }
        category_color = color_map.get(category, (255, 255, 255))
        
        # Texte d'information
        texts = [
            f"📸 CAPTURE AUTOMATIQUE",
            f"👤 Étudiant: {student_name}",
            f"🆔 ID: {face_id}",
            f"⚠️ Catégorie: {category.upper().replace('_', ' ')}",
            f"📅 Timestamp: {timestamp}"
        ]
        
        if additional_info:
            texts.append(f"ℹ️ Info: {additional_info}")
        
        # Dessiner le texte
        y_offset = 30
        for i, text in enumerate(texts):
            cv2.putText(screenshot_frame, text, (20, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, category_color, 2)
        
        # Timestamp en haut à droite
        time_text = datetime.now().strftime("%H:%M:%S")
        cv2.putText(screenshot_frame, time_text, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return screenshot_frame
        
    except Exception as e:
        print(f"⚠️ Erreur création overlay: {e}")
        return frame

def save_screenshot(frame, face_id, category, reason="", detected_students=None):
    """Sauvegarde une capture d'écran dans le dossier approprié"""
    try:
        # Créer le nom de fichier unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        student_name = tracker.face_recognizer.known_faces.get(face_id, {}).get('name', face_id)
        safe_name = "".join(c for c in student_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{timestamp}_{category}_{safe_name}_{face_id}.jpg"
        
        # Chemin complet
        category_dir = os.path.join(SCREENSHOTS_DIR, category)
        filepath = os.path.join(category_dir, filename)
        
        # Sauvegarder l'image
        cv2.imwrite(filepath, frame)
        
        # Log de la capture
        print(f"📸 CAPTURE SAUVEGARDÉE: {filename}")
        print(f"   📁 Dossier: {category}")
        print(f"   👤 Étudiant: {student_name} ({face_id})")
        print(f"   📋 Raison: {reason}")
        
        # Créer un fichier JSON de métadonnées
        metadata = {
            'filename': filename,
            'category': category,
            'face_id': face_id,
            'student_name': student_name,
            'timestamp': timestamp,
            'reason': reason,
            'detected_students_count': len(detected_students) if detected_students else 0,
            'screen_resolution': f"{frame.shape[1]}x{frame.shape[0]}"
        }
        
        metadata_file = filepath.replace('.jpg', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
        
    except Exception as e:
        print(f"⚠️ Erreur sauvegarde capture: {e}")
        return None

def trigger_screenshot_by_category(frame, face_id, category, reason="", detected_students=None):
    """Déclenche une capture basée sur la catégorie"""
    try:
        # Créer l'overlay avec informations
        screenshot_frame = take_screenshot_with_overlay(frame, face_id, category, reason)
        
        # Sauvegarder
        filepath = save_screenshot(screenshot_frame, face_id, category, reason, detected_students)
        
        return filepath
        
    except Exception as e:
        print(f"⚠️ Erreur capture par catégorie: {e}")
        return None

def load_students_db():
    if not os.path.exists(STUDENTS_DB_FILE):
        print(f"   📁 Fichier {STUDENTS_DB_FILE} non trouvé - création d'une nouvelle base")
        return {}
    try:
        with open(STUDENTS_DB_FILE, 'r') as f:
            data = json.load(f)
        print(f"   📁 Chargement base de données: {len(data)} étudiants trouvés")
        
        # Convertir les listes en numpy arrays et gérer la nouvelle structure
        new_data = {}
        for face_id, student_data in data.items():
            if isinstance(student_data, dict) and 'embedding' in student_data:
                # Nouvelle structure
                embedding = np.array(student_data['embedding'])
                if embedding.size > 0:  # Vérifier que l'embedding n'est pas vide
                    student_data['embedding'] = embedding
                    # Assurer que 'name' existe, sinon utiliser face_id
                    student_data['name'] = student_data.get('name', face_id)
                    new_data[face_id] = student_data
                    print(f"      ✅ {face_id}: embedding de {embedding.size} dimensions, nom: {student_data['name']}")
                else:
                    print(f"      ⚠️ {face_id}: embedding vide, ignoré")
            elif isinstance(student_data, list):
                # Ancienne structure (seulement l'embedding)
                embedding = np.array(student_data)
                if embedding.size > 0:
                    # Créer la nouvelle structure avec le nom par défaut
                    new_data[face_id] = {'embedding': embedding, 'name': face_id}
                    print(f"      ✅ {face_id}: format ancien, {embedding.size} dimensions, nom: {face_id}")
                else:
                    print(f"      ⚠️ {face_id}: embedding vide, ignoré")
            else:
                # Structure invalide, ignorer
                print(f"      ❌ {face_id}: structure invalide, ignoré")
                continue
        
        print(f"   📊 Base de données chargée: {len(new_data)} étudiants valides")
        return new_data
    except Exception as e:
        print(f"   ❌ Erreur chargement base de données: {e}")
        return {}

def save_students_db(db):
    try:
        # Convertir numpy arrays en listes pour JSON et gérer la nouvelle structure
        json_db = {}
        for face_id, student_data in db.items():
            if isinstance(student_data, dict) and 'embedding' in student_data:
                json_db[face_id] = {
                    'embedding': student_data['embedding'].tolist(),
                    'name': student_data.get('name', face_id)
                }
            else:
                # Ancienne structure ou invalide, on sauvegarde juste l'embedding si c'est un ndarray
                if isinstance(student_data, np.ndarray):
                    json_db[face_id] = {'embedding': student_data.tolist(), 'name': face_id}
                else:
                    # Structure invalide, on ignore
                    continue
        
        with open(STUDENTS_DB_FILE, 'w') as f:
            json.dump(json_db, f, indent=2)
        print(f"💾 Base de données sauvegardée ({len(db)} visages)")
    except Exception as e:
        print(f"⚠️  Erreur sauvegarde DB: {e}")

students_database = load_students_db()

# Initialiser les dossiers de captures
setup_screenshots_folders()

def save_session_data(tracker):
    # Convertir tous les types numpy en types Python natifs
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'phone_timers': convert_numpy_types(tracker.phone_timers),
        'all_detected_ids': [str(x) for x in tracker.all_detected_ids],
        'face_id_to_timer': convert_numpy_types(tracker.face_id_to_timer),
        'face_id_to_speaking_timer': convert_numpy_types(tracker.face_id_to_speaking_timer),
        'face_id_to_speaking': convert_numpy_types(tracker.face_id_to_speaking),
        'face_id_to_conversation': convert_numpy_types(tracker.face_id_to_conversation)
    }
    
    try:
        with open(SESSION_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️  Erreur sauvegarde: {e}")

def load_session_data():
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, 'r') as f:
            data = json.load(f)
        print(f"📂 Session chargée: {len(data.get('all_detected_ids', []))} étudiants")
        return data
    except Exception as e:
        print(f"⚠️  Erreur chargement: {e}")
        return None

# ========================================
# DÉTECTION CAMÉRAS
# ========================================

def list_available_cameras(max_test=10):
    print("\n📹 DÉTECTION DES CAMÉRAS...")
    available_cameras = []
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                backend_name = cap.getBackendName()
                available_cameras.append({
                    'index': i,
                    'backend': backend_name,
                    'name': f"Caméra {i}"
                })
                print(f"   ✅ Caméra {i} détectée ({backend_name})")
            cap.release()
    
    return available_cameras

def select_camera(cameras):
    if len(cameras) == 0:
        print("❌ Aucune caméra détectée!")
        return None
    
    if len(cameras) == 1:
        print(f"\n✅ Une seule caméra: Caméra {cameras[0]['index']}")
        return cameras[0]['index']
    
    print("\n📋 CAMÉRAS DISPONIBLES:")
    for cam in cameras:
        print(f"   [{cam['index']}] {cam['name']} - {cam['backend']}")
    
    if len(cameras) > 1:
        print(f"\n💡 Sélection automatique: Caméra {cameras[-1]['index']}")
        return cameras[-1]['index']
    
    return cameras[0]['index']

available_cameras = list_available_cameras()
selected_camera_index = select_camera(available_cameras)

if selected_camera_index is None:
    print("❌ Impossible de continuer sans caméra")
    exit(1)

# ========================================
# MODÈLES
# ========================================

print("📥 Chargement YOLOv8...")
model = YOLO('yolov8n.pt')
print("✅ Modèle chargé")

PERSON_CLASS = 0
PHONE_CLASS = 67

print("📥 Chargement YOLOv8-pose...")
try:
    pose_model = YOLO('yolov8n-pose.pt')
    pose_enabled = True
    print("✅ Détection posture activée")
except Exception as e:
    pose_enabled = False
    print(f"⚠️  Détection posture désactivée: {e}")

# ========================================
# CONFIGURATION LLM (SÉCURISÉE - SANS CLÉS EN DUR)
# ========================================

openai_enabled = False
client = None

print("\n🤖 CONFIGURATION ASSISTANT IA:")

if GOOGLE_API_KEY:
    print("🔄 Tentative configuration Google Gemini...")
    try:
        client = EducationLLMClient("google_gemini")
        if client.configured:
            openai_enabled = True
            openai_model = "gemini-2.5-flash"
            print("✅ Google Gemini ACTIF (3M tokens/jour)")
        else:
            print("⚠️  Google Gemini échoué, essai Groq...")
            if GROQ_API_KEY:
                client = EducationLLMClient("groq")
                if client.configured:
                    openai_enabled = True
                    openai_model = "llama3-70b-8192"
                    print("✅ Groq ACTIF en fallback (500K tokens/jour)")
    except Exception as e:
        print(f"❌ Erreur Google: {str(e)}")

elif GROQ_API_KEY:
    print("🔄 Tentative configuration Groq...")
    try:
        client = EducationLLMClient("groq")
        if client.configured:
            openai_enabled = True
            openai_model = "llama3-70b-8192"
            print("✅ Groq ACTIF (500K tokens/jour)")
    except Exception as e:
        print(f"❌ Erreur Groq: {str(e)}")

if not openai_enabled:
    print("⚠️  AUCUNE IA DISPONIBLE - Rapport et Chat désactivés")
    print("   📝 Pour activer, configurez .env avec:")
    print("      GOOGLE_API_KEY=votre_clé")
    print("      ou")
    print("      GROQ_API_KEY=votre_clé")

# ========================================
# RECONNAISSANCE FACIALE AMÉLIORÉE
# ========================================

class ImprovedFaceRecognizer:
    """Reconnaissance faciale qui vérifie à CHAQUE frame"""
    
    def __init__(self):
        global students_database
        self.known_faces = students_database  # Charger base existante
        self.next_face_id = len(self.known_faces) + 1
        self.db_lock = threading.Lock()
        
        # Initialisation de MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7)
        
        # Initialisation de MediaPipe Face Mesh pour détection de la bouche
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        print(f"✅ Reconnaissance faciale: {len(self.known_faces)} visages en mémoire")
        
        # Debug : Afficher les IDs existants au démarrage
        if self.known_faces:
            print(f"   📋 IDs étudiants existants: {list(self.known_faces.keys())}")
        else:
            print(f"   📋 Aucun visage enregistré - les nouveaux visages seront créés")
    
    def detect_speaking(self, frame, box):
        """Détecte si la personne parle en analysant les mouvements de la bouche"""
        try:
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return False
            
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                return False
            
            # Conversion BGR vers RGB pour MediaPipe
            rgb_frame = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return False
            
            landmarks = results.multi_face_landmarks[0]
            
            # Points de la bouche (indices MediaPipe)
            upper_lip_points = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
            lower_lip_points = [78, 79, 80, 81, 82, 83, 84, 85, 86, 87]
            
            # Calculer la hauteur de la bouche
            upper_y = np.mean([landmarks.landmark[i].y for i in upper_lip_points])
            lower_y = np.mean([landmarks.landmark[i].y for i in lower_lip_points])
            mouth_openness = abs(lower_y - upper_y)
            
            # Seuil pour déterminer si la bouche est ouverte
            speaking_threshold = 0.015
            
            return mouth_openness > speaking_threshold
            
        except Exception as e:
            return False

    def extract_face_embedding(self, frame, box):
        """Extrait embedding du visage avec prétraitement amélioré"""
        x1, y1, x2, y2 = map(int, box)
        
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return None

        try:
            # Prétraitement amélioré
            face_resized = cv2.resize(face_img, (64, 64))
            
            # Histogram equalization
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_equalized = cv2.equalizeHist(face_gray)
            
            # Normalisation robuste
            face_normalized = face_equalized.astype(float) / 255.0
            
            # Filtrage pour réduire le bruit
            face_filtered = cv2.GaussianBlur(face_normalized, (3, 3), 0)
            
            # Aplatir et retourner l'embedding
            embedding = face_filtered.flatten()
            
            return embedding
        except Exception as e:
            print(f"   ⚠️ Erreur extraction embedding: {e}")
            return None
    
    def compute_similarity(self, emb1, emb2):
        """Calcule similarité améliorée avec multiple méthodes"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        try:
            # Méthode 1: Corrélation de Pearson
            corr_matrix = np.corrcoef(emb1, emb2)
            pearson_sim = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
            
            # Méthode 2: Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
            
            # Méthode 3: Distance euclidienne (inversée)
            euclidean_dist = np.linalg.norm(emb1 - emb2)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            
            # Combinaison pondérée des trois méthodes
            combined_sim = (0.5 * abs(pearson_sim) + 0.3 * abs(cosine_sim) + 0.2 * euclidean_sim)
            
            # S'assurer que la similarité est entre 0 et 1
            combined_sim = max(0.0, min(1.0, combined_sim))
            
            return combined_sim
        except Exception as e:
            print(f"   ⚠️ Erreur calcul similarité: {e}")
            return 0.0
    
    def identify_face(self, frame, person_box):
        """
        Identifie un visage - APPELÉ À CHAQUE FRAME
        Retourne (face_id, confidence)
        """
        embedding = self.extract_face_embedding(frame, person_box)
        if embedding is None:
            return None, 0.0
        
        # Comparer avec TOUS les visages connus
        best_match = None
        best_score = 0.0
        threshold = 0.50
        
        print(f"   🔍 Recherche correspondance parmi {len(self.known_faces)} visages connus...")
        
        for face_id, student_data in self.known_faces.items():
            similarity = self.compute_similarity(embedding, student_data['embedding'])
            if similarity > best_score:
                best_score = similarity
                best_match = face_id
        
        print(f"   📊 Meilleure correspondance: {best_match} (score: {best_score:.3f}, seuil: {threshold})")
        
        # Si bonne correspondance
        if best_match and best_score >= threshold:
            # Mettre à jour embedding (moyenne mobile lente)
            with self.db_lock:
                old_embedding = self.known_faces[best_match]['embedding'].copy()
                self.known_faces[best_match]['embedding'] = (old_embedding * 0.85 + embedding * 0.15)
            print(f"   ✅ Visage reconnu: {best_match} (confiance: {best_score:.3f})")
            return best_match, best_score
        
        # SYSTÈME ANTI-FAUX POSITIFS
        potential_name_conflicts = []
        for face_id, student_data in self.known_faces.items():
            if 'name' in student_data and not student_data['name'].startswith('Student_'):
                similarity_with_named = self.compute_similarity(embedding, student_data['embedding'])
                if similarity_with_named > 0.55:
                    potential_name_conflicts.append((face_id, similarity_with_named))
        
        # NOUVEAU visage seulement si aucun score n'est suffisant
        if (best_score < 0.3 or best_match is None) and not potential_name_conflicts:  
            face_id = f"Student_{chr(65 + (self.next_face_id - 1) % 26)}{self.next_face_id:02d}"
            self.next_face_id += 1
            
            new_student_data = {'embedding': embedding, 'name': face_id}
            
            with self.db_lock:
                self.known_faces[face_id] = new_student_data
            
            print(f"   🆕 NOUVEAU visage créé: {face_id} (meilleur score: {best_score:.3f})")
            
            # Sauvegarder immédiatement
            save_students_db(self.known_faces)
            
            return face_id, 1.0
        
        # Si conflit de nom détecté
        if potential_name_conflicts:
            best_named_match, best_named_score = max(potential_name_conflicts, key=lambda x: x[1])
            print(f"   🔄 Conflit nom détecté: {best_named_match} (sim: {best_named_score:.3f})")
            with self.db_lock:
                old_embedding = self.known_faces[best_named_match]['embedding'].copy()
                self.known_faces[best_named_match]['embedding'] = (old_embedding * 0.8 + embedding * 0.2)
            return best_named_match, best_named_score
        
        print(f"   ⚠️ Correspondance incertaine: {best_match} (score: {best_score:.3f})")
        return best_match, best_score


class ContinuousFaceTracker:
    """Tracker qui vérifie l'identité faciale à CHAQUE frame"""
    
    def __init__(self, max_disappeared=25, decay_rate=0.6):
    
        # Timer par FACE_ID
        self.face_id_to_timer = defaultdict(float)
        self.face_id_to_last_active = defaultdict(float)
        self.face_id_to_state = defaultdict(bool)
        
        # Speaking tracking
        self.face_id_to_speaking = defaultdict(bool)
        self.face_id_to_conversation = defaultdict(bool)
        self.face_id_to_speaking_timer = defaultdict(float)
        self.speaking_history = defaultdict(lambda: deque(maxlen=20))
        
        # Mapping temporaire track_id -> face_id
        self.track_to_face = {}
        
        # Stats
        self.last_seen_frame = defaultdict(int)
        self.history = defaultdict(lambda: deque(maxlen=30))
        
        self.max_disappeared = max_disappeared
        self.decay_rate = decay_rate
        self.current_frame = 0
        self.active_face_ids = set()
        self.all_detected_ids = set()
        
        # Reconnaissance
        self.face_recognizer = ImprovedFaceRecognizer()
        
        # Phone timers
        self.phone_timers = defaultdict(float)
        
        # Nouvelles détections
        self.face_id_to_proximity_timer = defaultdict(float)
        self.face_id_to_proximity_detected = defaultdict(bool)
        self.face_id_to_top_color = {}
        self.frame_counter = 0
        
        # Charger session
        saved_data = load_session_data()
        if saved_data:
            print("🔄 Restauration session...")
            for face_id_str, timer in saved_data.get('face_id_to_timer', {}).items():
                self.face_id_to_timer[face_id_str] = float(timer)
            for face_id_str, speaking_timer in saved_data.get('face_id_to_speaking_timer', {}).items():
                self.face_id_to_speaking_timer[face_id_str] = float(speaking_timer)
            for face_id_str, is_speaking in saved_data.get('face_id_to_speaking', {}).items():
                self.face_id_to_speaking[face_id_str] = bool(is_speaking)
            for face_id_str, is_conversation in saved_data.get('face_id_to_conversation', {}).items():
                self.face_id_to_conversation[face_id_str] = bool(is_conversation)
            for face_id in self.face_id_to_timer.keys():
                self.all_detected_ids.add(face_id)
        
        print(f"✅ Tracker avec vérification continue ({decay_rate}x)")
    
    def is_looking_down(self, keypoints):
        if keypoints is None or len(keypoints) < 7:
            return False
        
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        if nose[2] < 0.3 or left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3:
            return False
        
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        nose_y = nose[1]
        
        if nose_y > shoulder_y - 20:
            return True
        
        return False
    
    def detect_conversation(self, current_face_ids, current_face_id, is_current_speaking):
        """Détecte si c'est une conversation entre étudiants"""
        if not is_current_speaking:
            return False
        
        # Compter combien d'étudiants parlent actuellement
        speaking_count = 0
        for face_id in current_face_ids:
            if not face_id.startswith("Track_") and self.face_id_to_speaking.get(face_id, False):
                speaking_count += 1
        
        # C'est une conversation si au moins 2 étudiants parlent
        return speaking_count >= 2
    
    def detect_proximity(self, current_face_ids, current_face_id, person_boxes):
        """Détecte si 2 étudiants sont trop proches pendant 6 secondes"""
        if current_face_id.startswith("Track_"):
            return False
            
        # Trouver la boîte de l'étudiant actuel
        current_box = None
        for track_id, box in person_boxes:
            face_id = self.track_to_face.get(track_id, f"Track_{track_id}")
            if face_id == current_face_id:
                current_box = box
                break
                
        if current_box is None:
            return False
            
        # Vérifier la proximité avec les autres étudiants
        proximity_detected = False
        for track_id, box in person_boxes:
            face_id = self.track_to_face.get(track_id, f"Track_{track_id}")
            if face_id != current_face_id and not face_id.startswith("Track_"):
                # Calculer la distance entre les boîtes
                dist_x = abs((current_box[0] + current_box[2])/2 - (box[0] + box[2])/2)
                dist_y = abs((current_box[1] + current_box[3])/2 - (box[1] + box[3])/2)
                
                # Si trop proche
                if dist_x < 100 and dist_y < 100:
                    proximity_detected = True
                    break
        
        # Mettre à jour le timer de proximité
        if proximity_detected:
            self.face_id_to_proximity_timer[current_face_id] += 1/30
        else:
            if self.face_id_to_proximity_timer[current_face_id] > 0:
                self.face_id_to_proximity_timer[current_face_id] = max(0, 
                    self.face_id_to_proximity_timer[current_face_id] - 1/30)
        
        # Détection vraie si timer > 5 secondes
        return self.face_id_to_proximity_timer[current_face_id] > 5.0
    
    def detect_top_color(self, frame, person_box, face_id):
        """Détecte la couleur du haut de l'étudiant"""
        self.frame_counter += 1
        
        # Détecter la couleur toutes les 30 frames (1 seconde à 30 FPS)
        if self.frame_counter % 30 != 0:
            return
            
        try:
            # Extraire la région du haut
            x1, y1, x2, y2 = person_box
            height = y2 - y1
            
            # Région du haut
            top_y1 = int(y1 + height * 0.4)
            top_y2 = int(y1 + height * 0.8)
            top_x1 = int(x1 + (x2-x1) * 0.2)
            top_x2 = int(x1 + (x2-x1) * 0.8)
            
            # Extraire la région
            top_region = frame[top_y1:top_y2, top_x1:top_x2]
            
            if top_region.size == 0:
                return
                
            # Calculer la couleur moyenne
            avg_color = np.mean(top_region, axis=(0, 1))
            
            # Déterminer la couleur dominante
            color_name = self._get_color_name(avg_color)
            
            # Stocker la couleur pour cette session
            if face_id not in self.face_id_to_top_color:
                self.face_id_to_top_color[face_id] = {
                    'color': color_name,
                    'rgb': avg_color.tolist(),
                    'confidence': 0.8
                }
            
        except Exception as e:
            pass
    
    def _get_color_name(self, rgb_color):
        """Convertit RGB en nom de couleur"""
        r, g, b = rgb_color
        
        if r > 150 and g < 100 and b < 100:
            return "Rouge"
        elif r < 100 and g > 150 and b < 100:
            return "Vert"
        elif r < 100 and g < 100 and b > 150:
            return "Bleu"
        elif r > 150 and g > 150 and b < 100:
            return "Jaune"
        elif r > 150 and g < 100 and b > 150:
            return "Magenta"
        elif r < 100 and g > 150 and b > 150:
            return "Cyan"
        elif r > 150 and g > 150 and b > 150:
            return "Blanc"
        elif r < 80 and g < 80 and b < 80:
            return "Noir"
        elif r > 100 and g > 100 and b > 100:
            if r > g and r > b:
                return "Orange"
            elif g > r and g > b:
                return "Vert olive"
            else:
                return "Violet"
        else:
            return "Autre"
    
    def update(self, frame, detections_with_ids, phone_boxes, pose_data, timestamp):
        """Mise à jour avec vérification faciale continue"""
        self.current_frame += 1
        current_face_ids = set()
        
        for track_id, person_box in detections_with_ids:
            # VÉRIFIER LE VISAGE À CHAQUE FRAME
            face_id, confidence = self.face_recognizer.identify_face(frame, person_box)
            
            if face_id is None:
                face_id = f"Track_{track_id}"
                print(f"   ⚠️ Aucun visage détecté pour track_id {track_id}")
            
            # Mettre à jour mapping
            self.track_to_face[track_id] = face_id
            current_face_ids.add(face_id)
            
            # N'ajouter aux IDs détectés que si ce n'est pas un track temporaire
            if not face_id.startswith("Track_"):
                self.all_detected_ids.add(face_id)
            
            # Vérifier téléphone
            has_phone = self._has_phone_nearby(person_box, phone_boxes)
            
            # Vérifier posture
            looking_down = False
            if track_id in pose_data:
                looking_down = self.is_looking_down(pose_data[track_id])
            
            # Détecter si la personne parle
            is_speaking = self.face_recognizer.detect_speaking(frame, person_box)
            
            # Détecter si c'est une conversation
            is_conversation = self.detect_conversation(current_face_ids, face_id, is_speaking)
            
            # Détecter proximité
            is_proximity = self.detect_proximity(current_face_ids, face_id, detections_with_ids)
            self.face_id_to_proximity_detected[face_id] = is_proximity
            
            # CAPTURE AUTOMATIQUE pour conversation
            if is_conversation and not hasattr(self, f'conversation_captured_{face_id}'):
                trigger_screenshot_by_category(frame, face_id, 'conversation', 
                                             f"Conversation détectée avec {len([f for f in current_face_ids if not f.startswith('Track_')])} étudiants")
                setattr(self, f'conversation_captured_{face_id}', True)
            
            # CAPTURE AUTOMATIQUE pour proximité
            if is_proximity and not hasattr(self, f'proximity_captured_{face_id}'):
                trigger_screenshot_by_category(frame, face_id, 'proximity', 
                                             f"Proximité détectée (>5s)")
                setattr(self, f'proximity_captured_{face_id}', True)
            
            # Détecter la couleur du haut
            if not face_id.startswith("Track_"):
                self.detect_top_color(frame, person_box, face_id)
            
            # Mettre à jour l'état de parole
            previous_speaking = self.face_id_to_speaking.get(face_id, False)
            previous_conversation = self.face_id_to_conversation.get(face_id, False)
            self.face_id_to_speaking[face_id] = is_speaking
            self.face_id_to_conversation[face_id] = is_conversation
            
            # Compter les étudiants qui parlent
            speaking_students = [fid for fid in self.active_face_ids 
                               if not fid.startswith("Track_") and self.face_id_to_speaking.get(fid, False)]
            
            # Log les conversations
            if len(speaking_students) >= 2 and is_speaking:
                print(f"   💬 Conversation détectée: {', '.join(speaking_students)}")
            elif is_speaking and len(speaking_students) == 1:
                print(f"   🗣️ {face_id} parle")
            
            # Gérer les détections de plus de 5 secondes
            if is_speaking or is_conversation:
                speaking_duration = self.face_id_to_speaking_timer.get(face_id, 0)
                if speaking_duration > 5.0:
                    if not hasattr(self, 'long_speaking_alerts'):
                        self.long_speaking_alerts = set()
                    if face_id not in self.long_speaking_alerts:
                        student_name = self.face_recognizer.known_faces.get(face_id, {}).get('name', face_id)
                        print(f"   ⚠️ {student_name} parle depuis {speaking_duration:.1f}s - Attention soutenue!")
                        self.long_speaking_alerts.add(face_id)
            else:
                if hasattr(self, 'long_speaking_alerts'):
                    self.long_speaking_alerts.discard(face_id)
            
            # Si la personne parle
            if is_speaking:
                if face_id in self.face_id_to_last_active:
                    time_delta = timestamp - self.face_id_to_last_active[face_id]
                    if time_delta < 2.0:
                        self.face_id_to_speaking_timer[face_id] += time_delta
            else:
                if previous_speaking and self.face_id_to_speaking_timer[face_id] > 0:
                    pass
                elif self.face_id_to_speaking_timer[face_id] > 3.0:
                    time_delta = timestamp - self.face_id_to_last_active.get(face_id, timestamp)
                    if time_delta < 2.0:
                        self.face_id_to_speaking_timer[face_id] = max(0, self.face_id_to_speaking_timer[face_id] - time_delta * 0.3)
            
            # Ajouter à l'historique
            self.speaking_history[face_id].append({
                'timestamp': timestamp,
                'speaking': is_speaking,
                'speaking_time': self.face_id_to_speaking_timer[face_id]
            })
            
            is_distracted = has_phone or looking_down
            
            # CAPTURE AUTOMATIQUE pour distraction
            should_take_distraction_screenshot = False
            screenshot_category = None
            screenshot_reason = ""
            
            if is_distracted:
                phone_time = float(self.face_id_to_timer.get(face_id, 0))
                if phone_time >= 10 and phone_time < 30:
                    if not hasattr(self, f'distraction_{face_id}_orange_captured'):
                        should_take_distraction_screenshot = True
                        screenshot_category = 'orange_distraction'
                        screenshot_reason = f"Distraction moyenne détectée ({phone_time:.1f}s)"
                        setattr(self, f'distraction_{face_id}_orange_captured', True)
                        
                elif phone_time >= 30:
                    if not hasattr(self, f'distraction_{face_id}_red_captured'):
                        should_take_distraction_screenshot = True
                        screenshot_category = 'red_distraction'
                        screenshot_reason = f"Distraction sévère détectée ({phone_time:.1f}s)"
                        setattr(self, f'distraction_{face_id}_red_captured', True)
            
            # Déclencher la capture
            if should_take_distraction_screenshot:
                trigger_screenshot_by_category(frame, face_id, screenshot_category, 
                                             screenshot_reason, current_face_ids)
            
            # PREMIÈRE DÉTECTION ou RÉAPPARITION
            if face_id not in self.face_id_to_last_active:
                is_new_student = False
                if not face_id.startswith("Track_"):
                    if face_id in self.all_detected_ids:
                        print(f"   🔄 RÉAPPAparition: {face_id} (conf: {confidence:.2f})")
                    else:
                        print(f"   👤 NOUVEAU DÉTECTÉ: {face_id} (conf: {confidence:.2f})")
                        is_new_student = True
                self.face_id_to_timer[face_id] = 0.0
                self.face_id_to_state[face_id] = False
                
                # CAPTURE AUTOMATIQUE pour nouveau étudiant
                if is_new_student:
                    trigger_screenshot_by_category(frame, face_id, 'new_student', 
                                                 f"Nouveau étudiant détecté (confiance: {confidence:.2f})")
            
            # Calculer temps
            if face_id in self.face_id_to_last_active:
                time_delta = timestamp - self.face_id_to_last_active[face_id]
                
                if time_delta < 5.0:
                    if is_distracted:
                        self.face_id_to_timer[face_id] += time_delta
                    else:
                        if self.face_id_to_timer[face_id] > 0:
                            decrement = time_delta * self.decay_rate
                            self.face_id_to_timer[face_id] = max(0, self.face_id_to_timer[face_id] - decrement)
            
            # Mise à jour
            self.last_seen_frame[face_id] = self.current_frame
            self.face_id_to_last_active[face_id] = timestamp
            self.face_id_to_state[face_id] = is_distracted
            
            # Historique
            self.history[face_id].append({
                'timestamp': timestamp,
                'frame': self.current_frame,
                'has_phone': has_phone,
                'looking_down': looking_down,
                'is_distracted': is_distracted,
                'phone_time': self.face_id_to_timer[face_id],
                'is_speaking': is_speaking,
                'speaking_time': self.face_id_to_speaking_timer[face_id]
            })
        
        self.active_face_ids = self._get_active_ids(current_face_ids)
        
        # Sync phone_timers
        for face_id, timer in self.face_id_to_timer.items():
            self.phone_timers[face_id] = timer
        
        return self.get_stats()
    
    def _get_active_ids(self, current_face_ids):
        active = set()
        active.update(current_face_ids)
        
        for face_id, last_frame in self.last_seen_frame.items():
            frames_missing = self.current_frame - last_frame
            if frames_missing <= self.max_disappeared:
                active.add(face_id)
        
        return active
    
    def _has_phone_nearby(self, person_box, phone_boxes, threshold=0.05):
        """Vérifie si un téléphone est proche de la personne"""
        if not phone_boxes:
            return False
        
        px1, py1, px2, py2 = person_box
        
        for phx1, phy1, phx2, phy2 in phone_boxes:
            # Calcul de l'intersection
            x_overlap = max(0, min(px2, phx2) - max(px1, phx1))
            y_overlap = max(0, min(py2, phy2) - max(py1, phy1))
            
            intersection_area = x_overlap * y_overlap
            
            if intersection_area > 0:
                # Calcul de l'IoU
                person_area = (px2 - px1) * (py2 - py1)
                phone_area = (phx2 - phx1) * (phy2 - phy1)
                union_area = person_area + phone_area - intersection_area
                
                iou = intersection_area / union_area
                
                if iou > threshold:
                    return True
            
            # Distance minimale entre les boîtes
            if px2 < phx1:
                dist_x = phx1 - px2
            elif phx2 < px1:
                dist_x = px1 - phx2
            else:
                dist_x = 0
            
            if py2 < phy1:
                dist_y = phy1 - py2
            elif phy2 < py1:
                dist_y = py1 - phy2
            else:
                dist_y = 0
            
            if dist_x < 70 and dist_y < 70:
                return True
                
            # Détection par zone de vision
            head_center_x = (px1 + px2) // 2
            head_center_y = py1 + (py2 - py1) * 0.3
            
            phone_center_x = (phx1 + phx2) // 2
            phone_center_y = (phy1 + phy2) // 2
            
            vision_dist = ((phone_center_x - head_center_x) ** 2 + (phone_center_y - head_center_y) ** 2) ** 0.5
            if vision_dist < 100:
                return True
        
        return False
    
    def merge_students_with_same_names(self, students_data):
        """Fusionne les étudiants portant le même nom personnalisé"""
        # Grouper par nom
        name_groups = {}
        for student in students_data:
            student_name = student.get('name', student.get('id', ''))
            if student_name not in name_groups:
                name_groups[student_name] = []
            name_groups[student_name].append(student)
        
        # Fusionner
        merged_students = []
        for name, group_students in name_groups.items():
            if len(group_students) > 1:
                merged_student = {
                    'id': f"{name} (fusionné {len(group_students)} IDs)",
                    'name': name,
                    'phone_time': max(s['phone_time'] for s in group_students),
                    'timer': max(s['timer'] for s in group_students),
                    'category': max(group_students, key=lambda s: s['phone_time'])['category'],
                    'has_phone_now': any(s['has_phone_now'] for s in group_students),
                    'is_speaking': any(s['is_speaking'] for s in group_students),
                    'is_conversation': any(s['is_conversation'] for s in group_students),
                    'speaking_time': max(s['speaking_time'] for s in group_students),
                    'long_speaking_alert': any(s['long_speaking_alert'] for s in group_students),
                    'speaking_status': max(group_students, key=lambda s: s['speaking_time'])['speaking_status'],
                    'is_proximity': any(s.get('is_proximity', False) for s in group_students),
                    'proximity_time': max(s.get('proximity_time', 0) for s in group_students),
                    'top_color': max(group_students, key=lambda s: s.get('top_color', ''))['top_color'],
                    'top_color_rgb': max(group_students, key=lambda s: s.get('top_color', ''))['top_color_rgb']
                }
                merged_students.append(merged_student)
                print(f"   🔄 FUSION: {len(group_students)} IDs avec nom '{name}' → 1 étudiant")
            else:
                merged_students.append(group_students[0])
        
        return merged_students
    
    def get_stats(self):
        stats = {
            'total_students': int(len(self.active_face_ids)),
            'students_with_phones': 0,
            'students_speaking': 0,
            'students_in_conversation': 0,
            'attention_breakdown': {
                'green': 0, 'yellow': 0, 'orange': 0, 'red': 0
            },
            'green_count': 0,
            'yellow_count': 0,
            'orange_count': 0,
            'red_count': 0,
            'speaking_count': 0,
            'students_details': [],
            'students': [],
            'all_students_ever': sorted([x for x in self.all_detected_ids if not x.startswith("Track_")])
        }
        
        for face_id in self.active_face_ids:
            if face_id.startswith("Track_"):
                continue
            
            # Récupérer le nom de l'étudiant
            student_name = self.face_recognizer.known_faces.get(face_id, {}).get('name', face_id)
            
            phone_time = float(self.face_id_to_timer.get(face_id, 0))
            is_distracted = bool(self.face_id_to_state.get(face_id, False))
            
            # Informations de parole
            is_speaking = bool(self.face_id_to_speaking.get(face_id, False))
            is_conversation = bool(self.face_id_to_conversation.get(face_id, False))
            speaking_time = float(self.face_id_to_speaking_timer.get(face_id, 0))
            
            if is_speaking:
                stats['students_speaking'] += 1
                stats['speaking_count'] += 1
            
            if is_conversation:
                stats['students_in_conversation'] = stats.get('students_in_conversation', 0) + 1
            
            if phone_time == 0:
                category = 'green'
            elif phone_time < 10:
                category = 'yellow'
            elif phone_time < 30:
                category = 'orange'
            else:
                category = 'red'
            
            if category == 'green':
                stats['attention_breakdown']['green'] = int(stats['attention_breakdown']['green'] + 1)
                stats['green_count'] = int(stats['green_count'] + 1)
            elif category == 'yellow':
                stats['attention_breakdown']['yellow'] = int(stats['attention_breakdown']['yellow'] + 1)
                stats['yellow_count'] = int(stats['yellow_count'] + 1)
            elif category == 'orange':
                stats['attention_breakdown']['orange'] = int(stats['attention_breakdown']['orange'] + 1)
                stats['orange_count'] = int(stats['orange_count'] + 1)
            elif category == 'red':
                stats['attention_breakdown']['red'] = int(stats['attention_breakdown']['red'] + 1)
                stats['red_count'] = int(stats['red_count'] + 1)
                stats['students_with_phones'] = int(stats['students_with_phones'] + 1)
            
            # Déterminer le statut de parole
            is_long_speaking = speaking_time > 5.0 and (is_speaking or is_conversation)
            
            student_data = {
                'id': str(face_id),
                'name': str(student_name),
                'phone_time': float(round(phone_time, 1)),
                'timer': float(round(phone_time, 1)),
                'category': str(category),
                'has_phone_now': bool(is_distracted),
                'is_speaking': bool(is_speaking),
                'is_conversation': bool(is_conversation),
                'speaking_time': float(round(speaking_time, 1)),
                'long_speaking_alert': bool(is_long_speaking),
                'speaking_status': 'conversation' if is_conversation else ('parle' if is_speaking else 'silencieux'),
                'is_proximity': bool(self.face_id_to_proximity_detected.get(face_id, False)),
                'proximity_time': float(round(self.face_id_to_proximity_timer.get(face_id, 0), 1)),
                'top_color': self.face_id_to_top_color.get(face_id, {}).get('color', 'Non détecté'),
                'top_color_rgb': self.face_id_to_top_color.get(face_id, {}).get('rgb', [0, 0, 0])
            }
            stats['students_details'].append(student_data)
            stats['students'].append(student_data)
        
        # Fusionner les étudiants avec le même nom
        merged_students = self.merge_students_with_same_names(stats['students_details'])
        stats['students_details'] = merged_students
        stats['students'] = merged_students
        
        # Recalculer les statistiques après fusion
        stats['total_students'] = len(merged_students)
        stats['students_with_phones'] = sum(1 for s in merged_students if s['has_phone_now'])
        stats['students_speaking'] = sum(1 for s in merged_students if s['is_speaking'])
        stats['speaking_count'] = stats['students_speaking']
        
        # Recalculer l'attention breakdown
        stats['attention_breakdown'] = {'green': 0, 'yellow': 0, 'orange': 0, 'red': 0}
        stats['green_count'] = stats['yellow_count'] = stats['orange_count'] = stats['red_count'] = 0
        
        for student in merged_students:
            category = student['category']
            if category in stats['attention_breakdown']:
                stats['attention_breakdown'][category] += 1
                
                if category == 'green':
                    stats['green_count'] += 1
                elif category == 'yellow':
                    stats['yellow_count'] += 1
                elif category == 'orange':
                    stats['orange_count'] += 1
                elif category == 'red':
                    stats['red_count'] += 1
        
        stats['students_details'].sort(key=lambda x: str(x['name']))
        stats['students'].sort(key=lambda x: str(x['name']))
        
        # S'assurer que tous les types sont JSON serializable
        for key, value in stats.items():
            if isinstance(value, (np.bool_, np.integer, np.floating)):
                stats[key] = value.item()
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (np.bool_, np.integer, np.floating)):
                        stats[key][subkey] = subvalue.item()
            elif isinstance(value, list):
                stats[key] = [
                    item.item() if isinstance(item, (np.bool_, np.integer, np.floating)) else item 
                    for item in value
                ]
        
        return stats

# ========================================
# FLASK APP
# ========================================

app = Flask(__name__)
tracker = ContinuousFaceTracker(max_disappeared=30, decay_rate=0.5)
current_stats = {
    'total_students': 0,
    'students_with_phones': 0,
    'attention_breakdown': {'green': 0, 'yellow': 0, 'orange': 0, 'red': 0},
    'students_details': [],
    'all_students_ever': []
}

# Sauvegarde auto
import threading

def auto_save():
    while True:
        time.sleep(30)
        save_session_data(tracker)
        save_students_db(tracker.face_recognizer.known_faces)

save_thread = threading.Thread(target=auto_save, daemon=True)
save_thread.start()

global frame_count_for_blink

frame_count_for_blink = frame_count_for_blink + 1 if 'frame_count_for_blink' in globals() else 0

def get_alert_color(phone_time, is_speaking=False, is_conversation=False, proximity_detected=False):
    # Rouge clignotant intense pour conversation ou proximité
    if is_conversation or proximity_detected:
        if frame_count_for_blink % 20 < 10:
            return (255, 0, 0)
        else:
            return (0, 0, 255)
    elif phone_time == 0:
        return (0, 255, 0)
    elif phone_time < 10:
        return (0, 255, 255)
    elif phone_time < 30:
        return (0, 165, 255)
    else:
        return (0, 0, 255)

def process_frame(frame):
    global current_stats
    
    # YOLO tracking
    results = model.track(frame, persist=True, verbose=False, classes=[PERSON_CLASS, PHONE_CLASS], conf=0.45)
    
    # Pose
    pose_data = {}
    if pose_enabled:
        pose_results = pose_model.track(frame, persist=True, verbose=False)
        if pose_results and pose_results[0].keypoints is not None:
            for i, box in enumerate(pose_results[0].boxes):
                if box.id is not None:
                    track_id = int(box.id[0])
                    keypoints = pose_results[0].keypoints.data[i].cpu().numpy()
                    pose_data[track_id] = keypoints
    
    if results[0].boxes is None or len(results[0].boxes) == 0:
        timestamp = time.time()
        current_stats = tracker.update(frame, [], [], pose_data, timestamp)
        return frame
    
    result = results[0]
    
    # Extraire personnes et téléphones
    persons_with_ids = []
    phone_boxes = []
    
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        if conf > 0.35:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls == PERSON_CLASS and box.id is not None:
                track_id = int(box.id[0])
                persons_with_ids.append((track_id, (x1, y1, x2, y2)))
            elif cls == PHONE_CLASS:
                phone_boxes.append((x1, y1, x2, y2))
    
    # Mise à jour
    timestamp = time.time()
    current_stats = tracker.update(frame, persons_with_ids, phone_boxes, pose_data, timestamp)
    
    # Annotation
    annotated_frame = frame.copy()
    
    # Téléphones
    for phone_box in phone_boxes:
        x1, y1, x2, y2 = phone_box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_frame, "Phone", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Personnes avec face_id
    for track_id, person_box in persons_with_ids:
        x1, y1, x2, y2 = person_box
        face_id = tracker.track_to_face.get(track_id, f"Track_{track_id}")
        
        if face_id.startswith("Track_"):
            continue
        
        phone_time = tracker.face_id_to_timer.get(face_id, 0)
        is_distracted = tracker.face_id_to_state.get(face_id, False)
        
        last_history = tracker.history[face_id][-1] if tracker.history[face_id] else {}
        has_phone = last_history.get('has_phone', False)
        looking_down = last_history.get('looking_down', False)
        is_speaking = last_history.get('is_speaking', False)
        
        # Récupérer les nouvelles détections
        is_conversation = last_history.get('is_conversation', False)
        proximity_detected = tracker.face_id_to_proximity_detected.get(face_id, False)
        
        # Couleur intelligente
        color = get_alert_color(phone_time, is_speaking, is_conversation, proximity_detected)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        
        # Label
        label = face_id
        if phone_time > 0:
            label += f" ({phone_time:.1f}s)"
        
        # Ajouter les indicateurs visuels
        if is_speaking:
            label += " 🗣️"
            center_x = x2 - 20
            center_y = y1 + 20
            cv2.circle(annotated_frame, (center_x, center_y), 8, (255, 165, 0), -1)
            cv2.circle(annotated_frame, (center_x, center_y), 8, (255, 255, 255), 2)
        
        # Indicateur de proximité
        if proximity_detected:
            triangle_points = np.array([
                [x1 + 30, y1 - 10],
                [x1 + 40, y1 - 30],
                [x1 + 50, y1 - 10]
            ], np.int32)
            cv2.fillPoly(annotated_frame, [triangle_points], (255, 0, 0))
            cv2.putText(annotated_frame, "!", (x1 + 35, y1 - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if is_distracted:
            if has_phone:
                label += " 📱"
            elif looking_down:
                label += " ⬇️"
        else:
            if not is_speaking:
                label += " ✓"
        
        # Background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame

def generate_frames():
    camera = cv2.VideoCapture(selected_camera_index)
    # Configuration caméra
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FPS, 60)
    
    if not camera.isOpened():
        print(f"❌ Impossible d'ouvrir caméra {selected_camera_index}")
        return
    
    print(f"✅ Caméra {selected_camera_index} ouverte - Vérification continue activée")
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            processed_frame = process_frame(frame)
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        camera.release()
        save_session_data(tracker)
        save_students_db(tracker.face_recognizer.known_faces)
        print("📹 Caméra fermée, données sauvegardées")

@app.route('/')
def index():
    return render_template('index_modern.html', camera_info={
        'index': selected_camera_index,
        'cameras': available_cameras
    })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    return jsonify(current_stats)

@app.route('/set_student_name', methods=['POST'])
def set_student_name():
    data = request.get_json()
    face_id = data.get('face_id')
    new_name = data.get('name')
    
    if not face_id or not new_name:
        return jsonify({'success': False, 'message': 'ID ou nom manquant'}), 400
    
    if face_id in tracker.face_recognizer.known_faces:
        with tracker.face_recognizer.db_lock:
            tracker.face_recognizer.known_faces[face_id]['name'] = new_name
            for student in current_stats.get('students_details', []):
                if student.get('id') == face_id:
                    student['name'] = new_name
                    break
        save_students_db(tracker.face_recognizer.known_faces)
        return jsonify({'success': True, 'message': f'Nom de {face_id} mis à jour à {new_name}'})
    else:
        return jsonify({'success': False, 'message': f'ID étudiant {face_id} non trouvé'}), 404

@app.route('/screenshots')
def list_screenshots():
    """Liste toutes les captures d'écran organisées par catégorie"""
    screenshots_by_category = {}
    
    for category in SCREENSHOT_CATEGORIES:
        category_dir = os.path.join(SCREENSHOTS_DIR, category)
        if os.path.exists(category_dir):
            screenshots = []
            for filename in os.listdir(category_dir):
                if filename.endswith('.jpg'):
                    # Lire les métadonnées
                    metadata_file = filename.replace('.jpg', '_metadata.json')
                    metadata_path = os.path.join(category_dir, metadata_file)
                    
                    metadata = {}
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                        except:
                            pass
                    
                    # Info sur le fichier
                    file_path = os.path.join(category_dir, filename)
                    file_stat = os.stat(file_path)
                    
                    screenshots.append({
                        'filename': filename,
                        'category': category,
                        'size': file_stat.st_size,
                        'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        'metadata': metadata
                    })
            
            # Trier par date
            screenshots.sort(key=lambda x: x['created'], reverse=True)
            screenshots_by_category[category] = screenshots
    
    return jsonify({
        'success': True,
        'screenshots': screenshots_by_category,
        'total_categories': len(screenshots_by_category)
    })

@app.route('/screenshots/<category>/<filename>')
def get_screenshot(category, filename):
    """Serve une capture d'écran spécifique"""
    file_path = os.path.join(SCREENSHOTS_DIR, category, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    else:
        return "Capture non trouvée", 404

@app.route('/reset', methods=['POST'])
def reset_stats():
    global tracker
    tracker = ContinuousFaceTracker(max_disappeared=25, decay_rate=0.6)
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    if os.path.exists(STUDENTS_DB_FILE):
        os.remove(STUDENTS_DB_FILE)
    return jsonify({'success': True, 'message': 'Statistiques et base visages réinitialisées'})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if not openai_enabled:
        return jsonify({'success': False, 'error': 'API LLM non disponible', 'report': '❌ API LLM non disponible. Vérifiez les clés API dans .env'})
    
    try:
        details = "\n".join([
            f"  - {s['name']}: {s['phone_time']}s ({s['category']}) {'[DISTRAIT]' if s['has_phone_now'] else '[ATTENTIF]'} {'[PARLE]' if s['is_speaking'] else ''}"
            for s in current_stats.get('students_details', [])
        ])
        
        prompt = f"""
Analyse détaillée de l'attention en classe (reconnaissance faciale continue):

📊 STATISTIQUES GLOBALES:
- Total étudiants détectés: {current_stats['total_students']}
- Étudiants attentifs (vert): {current_stats['green_count']}
- Légère distraction (jaune): {current_stats['yellow_count']}
- Distraction moyenne (orange): {current_stats['orange_count']}
- Distraction sévère (rouge): {current_stats['red_count']}
- Étudiants en train de parler: {current_stats['speaking_count']}

👥 DÉTAILS PAR ÉTUDIANT:
{details if details else "  Aucun étudiant détecté"}

📝 Génère un rapport complet et professionnel avec:
1. Résumé global de l'attention de la classe
2. Analyse des interactions et conversations en cours
3. Identification des étudiants nécessitant une attention particulière
4. Recommandations pédagogiques concrètes
5. Analyse des tendances observées

Format: Markdown avec emojis.
"""
        
        response = client.generate(prompt, max_tokens=800, temperature=0.7)
        return jsonify({'success': True, 'report': response})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'report': f'❌ Erreur: {str(e)}'})

@app.route('/chat', methods=['POST'])
def chat():
    if not openai_enabled:
        return jsonify({'response': '❌ Désolé, API LLM n\'est pas disponible. Vérifiez les clés API dans .env'})
    
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'response': '⚠️ Veuillez poser une question.'})
        
        # Préparer le contexte
        students_details = current_stats.get('students_details', [])
        details_text = "\n".join([
            f"  - {s['name']}: {s['phone_time']}s de distraction ({s['category']}) - Statut actuel: {'📱 DISTRAIT' if s['has_phone_now'] else '✅ ATTENTIF'} {'🗣️ PARLE' if s['is_speaking'] else ''}"
            for s in students_details
        ])
        
        context = f"""
Tu es un assistant pédagogique AI intégré dans un système de monitoring d'attention en classe.

📊 DONNÉES ACTUELLES DE LA CLASSE:
- Total étudiants: {current_stats['total_students']}
- Attentifs (0s): {current_stats['green_count']}
- Légère distraction (<10s): {current_stats['yellow_count']}
- Distraction moyenne (10-30s): {current_stats['orange_count']}
- Distraction sévère (>30s): {current_stats['red_count']}
- Étudiants parlant actuellement: {current_stats['speaking_count']}

👥 DÉTAILS DES ÉTUDIANTS:
{details_text if details_text else "  Aucun étudiant détecté pour le moment"}

QUESTION DE L'UTILISATEUR: {user_message}

Réponds de manière concise, précise et professionnelle.
"""
        
        response = client.generate(context, max_tokens=600, temperature=0.7)
        
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'response': f'❌ Désolé, une erreur est survenue: {str(e)}'})

if __name__ == '__main__':
    print("✅ Système prêt - Reconnaissance Faciale CONTINUE")
    print(f"\n🌐 SERVEUR: http://localhost:8080")
    print(f"📹 CAMÉRA: Index {selected_camera_index}")
    
    # Info LLM
    if openai_enabled:
        print(f"\n🤖 IA ASSISTANT: ACTIF")
        print(f"   🔑 API configurée depuis .env")
        print(f"   💡 Modèle: {openai_model}")
    else:
        print(f"\n🤖 IA ASSISTANT: DÉSACTIVÉ")
        print(f"   📝 Pour activer, configurez .env avec les clés API")
    
    print(f"\n✨ FONCTIONNALITÉS ACTIVES:")
    print(f"   - 👤 Vérification visage à CHAQUE frame")
    print(f"   - 🔄 Pas de nouveaux IDs quand sortie/retour")
    print(f"   - 🗣️ Détection de conversation")
    print(f"   - 📊 Suivi du temps de parole par étudiant")
    print(f"\n📁 Fichiers:")
    print(f"   - {SESSION_FILE}")
    print(f"   - {STUDENTS_DB_FILE}\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)

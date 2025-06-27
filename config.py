"""
Configuration file for Joel's Angels application
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application Settings
APP_NAME = "Joel's Angels"
APP_VERSION = "1.0.0"
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# API Keys (for future AI integrations)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY", "")

# Security Settings
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key-change-in-production")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "default-encryption-key")

# File Storage Settings
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default

# Supported file types
SUPPORTED_FILE_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain"
}

# Contract categories
CONTRACT_CATEGORIES = {
    "🏠 Real Estate": "real_estate",
    "🤝 Partnership": "partnership", 
    "💼 Employment": "employment",
    "💰 Investment": "investment",
    "📋 Service Agreement": "service",
    "🏢 Business": "business",
    "📄 NDA": "nda",
    "🏗️ Construction": "construction"
}

# Supported languages
SUPPORTED_LANGUAGES = {
    "🇺🇸 English": "en",
    "🇪🇸 Spanish": "es",
    "🇫🇷 French": "fr",
    "🇩🇪 German": "de",
    "🇨🇳 Chinese": "zh",
    "🇯🇵 Japanese": "ja",
    "🇮🇹 Italian": "it",
    "🇵🇹 Portuguese": "pt"
}

# Jurisdictions
JURISDICTIONS = [
    "United States",
    "United Kingdom", 
    "Canada",
    "Australia",
    "European Union",
    "India",
    "Singapore",
    "Other"
]

# Governing Laws
GOVERNING_LAWS = [
    "Common Law",
    "Civil Law", 
    "Mixed Jurisdiction",
    "Islamic Law",
    "Customary Law"
]

# Risk levels
RISK_LEVELS = {
    "low": {"color": "#d4edda", "text_color": "#155724", "label": "Low Risk"},
    "medium": {"color": "#fff3cd", "text_color": "#856404", "label": "Medium Risk"},
    "high": {"color": "#f8d7da", "text_color": "#721c24", "label": "High Risk"}
}

# UI Colors
UI_COLORS = {
    "primary": "#0A192F",      # Navy Blue
    "secondary": "#FFD700",    # Gold
    "accent": "#20B2AA",       # Teal
    "light_gray": "#F8F9FA",   # Light Gray
    "white": "#FFFFFF",        # White
    "charcoal": "#333333"      # Charcoal
} 
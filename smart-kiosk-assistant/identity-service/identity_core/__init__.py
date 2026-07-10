"""Smart Kiosk Identity Service.

Standalone microservice providing multimodal biometric authentication
(Face ID + Voiceprint) using Intel OpenVINO for feature extraction, FAISS for
similarity search, and SQLite for loyalty-profile persistence.
"""

__version__ = "0.1.0"

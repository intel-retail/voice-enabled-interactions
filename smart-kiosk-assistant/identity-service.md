4. Kiosk User Identification (Biometric Face ID)
Use Case
The modern ATMs are coming up with chat functionality and for users to be authenticated, we would like to utilise both the voice and camera combined as the means for authenticating the user.
the requirement is to match the user's face against facial features stored and the spoken voice against voice signature stored
For the voice recognition, the user is meant to repeat a fixed text that appears on the screen - Something like "My voice is my password"


Identity Service
It provides multimodal biometric authentication (Face ID and Voiceprint verification) by leveraging Intel OpenVINO for feature extraction, FAISS for fast similarity index matching, and SQLite for loyalty profile persistence.
0.1 Startup Bootstrapping Flow (Automatic Test Registration)
When the Docker container starts up:
1.	Bootstrap Check: The service checks the BOOTSTRAP_ON_START environment variable. If set to false, it bypasses the bootstrap sequence entirely.
2.	Configuration Loading: If enabled, the service reads the identity_config.yaml using PyYAML to parse the list of loyalty profiles to register.
3.	Database Pre-Existence Query: For each user record, the service queries SQLite (SELECT 1 FROM loyalty_profiles WHERE user_id = ?). If a record is found, it skips registration for that user to avoid duplicate indices across restarts.
4.	Face Extraction (video_path): If a video_path is specified, OpenCV opens the video file. It samples frames (every 10 frames) and passes them to the face detection model. Once a high-confidence face is found, landmark alignment is performed, a 256-d face embedding is extracted, and it is added to the Face FAISS index (face_index.bin), returning a face_faiss_id.
5.	Voice Extraction (audio_path): If an audio_path is specified, the service reads the WAV audio file bytes directly. It extracts log-mel filterbanks, computes the 192-d speaker embedding using the ECAPA-TDNN OpenVINO pipeline, and adds it to the Voice FAISS index (voice_index.bin), returning a voice_faiss_id.
6.	SQLite Profile Insertion: The profile details (User ID, name, favorites, restrictions) along with the retrieved index offsets (face_faiss_id, voice_faiss_id) are written to the SQLite loyalty_profiles table.
________________________________________
0.2 Client Verification Flow (Runtime Login)
When a customer approaches the kiosk and interacts with the system:
1.	API Invocation: The client UI captures camera frames and/or microphone audio chunks, converting them to Base64, and calls POST /api/v1/identity/verify via the orchestrator.
2.	Input Demuxing: The API endpoint parses the request to determine which biometric indicators are present.
3.	Biometric Inference (OpenVINO):
•	If Image Present: Runs face detection, rotates/aligns the crop via facial landmarks, and computes the normalized 256-d face vector.
•	If Audio Present: Ingests the PCM WAV buffer, resamples to 16kHz, computes spectrogram features, and extracts the 192-d speaker identity vector.
4.	Index Search (FAISS):
•	The face vector is searched against the face FAISS index using cosine similarity (Inner Product), returning the nearest index match and distance 
•	The voice vector is searched against the voice FAISS index, returning the nearest index match and distance ($Sim_{voice}$).
5.	Weighted Scoring Fusion: If both biometrics are present, the fusion engine computes a unified similarity score ($Score = 0.6 \cdot Sim_{face} + 0.4 \cdot Sim_{voice}$) and checks if it satisfies the combined verification threshold. If only one biometric is present, it compares the single similarity score directly against its respective threshold (Face: 0.80, Voice: 0.75).
6.	Profile Retrieval & Injection: If the threshold is satisfied, the service looks up the profile metadata in SQLite using the matching FAISS offset, and returns the profile JSON payload. The orchestrator then injects the customer’s favorites and restrictions directly into the prompt context for the LLM session.

rtspsrc location=rtsp://camera_ip/stream ! rtph264depay ! h264parse ! vaapidecodebin ! \
gvadetect model=models/face-detection-retail-0005.xml device=GPU ! \
gvaclassify model=models/face-reidentification-retail-0095.xml device=GPU ! \
appsink name=video_sink

rtspsrc location=rtsp://camera_ip/stream ! rtpmp4gdepay ! aacparse ! decodebin ! \
audioconvert ! audioresample ! audio/x-raw,rate=16000,channels=1 ! \
gvaclassify model=models/ecapa-tdnn-voice.xml device=CPU ! \
appsink name=audio_sink


users 
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    face_faiss_id INTEGER,
    voice_faiss_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP



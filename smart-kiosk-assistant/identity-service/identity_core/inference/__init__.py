"""Inference layer: OpenVINO face + voice embedding engines.

Phase 4 (implemented):
  * ``base.py``         — Strategy ABCs (``FaceEmbeddingEngine`` /
                          ``VoiceEmbeddingEngine``) returning L2-normalized
                          ``float32`` vectors.
  * ``openvino_face.py``  — face-detection-retail-0005 + face-reidentification-
                          retail-0095 → 256-d face embedding.
  * ``openvino_voice.py`` — converted ECAPA-TDNN IR → 192-d speaker embedding.
  * ``factory.py``        — builds engines from ``Settings`` (Factory), validates
                          model files, degrades gracefully to ``None`` when a
                          model is missing so the service still serves
                          /health, /challenge and /stats.

Models are fetched/converted by ``setup_models.sh --identity`` into
``models/identity/`` (mounted at ``/app/models`` via docker-compose).
"""

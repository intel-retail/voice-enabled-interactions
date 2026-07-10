"""Identity (biometric authentication) orchestration package for kiosk-core.

This package is the **client/orchestration** side of the identity feature.  The
heavy biometric inference (OpenVINO face + voice, FAISS search, bootstrap
registration) lives in the standalone ``identity-service`` microservice.

kiosk-core only:
  * exposes a thin REST facade (``api.py``) guarded by ``IDENTITY_ENABLED``,
  * proxies challenge/verify calls to identity-service via ``IdentityClient``,
  * injects the matched loyalty profile (favorites / restrictions) into the
    LLM session context.
"""

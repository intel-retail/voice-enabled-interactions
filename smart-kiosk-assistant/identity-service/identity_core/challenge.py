"""Random voice-challenge prompt provider.

Picks one prompt at random per attempt for anti-replay liveness.  Verification
itself is text-independent (ECAPA voiceprint), so the prompt content does not
affect the biometric match.
"""

from __future__ import annotations

import logging
import secrets
import uuid

logger = logging.getLogger(__name__)

# Fallback prompt if the YAML list is empty or missing.
_FALLBACK_PROMPT = "My voice is my password."


class ChallengeProvider:
    """Serves random challenge prompts from the configured list."""

    def __init__(self, prompts: list[str]):
        self._prompts = [p.strip() for p in prompts if p and p.strip()]
        if not self._prompts:
            logger.warning("[CHALLENGE] No prompts configured — using fallback prompt")
            self._prompts = [_FALLBACK_PROMPT]
        logger.info("[CHALLENGE] Loaded %d challenge prompt(s)", len(self._prompts))

    def next_challenge(self) -> tuple[str, str]:
        """Return ``(challenge_id, prompt_text)`` with a cryptographically
        random prompt selection and a unique challenge id."""
        prompt = secrets.choice(self._prompts)
        challenge_id = uuid.uuid4().hex
        return challenge_id, prompt

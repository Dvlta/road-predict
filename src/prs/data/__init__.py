"""Data layer: raw WOMD reading, target-centric samples, caching."""

# Bump whenever preprocessing logic changes; it is part of the cache fingerprint (design doc §11).
PREPROCESS_VERSION = 1

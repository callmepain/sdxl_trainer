# Multi-teacher distillation training for SDXL
#
# This module provides tools for training a student SDXL model using
# pre-computed predictions from multiple teacher models.
#
# Main components:
# - build_teacher_cache.py: Build teacher prediction cache
# - train_distill.py: Train student using distillation
# - verify_cache.py: Verify cache integrity

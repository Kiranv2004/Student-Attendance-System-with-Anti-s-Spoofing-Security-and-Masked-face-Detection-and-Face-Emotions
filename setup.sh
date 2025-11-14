#!/bin/bash
# Setup script for Railway deployment

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install cmake first (required for dlib)
pip install cmake

# Install dlib (this may take a while)
pip install dlib==19.24.2

# Install remaining requirements
pip install -r requirements.txt


# File konfigurasi dasar pytest
import sys
import os

# Tambahkan direktori utama project agar pytest bisa membaca folder src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

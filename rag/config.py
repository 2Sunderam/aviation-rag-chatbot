"""Project paths and shared settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"

# Display names used in citations ("Pilot's Handbook of Aeronautical Knowledge, p. 42")
TITLES = {
    "01_aviation_instructors_handbook": "Aviation Instructor's Handbook",
    "02_glider_flying_handbook": "Glider Flying Handbook",
    "03_weight_shift_control_handbook": "Weight-Shift Control Aircraft Flying Handbook",
    "04_instrument_procedures_handbook": "Instrument Procedures Handbook",
    "05_instrument_flying_handbook": "Instrument Flying Handbook",
    "06_helicopter_flying_handbook": "Helicopter Flying Handbook",
    "07_rotorcraft_flying_handbook": "Rotorcraft Flying Handbook",
    "08_pilots_handbook_aeronautical_knowledge": "Pilot's Handbook of Aeronautical Knowledge",
    "09_aerodynamics_for_naval_aviators_1965": "Aerodynamics for Naval Aviators (1965)",
    "10_mc4_parachute_technical_manual": "MC-4 Ram Air Parachute Technical Manual",
    "11_aviation_weather_handbook": "Aviation Weather Handbook",
}

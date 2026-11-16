"""Shared isometric frame for Pixel Coimbatore.

True isometric: camera faces north-east (heading 45 deg), tilted 35.264 deg below the horizon,
orthographic. A fixed global origin, so every tile of the city lands on one big canvas.
S = pixels per metre along the ground's (e-n) axis scale factor (see PX_PER_CAM_M).
"""
import math

LAT0, LON0 = 11.0, 76.96
MX = 111320 * math.cos(math.radians(LAT0)); MY = 110540
C30, S30 = math.cos(math.radians(30)), math.sin(math.radians(30))
HEADING, TILT = 45.0, math.degrees(math.atan(1 / math.sqrt(2)))  # 35.264


def en(lon, lat):
    return (lon - LON0) * MX, (lat - LAT0) * MY


def lonlat(e, n):
    return e / MX + LON0, n / MY + LAT0


def iso(e, n, h, S):
    return (e - n) * C30 * S, -(e + n) * S30 * S - h * S


def px_per_cam_m(S):
    # screen px per metre of the orthographic camera's view width
    return C30 * S * math.sqrt(2)


def ground_at(px, py, S):
    """Screen pixel (global iso canvas) -> ground (e, n) at height 0."""
    a = px / (C30 * S); b = -py / (S30 * S)
    return (a + b) / 2, (b - a) / 2

"""
AVATARS fusion pipeline: combine visual whale detections with tag AOA and dive model.

Implements the sensing component extension for Professor Stephanie Gil's
AVATARS (Autonomous Vehicles for whAle Tracking And Rendezvous by remote Sensing)
framework at Harvard SEAS.

Reference: https://seas.harvard.edu/news/new-methods-whale-tracking-and-rendezvous-using-autonomous-robots
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WhaleDetection:
    bbox: list[float]          # [x1, y1, x2, y2] pixels
    confidence: float
    class_name: str
    range_estimate_m: Optional[float] = None
    bearing_deg: Optional[float] = None  # relative to camera optical axis


@dataclass
class TagAOA:
    """Angle-of-arrival from CETI on-whale VHF tag."""
    azimuth_deg: float
    elevation_deg: float = 0.0
    signal_strength_dbm: float = -80.0
    tag_id: str = ""


@dataclass
class DiveState:
    """Sperm whale dive behavior model state."""
    phase: str = "unknown"     # surface, shallow, deep_foraging, ascent
    time_since_surface_s: float = 0.0
    predicted_surface_time_s: Optional[float] = None
    predicted_surface_lat: Optional[float] = None
    predicted_surface_lon: Optional[float] = None


@dataclass
class RendezvousTarget:
    """Fused target for AVATARS autonomy planner."""
    lat: float
    lon: float
    eta_s: float
    confidence: float
    source: str                # 'visual', 'aoa', 'dive_model', 'fused'
    whale_id: Optional[str] = None


@dataclass
class AVATARSPipeline:
    """
    Multi-sensor fusion for whale rendezvous planning.

    Inputs:
      - Visual detections from CETI whale detector + depth
      - VHF tag AOA from aerial drone antenna array
      - Sperm whale dive model predictions

    Output:
      - Ranked rendezvous targets for drone path planning
    """

    camera_focal_px: float = 800.0
    camera_hfov_deg: float = 60.0
    min_detection_confidence: float = 0.6
    rendezvous_depth_max_m: float = 50.0

    def visual_bearing(self, detection: WhaleDetection, image_width: int) -> float:
        """Compute bearing from bbox center relative to camera center."""
        cx = (detection.bbox[0] + detection.bbox[2]) / 2
        center_offset = (cx - image_width / 2) / (image_width / 2)
        return center_offset * (self.camera_hfov_deg / 2)

    def fuse(
        self,
        visual_detections: list[WhaleDetection],
        tag_aoa: Optional[TagAOA] = None,
        dive_state: Optional[DiveState] = None,
        drone_lat: float = 0.0,
        drone_lon: float = 0.0,
        drone_heading_deg: float = 0.0,
        image_width: int = 1920,
    ) -> list[RendezvousTarget]:
        """
        Fuse multi-modal observations into ranked rendezvous targets.

        Priority:
          1. Visual detection with high confidence + range estimate
          2. Tag AOA intersection with dive model surface prediction
          3. Dive model alone (predicted surfacing location)
        """
        targets: list[RendezvousTarget] = []

        # Visual detections
        for det in visual_detections:
            if det.confidence < self.min_detection_confidence:
                continue

            bearing = self.visual_bearing(det, image_width)
            absolute_bearing = drone_heading_deg + bearing

            if det.range_estimate_m and det.range_estimate_m <= self.rendezvous_depth_max_m:
                # Project target position from drone position + bearing + range
                lat_offset = (det.range_estimate_m * math.cos(math.radians(absolute_bearing))) / 111320
                lon_offset = (det.range_estimate_m * math.sin(math.radians(absolute_bearing))) / (111320 * math.cos(math.radians(drone_lat)))

                targets.append(RendezvousTarget(
                    lat=drone_lat + lat_offset,
                    lon=drone_lon + lon_offset,
                    eta_s=0.0,
                    confidence=det.confidence * 0.8,
                    source="visual",
                ))

        # Dive model surface prediction
        if dive_state and dive_state.predicted_surface_lat is not None:
            eta = dive_state.predicted_surface_time_s or 60.0
            targets.append(RendezvousTarget(
                lat=dive_state.predicted_surface_lat,
                lon=dive_state.predicted_surface_lon or drone_lon,
                eta_s=eta,
                confidence=0.6 if dive_state.phase == "ascent" else 0.4,
                source="dive_model",
            ))

        # Tag AOA (provides bearing even without visual contact)
        if tag_aoa and tag_aoa.signal_strength_dbm > -100:
            aoa_bearing = drone_heading_deg + tag_aoa.azimuth_deg
            # Without range, assume last known or default 500m
            default_range = 500.0
            lat_offset = (default_range * math.cos(math.radians(aoa_bearing))) / 111320
            lon_offset = (default_range * math.sin(math.radians(aoa_bearing))) / (111320 * math.cos(math.radians(drone_lat)))

            targets.append(RendezvousTarget(
                lat=drone_lat + lat_offset,
                lon=drone_lon + lon_offset,
                eta_s=30.0,
                confidence=0.5 + (tag_aoa.signal_strength_dbm + 100) / 200,
                source="aoa",
                whale_id=tag_aoa.tag_id,
            ))

        # Sort by confidence descending
        targets.sort(key=lambda t: t.confidence, reverse=True)
        return targets

    def plan_rendezvous_route(
        self,
        targets: list[RendezvousTarget],
        drone_lat: float,
        drone_lon: float,
        drone_speed_ms: float = 15.0,
    ) -> list[dict]:
        """
        Generate waypoint sequence for drone to intercept highest-confidence target.

        Returns list of waypoints: [{lat, lon, alt_m, eta_s}, ...]
        """
        if not targets:
            return []

        best = targets[0]
        distance_m = self._haversine(drone_lat, drone_lon, best.lat, best.lon)
        travel_time = distance_m / drone_speed_ms

        return [
            {"lat": drone_lat, "lon": drone_lon, "alt_m": 30, "eta_s": 0},
            {"lat": best.lat, "lon": best.lon, "alt_m": 30, "eta_s": travel_time},
        ]

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

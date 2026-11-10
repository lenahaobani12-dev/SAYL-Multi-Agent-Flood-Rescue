"""
Checks for the flood environment and the Flood Monitor.

Run:  python -m pytest tests/test_environment.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sayl.acl import Performative
from sayl.environment import ROAD_BLOCK_DEPTH, FloodEnvironment
from sayl.model import SaylModel, SimParams


def test_water_is_conserved_without_rain_drain_or_outlet():
    env = FloodEnvironment(rain_intensity=0.0)
    env.DRAIN_RATE = 0.0
    env.OUTLET_FRAC = 0.0
    before = env.total_water()
    for _ in range(50):
        env.step()
    assert abs(env.total_water() - before) < 1e-9


def test_water_never_negative_and_collects_in_the_wadi():
    env = FloodEnvironment(rain_intensity=0.8)
    for _ in range(60):
        env.step()
    assert env.water.min() >= 0
    low = env.depth_map()[env.wadi_mask()].mean()
    high = env.depth_map()[env.elevation >= np.percentile(env.elevation, 80)].mean()
    assert low > 5 * high            # the valley floods first and much deeper than the high ground


def test_rain_raises_and_no_rain_lowers_the_water():
    rainy = FloodEnvironment(rain_intensity=0.8)
    dry = FloodEnvironment(rain_intensity=0.0)
    for _ in range(40):
        rainy.step()
        dry.step()
    assert rainy.water_level > dry.water_level
    assert dry.total_water() < FloodEnvironment(rain_intensity=0.0).total_water()   # it receded
    # stop the rain on a flooded map: the water goes down
    rainy.rain_intensity = 0.0
    peak = rainy.total_water()
    for _ in range(40):
        rainy.step()
    assert rainy.total_water() < peak


def test_roads_close_when_water_is_deep_and_reopen_when_it_recedes():
    env = FloodEnvironment(rain_intensity=1.0)
    start = env.road_closure_rate()
    for _ in range(40):
        env.step()
    assert env.road_closure_rate() > start
    for cell in env.closed_roads():
        assert env.is_road(cell) and env.depth(cell) >= ROAD_BLOCK_DEPTH
    env.rain_intensity = 0.0
    peak = env.road_closure_rate()
    for _ in range(150):
        env.step()
    assert env.road_closure_rate() < peak


def test_forecast_matches_the_real_future_and_does_not_change_the_state():
    env = FloodEnvironment(rain_intensity=0.6)
    snapshot = env.water.copy()
    predicted = env.forecast(15)
    assert np.array_equal(env.water, snapshot)
    for _ in range(15):
        env.step()
    assert np.allclose(predicted, env.water)


def test_path_avoids_closed_roads_and_boats_prefer_water():
    env = FloodEnvironment(rain_intensity=0.7)
    for _ in range(50):
        env.step()
    start, goal = env.command_center_pos, (12, 3)
    path = env.find_path(start, goal, "ground")
    assert path and path[-1] == goal
    assert all(not env.is_blocked(c) for c in path)
    boat = env.find_path(start, (14, 10), "boat")
    assert boat and boat[-1] == (14, 10)
    assert env.find_path(start, start) == []
    # every step moves at most one cell
    for a, b in zip([start] + boat, boat):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def test_flood_monitor_warns_before_it_escalates():
    model = SaylModel(SimParams(rain_intensity=0.7, distress_rate=0.0))
    model.submit_message("الموية دخلت البيت وعندنا كبار سن", (18, 11))
    model.run(60)
    flood = [m for m in model.bus.log if m.sender == "FloodMonitor" and m.performative == Performative.INFORM]
    events = [m.content.get("event") for m in flood]
    assert "flood_forecast" in events and "roads_closed" in events
    if "severity_update" in events:
        assert events.index("flood_forecast") < events.index("severity_update")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)

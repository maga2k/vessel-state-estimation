import tempfile
from pathlib import Path

from vessel_state_estimation.config import Config, SimConfig, get_vessel, load_vessels

ROOT = Path(__file__).resolve().parents[1]
VESSELS = ROOT / "configs" / "vessels.yaml"


def test_default_yaml_loads():
    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    assert cfg.sim == SimConfig()
    assert cfg.vessel == Config().vessel
    assert cfg.env.current_speed > 0.0 and cfg.env.wind_speed > 0.0


def test_all_three_vessels_load():
    vessels = load_vessels(VESSELS)
    assert set(vessels) == {"merchant_ship", "sailboat", "small_boat"}
    assert get_vessel(Config().vessel, VESSELS).name == Config().vessel


def test_unknown_vessel_raises():
    try:
        get_vessel("submarine", VESSELS)
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_unknown_key_raises():
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.yaml"
        bad.write_text("sim:\n  dtt: 0.1\n")
        try:
            Config.from_yaml(bad)
        except TypeError:
            return
        raise AssertionError("expected TypeError")

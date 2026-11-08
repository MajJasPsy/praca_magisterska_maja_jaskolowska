"""Test regresyjny samej czesci glownej, bez treningu i bez PsychoPy."""
from pathlib import Path
import csv
import json
import tempfile
import yaml

from session_runner import run_session

ROOT = Path(__file__).resolve().parent
with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Regresyjny test samej czesci glownej, bez treningu.
config["training"]["enabled"] = False


class FakeScreens:
    left_key = "f"
    right_key = "j"

    def __init__(self, reward_timeout_trial=None, cue_timeout_trial=None, abort_trial=None):
        self.choice_number = 0
        self.trial = 0
        self.reward_timeout_trial = reward_timeout_trial
        self.cue_timeout_trial = cue_timeout_trial
        self.abort_trial = abort_trial
        self.messages = []

    def message(self, text, seconds=None, continue_required=False):
        self.messages.append(text)

    def choice(self, title, left, right, timeout):
        if "Decyzja 1" in title:
            self.trial += 1
            if self.trial == self.abort_trial:
                raise KeyboardInterrupt
            if self.trial == self.cue_timeout_trial:
                return None, None
            return "f", 0.35
        if self.trial == self.reward_timeout_trial:
            return None, None
        return "j", 0.55


def load_rows(folder):
    with (folder / "trials.csv").open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


with tempfile.TemporaryDirectory() as tmp:
    folder = run_session(config, FakeScreens(), "test_ok", output_dir=tmp)
    rows = load_rows(folder)
    assert len(rows) == config["experiment"]["n_trials"]
    assert all(row["trial_status"] == "completed" for row in rows)
    assert all(row["true_context"] in config["contexts"]["states"] for row in rows)
    assert all(row["reward_points"] != "" for row in rows)
    assert (folder / "config_snapshot.json").exists()
    assert (folder / "planned_trials.csv").exists()
    meta = json.loads((folder / "session_metadata.json").read_text(encoding="utf-8"))
    assert meta["n_planned_trials"] == len(rows)
    assert all(row["cue_signal_observed"] == "" for row in rows if row["cue_choice"] == "skip")
    assert all(row["cue_signal_observed"] for row in rows if row["cue_choice"] == "buy")
    assert all(abs(float(row["reward_points"]) - float(row["cue_cost_paid"]) - float(row["net_points"])) < 1e-9 for row in rows)
    assert all(float(row["cue_cost_paid"]) == (float(row["cue_cost_offered"]) if row["cue_choice"] == "buy" else 0.0) for row in rows)

    folder2 = run_session(config, FakeScreens(reward_timeout_trial=4,
                                              cue_timeout_trial=2), "test_timeout", output_dir=tmp)
    rows2 = load_rows(folder2)
    assert rows2[1]["trial_status"] == "cue_timeout"
    assert rows2[3]["trial_status"] == "reward_timeout"
    assert rows2[1]["net_points"] == "" and rows2[3]["net_points"] == ""
    assert rows2[4]["trial_status"] == "completed"

    folder3 = run_session(config, FakeScreens(abort_trial=3), "test_abort", output_dir=tmp)
    rows3 = load_rows(folder3)
    assert len(rows3) == 3 and rows3[-1]["trial_status"] == "aborted"

    # Nie wolno przypadkowo uruchomic testowego prototypu w trybie experiment.
    bad = dict(config)
    bad["experiment"] = dict(config["experiment"], mode="experiment")
    try:
        run_session(bad, FakeScreens(), "bad", output_dir=tmp)
    except ValueError:
        pass
    else:
        raise AssertionError("Prototyp powinien odmawiac uruchomienia jako eksperyment")

    print(f"OK - zapis {len(rows)} prob, punktacja, brakujace odpowiedzi i przerwanie sesji dzialaja.")
    print("OK - dane ukryte sa w CSV, a sygnal niekupiony nie jest prezentowany jako zaobserwowany.")
    print("OK - konfiguracja i plan sesji sa zapisywane osobno.")
    print("WYNIKI TESTU: folder tymczasowy automatycznie usuniety po sprawdzeniu.")

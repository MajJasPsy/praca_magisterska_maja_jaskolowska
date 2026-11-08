"""KROK 6: test treningu, czesci glownej i zapisu, bez uruchamiania PsychoPy."""
from copy import deepcopy
from pathlib import Path
import csv
import json
import tempfile
import yaml

from session_runner import run_session

ROOT = Path(__file__).resolve().parent
with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Test treningu używa konfiguracji bez quizu, niezależnie od ustawień użytkownika.
# Osobny scenariusz na końcu testuje blokadę po niezaliczonym quizie.
config["training"]["comprehension_check"]["enabled"] = False


class FakeScreens:
    left_key = "f"
    right_key = "j"

    def __init__(self, abort_training_trial=None):
        self.messages = []
        self.current_phase = None
        self.current_trial = 0
        self.abort_training_trial = abort_training_trial

    def message(self, text, seconds=None, continue_required=False):
        self.messages.append(str(text))

    def choice(self, title, left, right, timeout):
        if "Decyzja 1" in title:
            self.current_phase = "training" if "TRENING" in title else "experiment"
            self.current_trial = int(title.split("Proba ")[1].split("/")[0])
            if self.current_phase == "training" and self.current_trial == self.abort_training_trial:
                raise KeyboardInterrupt
            return "f", 0.33
        return "j", 0.57


def rows_of(folder, name):
    with (folder / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


with tempfile.TemporaryDirectory() as tmp:
    screens = FakeScreens()
    folder = run_session(config, screens, "test_training", output_dir=tmp)
    rows = rows_of(folder, "trials.csv")
    plans = rows_of(folder, "planned_trials.csv")
    meta = json.loads((folder / "session_metadata.json").read_text(encoding="utf-8"))
    ntrain = config["training"]["n_trials"]
    nmain = config["experiment"]["n_trials"]
    assert len(rows) == ntrain + nmain, (len(rows), ntrain, nmain)
    assert len(plans) == len(rows)
    assert [r["phase"] for r in rows] == ["training"] * ntrain + ["experiment"] * nmain
    assert [r["phase"] for r in plans] == [r["phase"] for r in rows]
    assert all(r["trial_status"] == "completed" for r in rows)
    assert [int(r["trial_index"]) for r in rows[:ntrain]] == list(range(1, ntrain + 1))
    assert [int(r["trial_index"]) for r in rows[ntrain:]] == list(range(1, nmain + 1))
    assert meta["n_planned_training_trials"] == ntrain
    assert meta["n_planned_trials"] == nmain
    assert meta["training_plan_seed"] == meta["plan_seed"] + 2
    assert meta["training_environment_seed"] == meta["plan_seed"] + 3
    assert meta["environment_seed"] == meta["plan_seed"] + 1
    for pl, r in zip(plans, rows):
        for key in ("phase", "trial_index", "block_index", "condition_id", "risky_side", "buy_cue_side", "cue_accuracy"):
            assert pl[key] == r[key], (key,pl[key],r[key])
    # Zaden wynik treningowy nie przenosi sie do bilansu czesci glownej.
    tr_end = float(rows[ntrain - 1]["running_total"])
    assert tr_end > 0
    assert abs(float(rows[ntrain]["running_total"]) - float(rows[ntrain]["net_points"]) - float(config["scoring"]["starting_points"])) < 1e-9
    assert any("Kontekst:" in msg for msg in screens.messages)
    # Po ostatnim wyborze treningowym FakeScreens.current_phase przechodzi na experiment.
    # Feedback jest zapisywany razem z aktualnie aktywna faza w oddzielnym tescie.
    assert config["training"]["feedback"]["show_true_context"] is True
    assert config["main_task"]["feedback"]["show_true_context"] is False
    for r in rows:
        assert r["cue_signal_observed"] == (r["cue_signal_generated"] if r["cue_choice"] == "buy" else "")
        assert abs(float(r["reward_points"]) - float(r["cue_cost_paid"]) - float(r["net_points"])) < 1e-9

    disabled = deepcopy(config)
    disabled["training"]["enabled"] = False
    folder_disabled = run_session(disabled, FakeScreens(), "without_training", output_dir=tmp)
    assert len(rows_of(folder_disabled, "trials.csv")) == nmain
    assert all(r["phase"] == "experiment" for r in rows_of(folder_disabled, "trials.csv"))
    # Trening nie moze zmieniac zaplanowanych prob ani losowan w czesci glownej.
    no_training_rows = rows_of(folder_disabled, "trials.csv")
    for a, b in zip(rows[ntrain:], no_training_rows):
        for key in ("condition_id", "true_context", "cue_signal_generated",
                    "risky_reward_generated", "buy_cue_side", "risky_side"):
            assert a[key] == b[key], (key, a[key], b[key])

    aborted = run_session(config, FakeScreens(abort_training_trial=2), "abort_training", output_dir=tmp)
    abort_rows = rows_of(aborted, "trials.csv")
    assert len(abort_rows) == 2 and abort_rows[-1]["trial_status"] == "aborted"
    assert all(r["phase"] == "training" for r in abort_rows)

    bad = deepcopy(config)
    bad["training"]["comprehension_check"]["enabled"] = True
    blocked = run_session(bad, FakeScreens(), "quiz_fail_training", output_dir=tmp)
    blocked_rows = rows_of(blocked, "trials.csv")
    assert len(blocked_rows) == ntrain, "Nieprzejscie quizu powinno blokowac faze glowna"
    summary = json.loads((blocked / "comprehension_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"

    print(f"OK - zapisano {ntrain} prob treningowych i {nmain} prob czesci glownej.")
    print("OK - oddzielna punktacja, konfiguracja feedbacku i niezalezne losowania.")
    print("OK - brak treningu, przerwanie oraz blokada czesci glownej po niezaliczonym quizie.")
    print("KROK 6 - TESTY ZAKONCZONE POMYSLNIE.")

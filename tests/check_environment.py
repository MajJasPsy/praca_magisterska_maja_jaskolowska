"""Test pojedynczych prob i kontrola statystyczna losowan — bez PsychoPy."""

from collections import Counter
from pathlib import Path
from math import isclose
import yaml
from environment import BanditEnvironment


ROOT = Path(__file__).resolve().parent
with (ROOT / "config.yaml").open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

engine = BanditEnvironment(config)
condition = config["design"]["conditions"][0]
trial = engine.prepare_trial(condition)
print("PRZYKLADOWA PROBA (podglad techniczny — nie ekran uczestnika):")
print(trial)

print("\nJESLI UCZESTNIK KUPI WSKAZOWKE I WYBIERZE RISKY:")
print(engine.resolve_trial(trial, buy_cue=True, reward_choice="risky"))
print("\nJESLI UCZESTNIK NIE KUPI WSKAZOWKI I WYBIERZE SAFE:")
print(engine.resolve_trial(trial, buy_cue=False, reward_choice="safe"))

# Statystyczne sprawdzenie zgodnosci generatora z konfiguracja.
N = 10000
check_engine = BanditEnvironment(config, seed=2026)
for cond in config["design"]["conditions"]:
    accurate = 0
    for _ in range(N):
        t = check_engine.prepare_trial(cond)
        accurate += (t.cue_signal == t.context)
    observed = accurate / N
    expected = cond["cue_accuracy"]
    assert abs(observed - expected) < 0.03, (cond["id"], observed, expected)
    print(f"\n{cond['id']}: trafnosc oczekiwana={expected:.2f}; z symulacji={observed:.3f}")

for name, state in config["contexts"]["states"].items():
    n = 10000
    samples = [check_engine.sample_risky_reward(name) for _ in range(n)]
    counts = Counter(samples)
    observed = [counts[float(value)] / n for value in config["rewards"]["risky_outcomes"]]
    expected = state["risky_probabilities"]
    assert all(abs(o - e) < .03 for o, e in zip(observed, expected)), name
    print(f"{name}: rozklad nagrod OK")

# Punkty: kupiona wskazowka jest odliczana dokladnie raz.
with_cue = engine.resolve_trial(trial, True, "safe")
without_cue = engine.resolve_trial(trial, False, "safe")
assert isclose(with_cue["net_points"], without_cue["net_points"] - trial.cue_cost)
assert without_cue["cue_signal_observed"] is None
print("\nOK - generator kontekstow, wskazowek, nagrod i punktacji dziala.")

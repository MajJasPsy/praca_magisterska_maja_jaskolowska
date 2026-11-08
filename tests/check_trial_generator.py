"""Sprawdzenie planu calej czesci eksperymentalnej, bez interfejsu PsychoPy."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import yaml

from trial_generator import TrialGenerator
from environment import BanditEnvironment

ROOT = Path(__file__).resolve().parent
with (ROOT / "config.yaml").open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

generator = TrialGenerator(config)
schedule = generator.generate()

print("SEED PLANU:", generator.seed)
print("LICZBA PROB:", len(schedule))
print("LICZBA BLOKOW:", config["experiment"]["n_blocks"])
print("WARUNKI:", dict(Counter(t.condition_id for t in schedule)))
print("ROZMIARY BLOKOW:", dict(sorted(Counter(t.block_index for t in schedule).items())))

print("\nPIERWSZE PROBY PLANU:")
for trial in schedule[:8]:
    print(asdict(trial))

# Liczba prob, numeracja i poprawne parametry.
assert len(schedule) == config["experiment"]["n_trials"]
assert [t.trial_index for t in schedule] == list(range(1, len(schedule) + 1))
assert set(t.condition_id for t in schedule) == {c["id"] for c in config["design"]["conditions"]}
assert all(t.buy_cue_side in {"left", "right"} and t.risky_side in {"left", "right"} for t in schedule)
conditions = {c["id"]: c for c in config["design"]["conditions"]}
for t in schedule:
    cond = conditions[t.condition_id]
    assert t.cue_accuracy == cond["cue_accuracy"] and t.cue_cost == cond["cue_cost"]

# Pozycje odpowiedzi powinny byc zrownowazone w kazdym warunku osobno.
for cid in conditions:
    subset = [t for t in schedule if t.condition_id == cid]
    for field in ("buy_cue_side", "risky_side"):
        positions = Counter(getattr(t, field) for t in subset)
        assert abs(positions["left"] - positions["right"]) <= 1, (cid, field)

# Ograniczenie liczby identycznych warunkow pod rzad.
max_run = config["design"]["max_identical_in_row"]
streak = 0
last = None
for t in schedule:
    streak = streak + 1 if t.condition_id == last else 1
    assert streak <= max_run
    last = t.condition_id

# Identyczne seed + konfiguracja => identyczna sekwencja.
assert schedule == TrialGenerator(config).generate()

# Nowa liczba prob i liczba blokow, bez zmieniania kodu generatora.
modified = yaml.safe_load(yaml.safe_dump(config))
modified["experiment"]["n_trials"] = 23
modified["experiment"]["n_blocks"] = 5
other = TrialGenerator(modified).generate()
assert len(other) == 23
assert len({t.block_index for t in other}) == 5

# Kontrola przy niemozliwych proporcjach.
impossible = yaml.safe_load(yaml.safe_dump(config))
impossible["experiment"]["n_trials"] = 15
impossible["design"]["max_identical_in_row"] = 1
impossible["design"]["conditions"][0]["weight"] = 1000
try:
    TrialGenerator(impossible).generate()
except ValueError as error:
    print("\nPoprawnie wykryto niemozliwe proporcje:", error)
else:
    raise AssertionError("Generator powinien odmowic niemozliwej konfiguracji")

# Integracja z istniejacym silnikiem srodowiska.
# UWAGA: wspolne ziarno bazowe, inny strumien losowania dla srodowiska.
engine = BanditEnvironment(config, seed=generator.seed + 1)
for planned in schedule:
    t = engine.prepare_trial(conditions[planned.condition_id])
    assert t.condition_id == planned.condition_id
    assert t.cue_accuracy == planned.cue_accuracy
    assert t.cue_cost == planned.cue_cost

print("\nOK - generator sekwencji prob, blokow i kolejnosci warunkow dziala.")

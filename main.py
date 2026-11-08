from pathlib import Path
from math import isclose
import yaml

# Znajdz config.yaml obok tego pliku, niezaleznie od katalogu uruchomienia.
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

experiment = config["experiment"]
contexts = config["contexts"]["states"]
outcomes = config["rewards"]["risky_outcomes"]
conditions = config["design"]["conditions"]

assert experiment["n_trials"] > 0, "Liczba prob musi byc dodatnia."
assert 1 <= experiment["n_blocks"] <= experiment["n_trials"], "Niepoprawna liczba blokow."
assert len(contexts) >= 2, "Potrzebne sa co najmniej dwa konteksty."
assert len(outcomes) >= 2, "Potrzebne sa co najmniej dwie mozliwe nagrody ryzykowne."
assert sum(state["weight"] for state in contexts.values()) > 0, "Niepoprawne wagi kontekstow."
for name, state in contexts.items():
    assert state["weight"] >= 0, f"Ujemna waga kontekstu {name}."
    probs = state["risky_probabilities"]
    assert len(probs) == len(outcomes), f"Kontekst {name}: zla liczba prawdopodobienstw."
    assert all(0 <= p <= 1 for p in probs), f"Kontekst {name}: prawdopodobienstwa poza [0,1]."
    assert isclose(sum(probs), 1.0, abs_tol=1e-8), f"Kontekst {name}: suma prawdopodobienstw != 1."
assert conditions, "Brak warunkow eksperymentalnych."
ids = [condition["id"] for condition in conditions]
assert len(ids) == len(set(ids)), "Identyfikatory warunkow nie sa unikatowe."
for condition in conditions:
    assert 0 <= condition["cue_accuracy"] <= 1, f"{condition['id']}: bledna wiarygodnosc."
    assert condition["cue_cost"] >= 0, f"{condition['id']}: ujemny koszt."
    assert condition["weight"] > 0, f"{condition['id']}: niepoprawna waga warunku."

print("OK - wczytano i sprawdzono podstawowe parametry konfiguracji.")
print("Tryb:", experiment["mode"])
print("Proby eksperymentalne:", experiment["n_trials"])
print("Bloki:", experiment["n_blocks"])
print("Konteksty:", ", ".join(contexts))
print("Warunki:", ", ".join(ids))
print("Trening:", config["training"]["n_trials"], "prob")

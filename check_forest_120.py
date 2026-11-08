"""Test aktualnej procedury: 30 darmowych + 90 płatnych prób, portfel i siatka kosztów.

Nie wymaga PsychoPy ani obrazów PNG. Uruchom z folderu głównego projektu.
"""
import csv
import json
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from environment import BanditEnvironment
from session_runner import make_schedules, run_session, verify_demo_config
from trial_generator import TrialGenerator
from participant_guidance import instruction_pages

ROOT = Path(__file__).resolve().parent
cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
verify_demo_config(cfg)

price_path = ROOT / "siatka_cen_skanera_czesc_eksperymentalna.csv"
with price_path.open(encoding="utf-8-sig", newline="") as handle:
    price_rows = list(csv.DictReader(handle))
source_prices = [Decimal(row["scanner_cost"]) for row in price_rows]
assert len(source_prices) == len(set(source_prices)) == 30
assert all(row["phase"] == "experimental" and Decimal(row["scanner_accuracy"]) == 1
           and int(row["repetitions"]) == 3 for row in price_rows)
assert sum(int(row["repetitions"]) for row in price_rows) == 90
assert [Decimal(str(c["cue_cost"])) for c in cfg["design"]["conditions"]] == source_prices
assert all(c["cue_accuracy"] == 1 and c["weight"] == 1 for c in cfg["design"]["conditions"])
assert cfg["training"]["n_trials"] == 30 and cfg["training"]["trials_per_context"] == 10
assert cfg["experiment"]["n_trials"] == 90
assert "W treningu leśniczy udziela wskazówek za darmo" in instruction_pages(cfg)[2]

main_gen, main_plan, train_plan, *_ = make_schedules(cfg)
assert len(train_plan) == 30 and len(main_plan) == 90
assert Counter(p.assigned_context for p in train_plan) == {k: 10 for k in cfg["contexts"]["states"]}
assert all(p.condition_id == "training_free" and p.cue_cost == 0.0 and p.cue_accuracy == 1.0 for p in train_plan)
assert all(p.risky_side == "right" for p in main_plan + train_plan)
assert set(p.buy_cue_side for p in main_plan) == {"left", "right"}
assert set(p.buy_cue_side for p in train_plan) == {"left", "right"}
assert Counter(Decimal(str(p.cue_cost)) for p in main_plan) == {price: 3 for price in source_prices}
assert all(p.assigned_context is None for p in main_plan)
assert main_plan == TrialGenerator(cfg).generate()


class FakeScreens:
    left_key, right_key = "left", "right"

    def __init__(self, reward_timeout_at=None):
        self.phase = None
        self.balance = None
        self.offers = []
        self.paths = []
        self.found = []
        self.messages = []
        self.reward_timeout_at = reward_timeout_at

    def set_phase(self, phase):
        self.phase = phase

    def set_balance(self, balance):
        self.balance = round(float(balance), 2)

    def message(self, text, seconds=None, continue_required=False):
        self.messages.append((self.phase, str(text)))

    def choose_forester(self, cost, timeout, buy_cue_side="right"):
        self.offers.append(dict(phase=self.phase, cost=cost,
                                buy_side=buy_cue_side, balance=self.balance))
        # co druga oferta kupiona, niezależnie od tego, po której jest stronie
        buy = len(self.offers) % 2 == 1
        chosen_side = buy_cue_side if buy else ("right" if buy_cue_side == "left" else "left")
        return chosen_side, 0.35

    def choose_path(self, forest, timeout):
        self.paths.append(dict(phase=self.phase, forest=forest, balance=self.balance))
        if self.reward_timeout_at == len(self.paths):
            return None, None
        return ("left" if len(self.paths) % 2 else "right"), 0.52

    def show_found(self, reward, seconds):
        self.found.append(dict(phase=self.phase, reward=reward, balance=self.balance))

    def choice(self, title, left, right, timeout):
        raise AssertionError("Test zrozumienia został wyłączony tylko dla tej symulacji")


def load_rows(folder):
    with (folder / "trials.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


with TemporaryDirectory() as temp:
    local_cfg = deepcopy(cfg)
    local_cfg["training"]["comprehension_check"]["enabled"] = False
    screens = FakeScreens()
    folder = run_session(local_cfg, screens, "test_wallet_120", output_dir=temp)
    rows = load_rows(folder)
    assert len(rows) == 120
    assert len(screens.offers) == len(screens.paths) == len(screens.found) == 120
    assert all(row["trial_status"] == "completed" for row in rows)
    assert Counter(row["true_context"] for row in rows if row["phase"] == "training") == {
        c: 10 for c in cfg["contexts"]["states"]}
    assert Counter(Decimal(row["cue_cost_offered"]) for row in rows if row["phase"] == "experiment") == {
        price: 3 for price in source_prices}
    training_seen = rows[:30]
    assert all(row["condition_id"] == "training_free" for row in training_seen)
    assert all(Decimal(row["cue_cost_offered"]) == 0 and Decimal(row["cue_cost_paid"]) == 0
               for row in training_seen)
    assert [offer["phase"] for offer in screens.offers] == [row["phase"] for row in rows]
    assert all(row["forest_image_displayed"] == cfg["images"]["forest_files"]["unknown"]
               for row in rows if row["cue_choice"] == "skip")

    previous_main = Decimal(str(cfg["scoring"]["starting_points"]))
    for i, (row, offer, path, found) in enumerate(zip(rows, screens.offers, screens.paths, screens.found)):
        bought = row["cue_choice"] == "buy"
        cost = Decimal(row["cue_cost_paid"])
        reward = Decimal(row["reward_points"])
        net = Decimal(row["net_points"])
        assert cost == (Decimal(row["cue_cost_offered"]) if bought else 0)
        assert net == reward - cost
        assert row["risky_side"] == "right"
        assert path["forest"] in {"A", "B", "C"} if bought else path["forest"] == "unknown"
        assert found["reward"] == float(reward)
        if row["phase"] == "experiment":
            assert Decimal(str(offer["balance"])) == previous_main
            assert Decimal(str(path["balance"])) == previous_main - cost
            previous_main += net
            assert Decimal(str(found["balance"])) == previous_main
            assert Decimal(row["running_total"]) == previous_main
        else:
            assert offer["phase"] == path["phase"] == found["phase"] == "training"
            assert Decimal(str(path["balance"])) == Decimal(str(offer["balance"]))
    assert screens.phase is None  # po sesji portfel jest chowany
    assert any("Od teraz wskazówki leśniczego są płatne" in msg for _, msg in screens.messages)
    assert any("W treningu możesz pytać leśniczego za darmo" in msg for _, msg in screens.messages)
    assert json.loads((folder / "session_metadata.json").read_text(encoding="utf-8"))["n_planned_trials"] == 90

    # Osobny scenariusz: zakup wskazówki + timeout ścieżki nie zmienia salda
    # (obowiązująca polityka timeout_policy: mark_missing).
    screens2 = FakeScreens(reward_timeout_at=31)  # pierwsza próba po 30 treningowych
    folder2 = run_session(local_cfg, screens2, "test_wallet_timeout", output_dir=temp)
    data2 = load_rows(folder2)
    assert data2[30]["trial_status"] == "reward_timeout"
    assert data2[30]["net_points"] == ""
    assert Decimal(data2[31]["running_total"]) == Decimal(data2[31]["net_points"]) + Decimal(str(cfg["scoring"]["starting_points"]))
    # saldo jest cofnięte, jeśli zakupiona wskazówka nie skutkuje ukończoną próbą
    assert screens2.offers[31]["balance"] == float(cfg["scoring"]["starting_points"])

print("OK — 30 darmowych prób treningu; po 10 na każdy las.")
print("OK — 90 prób części głównej; dokładnie 30 kosztów z CSV, po 3 powtórzenia.")
print("OK — leśniczy zawsze mówi prawdę; zmienne strony TAK/NIE, lewa droga bezpieczna.")
print("OK — saldo na każdym etapie: przed wskazówką, po opłacie i po nagrodzie.")
print("OK — trening bez kosztów, CSV poprawne, brak naliczania kosztu po timeout.")

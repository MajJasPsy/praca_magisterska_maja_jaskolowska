"""Logika srodowiska bandyty kontekstowego — bez interfejsu PsychoPy.

Na razie obsluguje model bledow wskazowki 'symmetric_other'
i niezalezne losowanie kontekstu w kazdej probie.
"""

from dataclasses import dataclass
from math import isclose, isfinite
import random


@dataclass(frozen=True)
class Trial:
    """Wszystkie wylosowane elementy proby (w tym dane ukryte)."""

    condition_id: str
    context: str
    cue_accuracy: float
    cue_cost: float
    cue_signal: str
    safe_reward: float
    risky_reward: float


class BanditEnvironment:
    def __init__(self, config: dict, seed: int | None = None):
        self.config = config
        self.rng = random.Random(
            config["experiment"].get("random_seed") if seed is None else seed
        )
        context_config = config["contexts"]
        if context_config["sampling"] != "independent":
            raise ValueError("Obslugiwany typ losowania kontekstow: independent.")
        if config["cue"]["error_model"] != "symmetric_other":
            raise ValueError("Obslugiwany model bledow wskazowki: symmetric_other.")

        self.states = context_config["states"]
        self.names = list(self.states)
        self.context_weights = [float(self.states[n]["weight"]) for n in self.names]
        self.outcomes = list(config["rewards"]["risky_outcomes"])
        self.safe_reward = float(config["rewards"]["safe_points"])

        if len(self.names) < 2:
            raise ValueError("Potrzebne sa co najmniej dwa konteksty.")
        if any(not isfinite(w) or w < 0 for w in self.context_weights):
            raise ValueError("Wagi kontekstow musza byc nieujemnymi liczbami skonczonymi.")
        if sum(self.context_weights) <= 0:
            raise ValueError("Suma wag kontekstow musi byc dodatnia.")
        if not self.outcomes or not all(isfinite(float(x)) for x in self.outcomes):
            raise ValueError("Nagrody musza byc liczbami skonczonymi.")

        for name in self.names:
            probs = self.states[name]["risky_probabilities"]
            if len(probs) != len(self.outcomes):
                raise ValueError(f"{name}: liczba prawdopodobienstw nie zgadza sie z nagrodami.")
            if any(not isfinite(float(p)) or not 0 <= p <= 1 for p in probs):
                raise ValueError(f"{name}: nieprawidlowe prawdopodobienstwa nagrod.")
            if not isclose(sum(probs), 1, abs_tol=1e-8):
                raise ValueError(f"{name}: prawdopodobienstwa nagrod musza sumowac sie do 1.")

    def sample_context(self) -> str:
        return self.rng.choices(self.names, weights=self.context_weights, k=1)[0]

    def sample_signal(self, true_context: str, accuracy: float) -> str:
        """P(poprawny sygnal) = accuracy; bledy rozdzielone rowno."""
        if true_context not in self.states:
            raise ValueError(f"Nieznany kontekst: {true_context}")
        if not isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Wiarygodnosc musi byc w przedziale [0, 1].")
        n = len(self.names)
        weights = [accuracy if name == true_context else (1 - accuracy) / (n - 1)
                   for name in self.names]
        return self.rng.choices(self.names, weights=weights, k=1)[0]

    def sample_risky_reward(self, context: str) -> float:
        if context not in self.states:
            raise ValueError(f"Nieznany kontekst: {context}")
        probabilities = self.states[context]["risky_probabilities"]
        return float(self.rng.choices(self.outcomes, weights=probabilities, k=1)[0])

    def prepare_trial(self, condition: dict, *, context: str | None = None) -> Trial:
        """Losuje elementy proby przed wyborem uczestnika.

        Sygnał jest generowany także wtedy, gdy uczestnik go nie kupi,
        ale nie bedzie mu pokazywany. To zapobiega zmianom w dalszej
        sekwencji losowan wskutek decyzji o zakupie.
        """
        accuracy = float(condition["cue_accuracy"])
        cost = float(condition["cue_cost"])
        if not 0 <= accuracy <= 1 or not isfinite(accuracy):
            raise ValueError("Nieprawidlowa wiarygodnosc wskazowki.")
        if cost < 0 or not isfinite(cost):
            raise ValueError("Nieprawidlowy koszt wskazowki.")
        # W treningu można wymusić konkretny las, żeby mieć po równo prób.
        # W części głównej context=None: ukryty las losujemy jak dawniej.
        if context is None:
            context = self.sample_context()
        elif context not in self.states:
            raise ValueError(f"Nieznany kontekst treningowy: {context}")
        return Trial(
            condition_id=str(condition["id"]),
            context=context,
            cue_accuracy=accuracy,
            cue_cost=cost,
            cue_signal=self.sample_signal(context, accuracy),
            safe_reward=self.safe_reward,
            risky_reward=self.sample_risky_reward(context),
        )

    @staticmethod
    def resolve_trial(trial: Trial, buy_cue: bool, reward_choice: str) -> dict:
        """Rozlicza jedna probe. Pelny slownik sluzy do zapisu danych, NIE do ekranu."""
        if not isinstance(buy_cue, bool):
            raise ValueError("buy_cue musi byc True albo False.")
        if reward_choice not in {"safe", "risky"}:
            raise ValueError("reward_choice musi wynosic 'safe' albo 'risky'.")
        reward = trial.safe_reward if reward_choice == "safe" else trial.risky_reward
        cost = trial.cue_cost if buy_cue else 0.0
        return {
            "condition_id": trial.condition_id,
            "true_context": trial.context,          # do danych, nie na ekran
            "cue_accuracy": trial.cue_accuracy,
            "cue_cost_offered": trial.cue_cost,
            "cue_bought": buy_cue,
            "cue_signal_observed": trial.cue_signal if buy_cue else None,
            "reward_choice": reward_choice,
            "reward_points": reward,
            "cue_cost_paid": cost,
            "net_points": reward - cost,
        }

"""Planowanie prob eksperymentu niezalezne od PsychoPy.

WAZNE: ta czesc losuje kolejnosc WARUNKOW i strony przyciskow,
ale nie ukryte konteksty, sygnaly i nagrody (to robi environment.py).

'balanced' przydziela calkowite liczby prob do warunkow wg wag, a potem
rownomiernie rozprowadza je w czasie z ograniczeniem dlugosci serii.
W obrebie blokow rownowaga jest przyblizona, a nie gwarantowana osobno.
"""

from dataclasses import dataclass
import math
import random
import secrets


@dataclass(frozen=True)
class PlannedTrial:
    trial_index: int       # numer proby od 1 w calej czesci eksperymentalnej
    block_index: int       # numer bloku od 1
    in_block_index: int    # numer proby od 1 w bloku
    condition_id: str
    cue_accuracy: float
    cue_cost: float
    buy_cue_side: str      # left / right
    risky_side: str        # left / right
    assigned_context: str | None = None  # tylko trening: z góry zaplanowany las


class TrialGenerator:
    def __init__(self, config: dict, seed: int | None = None):
        self.config = config
        exp = config["experiment"]
        self.seed = (seed if seed is not None else exp.get("random_seed"))
        if self.seed is None:
            self.seed = secrets.randbits(64)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("random_seed musi byc calkowita liczba lub null")
        self.rng = random.Random(self.seed)

        self.n_trials = exp["n_trials"]
        self.n_blocks = exp["n_blocks"]
        if (isinstance(self.n_trials, bool) or not isinstance(self.n_trials, int)
                or self.n_trials < 1):
            raise ValueError("n_trials musi byc dodatnia liczba calkowita")
        if (isinstance(self.n_blocks, bool) or not isinstance(self.n_blocks, int)
                or not 1 <= self.n_blocks <= self.n_trials):
            raise ValueError("n_blocks musi byc liczba calkowita od 1 do n_trials")

        design = config["design"]
        if design["assignment"] != "balanced":
            raise ValueError("Na razie obslugujemy tylko design.assignment = balanced")
        self.randomize_order = design["randomize_order"]
        if not isinstance(self.randomize_order, bool):
            raise ValueError("design.randomize_order musi byc true albo false")
        self.max_run = design["max_identical_in_row"]
        if (isinstance(self.max_run, bool) or not isinstance(self.max_run, int)
                or self.max_run < 1):
            raise ValueError("max_identical_in_row musi byc dodatnia liczba calkowita")
        conditions = design["conditions"]
        if not isinstance(conditions, list) or not conditions:
            raise ValueError("design.conditions musi byc niepusta lista")
        self.conditions = {str(c["id"]): c for c in conditions}
        if len(self.conditions) != len(conditions):
            raise ValueError("Identyfikatory warunkow musza byc unikatowe")
        for c in conditions:
            for field in ("cue_accuracy", "cue_cost", "weight"):
                val = c[field]
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(f"{c['id']}: {field} musi byc skonczona liczba")
            if not 0 <= c["cue_accuracy"] <= 1 or c["cue_cost"] < 0 or c["weight"] <= 0:
                raise ValueError(f"Nieprawidlowe parametry warunku {c['id']}")

        controls = config["controls"]
        self.shuffle_cue_sides = controls["randomize_cue_option_positions"]
        self.shuffle_reward_sides = controls["randomize_reward_option_positions"]
        if not isinstance(self.shuffle_cue_sides, bool) or not isinstance(self.shuffle_reward_sides, bool):
            raise ValueError("Ustawienia randomizacji stron musza byc true albo false")

    def _balanced_counts(self) -> dict[str, int]:
        """Metoda najwiekszych reszt: proporcje w calym badaniu jak najblizej wagom."""
        ids = list(self.conditions)
        weights = [float(self.conditions[c]["weight"]) for c in ids]
        total = sum(weights)
        quotas = [self.n_trials * w / total for w in weights]
        counts = [math.floor(x) for x in quotas]
        missing = self.n_trials - sum(counts)
        tie = list(range(len(ids)))
        if self.randomize_order:
            self.rng.shuffle(tie)
        tie_rank = {value: idx for idx, value in enumerate(tie)}
        ranked = sorted(range(len(ids)), key=lambda i: (-(quotas[i] - counts[i]), tie_rank[i]))
        for idx in ranked[:missing]:
            counts[idx] += 1
        return dict(zip(ids, counts))

    def _feasible(self, remaining: dict[str, int], last: str | None, run: int) -> bool:
        """Warunek mozliwosci ulozenia pozostalych symboli bez serii > max_run."""
        left = sum(remaining.values())
        for name, count in remaining.items():
            if count == 0:
                continue
            others = left - count
            capacity = self.max_run * (others + 1)
            if name == last:
                capacity -= run
            if count > capacity:
                return False
        return True

    def _make_condition_order(self, counts: dict[str, int]) -> list[str]:
        if not self._feasible(counts, None, 0):
            raise ValueError(
                "Nie da sie ulozyc wskazanej proporcji warunkow przy "
                f"max_identical_in_row={self.max_run}. Zmien wagi/limit albo liczbe prob."
            )
        remaining = counts.copy()
        used = dict.fromkeys(counts, 0)
        order: list[str] = []
        last = None
        run = 0
        for position in range(self.n_trials):
            candidates = []
            for name in counts:
                if remaining[name] == 0 or (name == last and run >= self.max_run):
                    continue
                remaining[name] -= 1
                next_run = run + 1 if name == last else 1
                possible = self._feasible(remaining, name, next_run)
                remaining[name] += 1
                if not possible:
                    continue
                # Uzywamy deficytu wzgledem docelowej proporcji, aby nie skupic
                # calych warunkow na poczatku/koncu badania.
                target = (position + 1) * counts[name] / self.n_trials
                deficit = target - used[name]
                noise = self.rng.uniform(-0.30, 0.30) if self.randomize_order else 0.0
                candidates.append((deficit + noise, name))
            if not candidates:
                raise ValueError("Nie udalo sie ulozyc warunkow bez przekroczenia limitu serii")
            # Tie ordering is stable if no randomization.
            _, chosen = max(candidates, key=lambda pair: pair[0])
            order.append(chosen)
            used[chosen] += 1
            remaining[chosen] -= 1
            run = run + 1 if chosen == last else 1
            last = chosen
        return order

    def generate(self) -> list[PlannedTrial]:
        counts = self._balanced_counts()
        order = self._make_condition_order(counts)
        block_base, extra = divmod(self.n_trials, self.n_blocks)
        block_sizes = [block_base + int(i < extra) for i in range(self.n_blocks)]
        # Rowne (lub rozniace sie o 1) liczby pozycji lewo/prawo
        # OSOBNO w kazdym warunku - wazne dla kontroli tendencji motorycznych.
        from collections import defaultdict
        occurrences = defaultdict(list)
        for pos, cid in enumerate(order):
            occurrences[cid].append(pos)

        cue_sides = ["right"] * len(order)
        reward_sides = ["right"] * len(order)  # ścieżka ryzykowna ZAWSZE po prawej
        extra_goes_left = True
        for positions in occurrences.values():
            k = len(positions)
            if self.shuffle_cue_sides:
                # Równoważymy TAK/NIE w każdym warunku; przy nieparzystym k
                # nadmiarową stronę przydzielamy naprzemiennie warunkom.
                sides = ["left"] * (k // 2) + ["right"] * (k // 2)
                if k % 2:
                    sides.append("left" if extra_goes_left else "right")
                    extra_goes_left = not extra_goes_left
                self.rng.shuffle(sides)
                for pos, side in zip(positions, sides):
                    cue_sides[pos] = side
            if self.shuffle_reward_sides:
                # Funkcja generatora pozostała ogólna, ale tryb trzech lasów
                # wymaga ustawienia randomize_reward_option_positions: false.
                sides = ["left"] * (k // 2) + ["right"] * (k - k // 2)
                self.rng.shuffle(sides)
                for pos, side in zip(positions, sides):
                    reward_sides[pos] = side

        planned = []
        ix = 0
        for block_idx, size in enumerate(block_sizes, start=1):
            for within_idx in range(1, size + 1):
                condition = self.conditions[order[ix]]
                buy_cue_side = cue_sides[ix]
                risky_side = reward_sides[ix]
                planned.append(PlannedTrial(
                    trial_index=ix + 1,
                    block_index=block_idx,
                    in_block_index=within_idx,
                    condition_id=str(condition["id"]),
                    cue_accuracy=float(condition["cue_accuracy"]),
                    cue_cost=float(condition["cue_cost"]),
                    buy_cue_side=buy_cue_side,
                    risky_side=risky_side,
                ))
                ix += 1
        assert len(planned) == self.n_trials
        return planned

"""KROK 6: trening oraz czesc eksperymentalna, w jednym pliku CSV.

Ten program jest tylko prototypem technicznym. Parametry naukowe,
instrukcje, test rozumienia i warunki diagnostyczne wymagaja zatwierdzenia.
Logika jednej proby jest WSPOLNA dla treningu i glownego eksperymentu.
Nie wymaga importu PsychoPy: ekranem steruje obiekt 'screens'.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random

from environment import BanditEnvironment
from trial_generator import TrialGenerator
from data_manager import SessionRecorder
from participant_guidance import show_instructions, run_comprehension_check
from text_labels import display_context


def _money(value):
    return f"{float(value):.2f}".replace(".", ",") + " zł"


def fmt(value):
    return f"{float(value):g}"


def verify_demo_config(config: dict):
    if config["experiment"]["mode"] != "test":
        raise ValueError("Ten prototyp działa tylko przy experiment.mode: test.")
    if config["confidence"]["enabled"]:
        raise NotImplementedError("Pomiar pewności nie jest przewidziany w tej procedurze.")
    if config["controls"]["left_key"] != "left" or config["controls"]["right_key"] != "right":
        raise ValueError("W tej wersji wybór odbywa się strzałkami: left / right")
    if config["controls"]["randomize_cue_option_positions"] is not True:
        raise ValueError("Strony TAK i NIE powinny się zmieniać — ustaw randomize_cue_option_positions: true")
    if config["controls"]["randomize_reward_option_positions"] is not False:
        raise ValueError("Lewa ścieżka zawsze bezpieczna, prawa ryzykowna — ustaw randomize_reward_option_positions: false")
    if any(float(c["cue_accuracy"]) != 1.0 for c in config["design"]["conditions"]):
        raise ValueError("Leśniczy ma zawsze mówić prawdę: każdy cue_accuracy musi być 1.0")
    if config["training"]["feedback"]["show_true_context"] or config["main_task"]["feedback"]["show_true_context"]:
        raise ValueError("Feedback ma pokazywać tylko znalezioną kwotę, nie prawdziwy las")
    if config["scoring"]["bonus"]["enabled"]:
        raise NotImplementedError("Wypłaty finansowe nie są jeszcze zaimplementowane.")
    if config["controls"]["timeout_policy"] != "mark_missing":
        raise ValueError("Na razie obsługiwany timeout_policy: mark_missing")
    if config["training"]["comprehension_check"]["enabled"]:
        if not config["training"]["enabled"]:
            raise ValueError("Test zrozumienia wymaga włączonego treningu.")
        quiz = config["training"]["comprehension_check"]
        if not 0 < float(quiz["min_correct_fraction"]) <= 1:
            raise ValueError("min_correct_fraction musi należeć do (0, 1].")
        if type(quiz["max_attempts"]) is not int or quiz["max_attempts"] < 1:
            raise ValueError("max_attempts musi być liczbą całkowitą >= 1.")
    if config["training"]["enabled"]:
        if config["training"].get("use_main_conditions") is not False:
            raise ValueError("Trening wymaga odrębnej, bezpłatnej wskazówki: use_main_conditions: false")
        n = config["training"].get("n_trials")
        if type(n) is not int or n < 1:
            raise ValueError("training.n_trials musi być dodatnią liczbą całkowitą")
        per_context = config["training"].get("trials_per_context")
        n_contexts = len(config["contexts"]["states"])
        if type(per_context) is not int or per_context < 1:
            raise ValueError("training.trials_per_context musi być dodatnią liczbą całkowitą")
        if n != n_contexts * per_context:
            raise ValueError("Trening wymaga dokładnie trials_per_context prób na każdy las")
    total_planned = config["experiment"]["n_trials"] + (config["training"]["n_trials"] if config["training"]["enabled"] else 0)
    if total_planned != 120:
        raise ValueError(f"Ustalona procedura wymaga 120 prób łącznie, obecnie: {total_planned}")


def make_schedules(config):
    """Wazne: losowania treningu sa niezalezne od losowan w czesci glownej."""
    main_gen = TrialGenerator(config)
    main_schedule = main_gen.generate()
    train_seed = None
    train_env_seed = None
    training_schedule = []
    if config["training"]["enabled"]:
        training_config = deepcopy(config)
        training_config["experiment"]["n_trials"] = config["training"]["n_trials"]
        training_config["experiment"]["n_blocks"] = 1
        # Jeden odrębny warunek: każda wskazówka treningowa jest za darmo.
        training_config["design"]["conditions"] = [
            {"id": "training_free", "cue_accuracy": 1.0, "cue_cost": 0.0, "weight": 1}
        ]
        # Przy pojedynczym warunku ograniczenie długości serii nie ma sensu.
        training_config["design"]["max_identical_in_row"] = config["training"]["n_trials"]
        train_seed = main_gen.seed + 2
        train_env_seed = main_gen.seed + 3
        train_gen = TrialGenerator(training_config, seed=train_seed)
        training_schedule = train_gen.generate()
        # Po 10 rzeczywistych kontekstów A/B/C: osobne losowanie,
        # niezależne od kolejności kosztów i planu części głównej.
        labels = list(config["contexts"]["states"])
        per_context = int(config["training"]["trials_per_context"])
        context_rng = random.Random(main_gen.seed + 4)
        contexts = [label for label in labels for _ in range(per_context)]
        limit = int(config["design"]["max_identical_in_row"])
        for _ in range(10000):
            context_rng.shuffle(contexts)
            longest = 1
            streak = 1
            for idx in range(1, len(contexts)):
                streak = streak + 1 if contexts[idx] == contexts[idx - 1] else 1
                longest = max(longest, streak)
            if longest <= limit:
                break
        else:
            raise ValueError("Nie udało się ułożyć zrównoważonych kontekstów treningu")
        training_schedule = [replace(p, assigned_context=c) for p, c in zip(training_schedule, contexts)]
    return main_gen, main_schedule, training_schedule, train_seed, train_env_seed


def _run_phase(*, phase, schedule, env, conditions, config, screens, recorder):
    """Zwraca (suma_w_fazie, czy_przerwano). Brak punktow treningu w sumie glownej."""
    is_training = phase == "training"
    phase_name = "TRENING" if is_training else "EKSPERYMENT"
    timing = config["timing"]
    feedback_options = (config["training"]["feedback"] if is_training
                        else config["main_task"]["feedback"])
    total = 0.0 if is_training else float(config["scoring"]["starting_points"])
    # Portfel nie istnieje w treningu, instrukcji ani quizie.
    if hasattr(screens, "set_phase"):
        screens.set_phase(phase)

    for p in schedule:
        if hasattr(screens, "set_balance"):
            screens.set_balance(total)
        trial = env.prepare_trial(conditions[p.condition_id], context=p.assigned_context)
        row = {
            "phase": phase,
            "trial_index": p.trial_index,
            "block_index": p.block_index,
            "in_block_index": p.in_block_index,
            "condition_id": p.condition_id,
            "cue_accuracy": trial.cue_accuracy,
            "cue_cost_offered": trial.cue_cost,
            "buy_cue_side": p.buy_cue_side,
            "risky_side": p.risky_side,
            "planned_context": p.assigned_context,  # Tylko trening: równoważenie lasów
            "true_context": trial.context,  # Tylko w pliku z danymi
            "cue_signal_generated": trial.cue_signal,  # Tylko w pliku z danymi
            "risky_reward_generated": trial.risky_reward,  # Tylko w pliku z danymi
            "running_total": total,
            "trial_status": "started",
        }
        try:
            if p.in_block_index == 1:
                screens.message(
                    f"{phase_name} — blok {p.block_index}/{1 if is_training else config['experiment']['n_blocks']}\n"
                    "Naciśnij SPACJĘ, aby zacząć.",
                    continue_required=True,
                )
            screens.message("+", seconds=float(timing["fixation_seconds"]))
            # Pierwszy ekran: TAK/NIE zmieniają strony. Drugi ekran:
            # LEWO zawsze bezpieczna, PRAWO zawsze ryzykowna.
            # Nowe ekrany wybieramy przez metody PreviewScreens; starsze
            # atrapy testowe mogą nadal korzystać ze zwykłego choice().
            if hasattr(screens, "choose_forester"):
                k1, rt1 = screens.choose_forester(
                    trial.cue_cost, timing["cue_decision_timeout_seconds"],
                    buy_cue_side=p.buy_cue_side)
            else:
                k1, rt1 = screens.choice(
                    f"{phase_name} | Proba {p.trial_index}/{len(schedule)} | Decyzja 1",
                    "TAK" if p.buy_cue_side == "left" else "NIE",
                    "TAK" if p.buy_cue_side == "right" else "NIE",
                    timing["cue_decision_timeout_seconds"])
            row["rt_cue_seconds"] = rt1
            if k1 is None:
                row["trial_status"] = "cue_timeout"
                screens.message("Brak odpowiedzi.", seconds=1.0)
            else:
                side1 = "left" if k1 == screens.left_key else "right"
                bought = side1 == p.buy_cue_side
                row["cue_choice"] = "buy" if bought else "skip"
                row["cue_signal_observed"] = trial.cue_signal if bought else None
                forest = display_context(trial.cue_signal, config) if bought else "unknown"
                # W części głównej koszt zostaje od razu odjęty na ekranie ścieżek.
                # Przy braku ukończenia próby cofamy tę prezentację, zgodnie z mark_missing.
                if hasattr(screens, "set_balance") and not is_training:
                    screens.set_balance(round(total - (trial.cue_cost if bought else 0.0), 2))
                files = config["images"]["forest_files"]
                row["forest_image_displayed"] = files[forest]

                # Gotowe plansze zawierają już dwie ścieżki i strzałki.
                # Nic nie jest nakładane na ilustrację.
                if hasattr(screens, "choose_path"):
                    k2, rt2 = screens.choose_path(
                        forest, timing["reward_decision_timeout_seconds"])
                else:
                    k2, rt2 = screens.choice(
                        f"{phase_name} | Proba {p.trial_index}/{len(schedule)} | Decyzja 2",
                        "BEZPIECZNA", "RYZYKOWNA", timing["reward_decision_timeout_seconds"])
                row["rt_reward_seconds"] = rt2
                if k2 is None:
                    row["trial_status"] = "reward_timeout"
                    if hasattr(screens, "set_balance"):
                        screens.set_balance(total)  # brak naliczenia kosztu przy reward_timeout
                    screens.message("Brak odpowiedzi.", seconds=1.0)
                else:
                    side2 = "left" if k2 == screens.left_key else "right"
                    reward_choice = "safe" if side2 == "left" else "risky"
                    resolved = env.resolve_trial(trial, buy_cue=bought, reward_choice=reward_choice)
                    row["reward_choice"] = reward_choice
                    row["reward_points"] = resolved["reward_points"]
                    row["cue_cost_paid"] = resolved["cue_cost_paid"]
                    row["net_points"] = resolved["net_points"]
                    row["trial_status"] = "completed"
                    total = round(total + resolved["net_points"], 2)
                    row["running_total"] = total
                    if hasattr(screens, "set_balance"):
                        screens.set_balance(total)  # saldo po nagrodzie i opłacie
                    if hasattr(screens, "show_found"):
                        screens.show_found(resolved["reward_points"],
                                           float(timing["reward_feedback_seconds"]))
                    else:
                        screens.message(f"Znaleziono: {_money(resolved['reward_points'])}",
                                        seconds=float(timing["reward_feedback_seconds"]))
            # Polityka timeout w prototypie: brak punktow i brak kosztu.
            recorder.save_trial(row)
            if p.trial_index < len(schedule):
                screens.message("+", seconds=float(timing["intertrial_seconds"]))
        except KeyboardInterrupt:
            row["trial_status"] = "aborted"
            recorder.save_trial(row)
            return total, True
    return total, False


def run_session(config: dict, screens, participant_id: str, output_dir=None) -> Path:
    verify_demo_config(config)
    main_gen, schedule, training_schedule, train_seed, train_env_seed = make_schedules(config)
    main_env_seed = main_gen.seed + 1
    main_env = BanditEnvironment(config, seed=main_env_seed)
    train_env = BanditEnvironment(config, seed=train_env_seed) if training_schedule else None
    conditions = {str(c["id"]): c for c in config["design"]["conditions"]}
    # Wyłącznie trening: osobny darmowy warunek, niezależny od siatki 30 cen.
    training_conditions = {"training_free": {"id": "training_free",
                         "cue_accuracy": 1.0, "cue_cost": 0.0, "weight": 1}}

    with SessionRecorder(config, participant_id, main_gen.seed, main_env_seed,
                         schedule, output_dir=output_dir,
                         training_schedule=training_schedule,
                         training_plan_seed=train_seed,
                         training_environment_seed=train_env_seed) as recorder:
        print(f"Folder wyników: {recorder.folder}")
        try:
            if hasattr(screens, "set_phase"):
                screens.set_phase(None)
            show_instructions(config, screens)
            if training_schedule:
                screens.message(
                    f"TRENING — {len(training_schedule)} prób.\n"
                    "W treningu możesz pytać leśniczego za darmo.\n"
                    "Pieniądze z treningu nie będą doliczone do części głównej.\n"
                    "Pamiętaj: po LEWEJ stronie znajduje się bezpieczna ścieżka, po PRAWEJ - ryzykowna. \n"
                    "Naciśnij SPACJĘ, aby przejść dalej.",
                    continue_required=True,
                )
                train_total, interrupted = _run_phase(
                    phase="training", schedule=training_schedule,
                    env=train_env, conditions=training_conditions, config=config,
                    screens=screens, recorder=recorder,
                )
                if interrupted:
                    print("Przerwano podczas treningu. Zachowano zapisane próby.")
                    return recorder.folder
                screens.message(
                    "Koniec treningu.\n"
                    "W części głównej zaczynasz zbieranie pieniędzy od nowa.\n"
                    "Naciśnij SPACJĘ.",
                    continue_required=True,
                )
                if config["training"]["comprehension_check"]["enabled"]:
                    passed = run_comprehension_check(config, screens, recorder.folder)
                    if not passed:
                        print("Sprawdzian niezaliczony. Część główna nie została uruchomiona.")
                        return recorder.folder
            screens.message(
                f"CZĘŚĆ GŁÓWNA — {len(schedule)} prób.\n"
                "Od teraz wskazówki leśniczego są płatne.\n"
                "Przy każdej decyzji zobaczysz aktualną cenę.\n"
                "Twoim zadaniem jest zebrać jak najwięcej pieniędzy.\n"
                "Stan konta będzie widoczny na ekranie podczas prób.\n"
                "Naciśnij SPACJĘ, aby zacząć.",
                continue_required=True,
            )
            total, interrupted = _run_phase(
                phase="experiment", schedule=schedule,
                env=main_env, conditions=conditions, config=config,
                screens=screens, recorder=recorder,
            )
            if hasattr(screens, "set_phase"):
                screens.set_phase(None)  # nie pokazuj portfela na ekranie końcowym
            if not interrupted:
                screens.message(
                    "Koniec części głównej.\n"
                    "Dziękujemy za udział! Naciśnij SPACJĘ, aby zakończyć.",
                    continue_required=True,
                )
            else:
                print("Przerwano podczas części głównej. Zachowano zapisane próby.")
        except KeyboardInterrupt:
            print("Sesja przerwana między próbami. Zapisano dotychczasowe dane.")
        return recorder.folder

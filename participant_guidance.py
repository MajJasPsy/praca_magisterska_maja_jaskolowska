"""KROK 7 - wstepne instrukcje i PROTOTYP testu zrozumienia.

Instrukcja i pytania sa po polsku i oparte na config.yaml. Sa przykladowe:
przed badaniem wymagaja oceny merytorycznej i testow z uczestnikami.
Ten modul nie losuje prob ani nie wplywa na punktacje lub modele.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from text_labels import display_context

QUIZ_COLUMNS = ["attempt", "question_id", "question_text", "left_text", "right_text",
                "correct_side", "chosen_side", "is_correct", "rt_seconds"]


def _fmt(n):
    return f"{float(n):g}"


def instruction_pages(config: dict) -> list[str]:
    """Robocza instrukcja zgodna z trzema lasami, q=1 i przyciskami strzałek."""
    safe = _fmt(config["rewards"]["safe_points"])
    possible = ", ".join(_fmt(x) for x in config["rewards"]["risky_outcomes"])
    train = config["training"]["n_trials"] if config["training"]["enabled"] else 0
    main = config["experiment"]["n_trials"]
    return [
        "INSTRUKCJA (1/4)\n\n"
        "Jesteś poszukiwaczem skarbów i szukasz ukrytych pieniędzy "
        "za pomocą wykrywacza metalu. W każdej próbie trafiasz do jednego "
        "z trzech lasów: A, B albo C. Nie wiesz jednak, do którego.",
        "INSTRUKCJA (2/4)\n\n"
        "W każdym lesie możesz wybrać jedną z dwóch ścieżek. "
        f"LEWA ścieżka zawsze daje {safe} zł.\n "
        f"Możliwe znaleziska na PRAWEJ ścieżce to: {possible} zł. "
        "Ważne jest jednak to, że nie wszystkie kwoty pojawiają się równie często. "
        "Każdy las ma inne prawdopodobieństwa znalezienia poszczególnych kwot. "
        "Na przykład w jednym lesie częściej możesz znaleźć 12 zł, "
        "a w innym częściej wyjdziesz z pustymi rękoma.\n"
        "Sam możesz odkryć, jak opłacalny jest każdy z lasów.",
        "INSTRUKCJA (3/4)\n\n"
        "Przed wyborem ścieżki możesz zapytać leśniczego, "
        "w którym lesie jesteś. Leśniczy ZAWSZE mówi prawdę. "
        "W treningu leśniczy udziela wskazówek za darmo. "
        "W części głównej wskazówki są płatne, a ich cena jest podawana przy każdym pytaniu.\n"
        "Wybierając TAK, prosisz o wskazówkę; wybierając NIE, rezygnujesz z niej. "
        "Wyboru dotyczącego wskazówki możesz dokonać "
        "za pomocą odpowiedniej strzałki. "
        "Położenie odpowiedzi TAK i NIE może zmieniać się między próbami. "
        "Gdy nie zapytasz, nie dostaniesz informacji, w którym jesteś lesie.",
        "INSTRUKCJA (4/4)\n\n"
        "Następnie wybierasz ścieżkę strzałkami: LEWO oznacza "
        "ścieżkę bezpieczną, PRAWO - ryzykowną. "
        "Po wyborze zobaczysz znalezioną kwotę; "
        "w części głównej opłata za leśniczego będzie odjęta od Twojego bilansu. "
        + (f"Najpierw wykonasz {train} prób treningowych, potem {main} prób części głównej. "
           "Pieniądze z treningu nie przechodzą do części głównej. "
           if train else f"W części głównej wykonasz {main} prób. ")
        + "W części głównej na ekranie zobaczysz swój aktualny stan konta. "
          "Twoim celem jest zebranie jak największej sumy pieniędzy.",
    ]


def show_instructions(config: dict, screens) -> None:
    for page in instruction_pages(config):
        screens.message(page, continue_required=True)


def quiz_items(config: dict) -> list[dict]:
    """Kontrola REGUŁ, nie znajomości rzeczywistych rozkładów nagród."""
    safe = _fmt(config["rewards"]["safe_points"])
    wrong = _fmt(float(config["rewards"]["safe_points"]) + 1)
    return [
        dict(id="safe_constant", prompt="Co daje lewa ścieżka?",
             left=f"Zawsze {safe} zł", right=f"Zawsze {wrong} zł", correct="left"),
        dict(id="risky_context", prompt="Od czego zależą znaleziska na prawej ścieżce?",
             left="Od koloru strzałki", right="Od lasu", correct="right"),
        dict(id="forester_truth", prompt="Czy leśniczy może wskazać niewłaściwy las?",
             left="Tak", right="Nie", correct="right"),
        dict(id="forester_fee", prompt="Kiedy płacisz za wskazówkę w części głównej?",
             left="Zawsze", right="Gdy zapytasz", correct="right"),
        dict(id="skip_signal", prompt="Co widzisz, gdy nie zapytasz?",
             left="Brak nazwy lasu", right="Nazwę lasu", correct="left"),
        dict(id="path_keys", prompt="Jak wybierasz ścieżkę ryzykowną?",
             left="Strzałką w lewo", right="Strzałką w prawo", correct="right"),
    ]


def _write_row(folder: Path, row: dict) -> None:
    path = folder / "comprehension.csv"
    fresh = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUIZ_COLUMNS)
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _save_status(folder: Path, status: str, attempts: int, n_correct: int, n_questions: int) -> None:
    summary = dict(status=status, attempts=attempts,
                   correct_on_last_attempt=n_correct,
                   total_questions=n_questions,
                   fraction_correct=round(n_correct / n_questions, 4) if n_questions else None)
    (folder / "comprehension_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_comprehension_check(config: dict, screens, folder: Path) -> bool:
    """Uruchamia pytania i zapisuje wyniki; False blokuje czesc glowna.

    Wszystkie 6 pytan to test zrozumienia REGUL, nie znajomosci
    rozkladow nagrod. Wynik nie jest miernikiem uczenia sie modelu.
    """
    folder = Path(folder)
    settings = config["training"]["comprehension_check"]
    items = quiz_items(config)
    min_fraction = float(settings["min_correct_fraction"])
    max_attempts = int(settings["max_attempts"])
    timeout = float(settings.get("question_timeout_seconds", 60))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("question_timeout_seconds musi byc dodatni i skonczony")
    n_last = 0
    attempt = 0
    try:
        screens.message(
            "SPRAWDZENIE ZROZUMIENIA\n\n"
            "Odpowiedz na pytania o zasady zadania. "
            "Nie są to pytania o zapamiętane wyniki prób.\n"
            "Jeżeli nie osiągniesz wymaganego wyniku, możesz spróbować jeszcze raz.",
            continue_required=True,
        )
        for attempt in range(1, max_attempts + 1):
            n_correct = 0
            for index, q in enumerate(items, 1):
                title = f"Pytanie {index}/{len(items)} | Podejście {attempt}/{max_attempts}\n\n{q['prompt']}"
                key, rt = screens.choice(title, q["left"], q["right"], timeout)
                chosen = ("left" if key == screens.left_key else
                          "right" if key == screens.right_key else None)
                correct = chosen == q["correct"]
                n_correct += int(correct)
                _write_row(folder, {
                    "attempt": attempt, "question_id": q["id"],
                    "question_text": q["prompt"], "left_text": q["left"],
                    "right_text": q["right"], "correct_side": q["correct"],
                    "chosen_side": chosen or "", "is_correct": int(correct),
                    "rt_seconds": rt if rt is not None else "",
                })
            n_last = n_correct
            passed = n_correct / len(items) >= min_fraction
            screens.message(
                f"Wynik sprawdzenia zasad: {n_correct}/{len(items)} poprawnych odpowiedzi.\n\n"
                + ("Zaliczono. Możesz przejść do części głównej." if passed else
                   "Nie osiągnięto wymaganego wyniku. " +
                   ("Instrukcja zostanie powtórzona." if attempt < max_attempts else
                    "Sesja zakończy się przed częścią główną.")),
                continue_required=True,
            )
            if passed:
                _save_status(folder, "passed", attempt, n_correct, len(items))
                return True
            if attempt < max_attempts:
                show_instructions(config, screens)
    except KeyboardInterrupt:
        _save_status(folder, "aborted", attempt, n_last, len(items))
        raise
    _save_status(folder, "failed", attempt, n_last, len(items))
    return False

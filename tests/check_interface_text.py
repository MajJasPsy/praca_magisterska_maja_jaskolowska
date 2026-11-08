"""Kontrola zmian tekstowych bez uruchamiania okna PsychoPy."""
from pathlib import Path
from text_labels import display_context
from participant_guidance import instruction_pages
import yaml

folder = Path(__file__).resolve().parent
with (folder / "config.yaml").open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
source_session = (folder / "session_runner.py").read_text(encoding="utf-8")
source_demo = (folder / "demo_one_trial.py").read_text(encoding="utf-8")
for src in (source_session, source_demo):
    assert 'else "BEZPIECZNA"' in src
    assert 'BEZPIECZNA (+' not in src
    assert 'BEZPIECZNA\\n+' not in src
    assert 'KUP WSKAZÓWKĘ' in src
    assert 'BEZ WSKAZÓWKI' in src
assert display_context("high", cfg) == "A"
assert display_context("neutral", cfg) == "B"
assert display_context("low", cfg) == "C"
assert f"{float(cfg['rewards']['safe_points']):g} punktów" in instruction_pages(cfg)[1]
assert cfg['confidence']['enabled'] is False, "Nie włączamy pomiaru pewności."
print("OK — polskie etykiety kontekstów i wskazówki.")
print("OK — przy wyborze opcji bezpiecznej nie ma kwoty nagrody.")
print("OK — wartość bezpiecznej nagrody pozostaje w instrukcji.")
print("OK — pomiar pewności jest wyłączony.")

# Kontrola kroju pisma i niewyswietlania instrukcji klawisza Esc.
assert 'display.get("font", "Segoe UI")' in source_demo
assert 'Esc — przerwij' not in source_demo
assert 'self.quit_key' in source_demo  # Esc pozostaje przyciskiem awaryjnym.
print("OK — czcionka Segoe UI i brak informacji o Esc na ekranie.")

# Kontrola nazw widocznych dla badanego po uproszczeniu ekranów.
source_guidance = (folder / "participant_guidance.py").read_text(encoding="utf-8")
source_dialog = (folder / "demo_session.py").read_text(encoding="utf-8")
assert 'ZADANIE DECYZYJNE' not in source_demo
assert 'WERSJA TESTOWA' not in source_demo
assert 'self._base("INFORMACJA")' not in source_demo
assert 'self._base()' in source_demo
assert 'WERSJA TESTOWA' not in source_guidance
assert 'wersja testowa' not in source_dialog.lower()
assert all(p.splitlines()[0] == f"INSTRUKCJA ({i}/4)"
           for i, p in enumerate(instruction_pages(cfg), start=1))
print("OK — bez nagłówków technicznych i bez etykiety «INFORMACJA».")
print("OK — numeracja instrukcji 1/4–4/4 pozostała.")

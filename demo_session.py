"""KROK 5: testowa sesja 16 prob w PsychoPy z zapisem danych do CSV.

Wymaga oryginalnych plikow z poprzednich krokow w tym samym folderze:
config.yaml, trial_generator.py, environment.py, demo_one_trial.py.
Nie zmienia main.py i nie nadaje sie jeszcze do badania docelowego.
"""
from pathlib import Path
import yaml
from psychopy import core, gui

from demo_one_trial import PreviewScreens
from session_runner import run_session, verify_demo_config

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    verify_demo_config(config)

    dialog = gui.Dlg(title="Eksperyment")
    dialog.addText("Wpisz anonimowy identyfikator testowy (bez nazwiska):")
    dialog.addField("ID uczestnika:", initial="test_001")
    response = dialog.show()
    if dialog.OK is False or response is None:
        return
    participant_id = str(response[0]).strip()

    screens = None
    try:
        screens = PreviewScreens(config)
        folder = run_session(config, screens, participant_id=participant_id,
                             output_dir=ROOT / config["data"]["output_directory"])
        print("OK — zapisano sesję testową w:", folder)
    except KeyboardInterrupt:
        print("Przerwano przed rozpoczęciem lub zakończeniem sesji.")
    finally:
        if screens is not None:
            screens.close()
        core.quit()


if __name__ == "__main__":
    main()

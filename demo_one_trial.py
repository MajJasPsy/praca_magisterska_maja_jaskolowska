"""Ekrany prototypu trzech lasów w PsychoPy.

- Szara plansza leśniczego i plansza wyniku rysowane w PsychoPy.
- Gotowe obrazy z images/ pokazują WYBÓR ścieżki, nie osobną wskazówkę.
- Wskazówka jest zawsze prawdziwa (sprawdzane w konfiguracji).
- Używamy strzałek w lewo / w prawo; Esc działa awaryjnie, ale nie jest opisany.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from psychopy import core, visual
from psychopy.hardware import keyboard

from environment import BanditEnvironment
from trial_generator import TrialGenerator
from text_labels import display_context

ROOT = Path(__file__).resolve().parent


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def money(value):
    """Tekst kwoty z przecinkiem, np. 1,40 zł. Zero dodatkowych obliczeń."""
    return f"{float(value):.2f}".replace(".", ",") + " zł"


class PreviewScreens:
    """Ekrany + klawiatura. Nie losuje nagród i nie aktualizuje przekonań."""

    def __init__(self, config):
        self.config = config
        display = config["display"]
        controls = config["controls"]
        self.left_key = controls["left_key"]
        self.right_key = controls["right_key"]
        self.continue_key = controls["continue_key"]
        self.quit_key = controls["quit_key"]
        if [self.left_key, self.right_key] != ["left", "right"]:
            raise ValueError("Nowe plansze wymagają left_key: left i right_key: right")
        if len({self.left_key, self.right_key, self.continue_key, self.quit_key}) != 4:
            raise ValueError("Klawisze sterowania muszą być różne")
        if display["units"] != "height":
            raise ValueError("Interfejs wymaga display.units: height")

        self.font = display.get("font", "Segoe UI")
        self.gray = display.get("forest_gray", "#898989")
        self.wallet_caption = display.get("wallet_caption", "Stan konta:")
        self.phase = None
        self.balance = 0.0
        # Kontrola grafik PRZED otwarciem okna PsychoPy: przy braku PNG
        # wyświetlamy prosty błąd zamiast pozostawiać otwarte okno.
        image_dir = ROOT / config.get("images", {}).get("directory", "images")
        mapping = config.get("images", {}).get("forest_files", {
            "A": "context_1.png", "B": "context_2.png", "C": "context_3.png",
            "unknown": "context_null.png",
        })
        # Portfel ma być widoczny tylko podczas 90 prób części głównej.
        wallet_path = image_dir / config.get("images", {}).get("wallet_file", "wallet.png")
        if not wallet_path.is_file():
            raise FileNotFoundError(
                f"Brakuje {wallet_path}. Umieść przygotowany plik wallet.png w folderze images/"
            )
        with wallet_path.open("rb") as wallet_handle:
            wallet_header = wallet_handle.read(24)
        if len(wallet_header) < 24 or wallet_header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Portfel musi być plikiem PNG: {wallet_path}")
        wallet_dimensions = (int.from_bytes(wallet_header[16:20], "big"),
                             int.from_bytes(wallet_header[20:24], "big"))
        if min(wallet_dimensions) <= 0:
            raise ValueError("Niepoprawne wymiary grafiki wallet.png")
        checked = {}
        for key in ("A", "B", "C", "unknown"):
            path = image_dir / mapping[key]
            if not path.is_file():
                raise FileNotFoundError(f"Brakuje grafiki: {path}")
            with path.open("rb") as f:
                header = f.read(24)
            if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Nieprawidłowy obraz PNG: {path}")
            checked[key] = (path, int.from_bytes(header[16:20], "big"),
                            int.from_bytes(header[20:24], "big"))

        self.win = visual.Window(
            size=tuple(display["window_size_pixels"]),
            fullscr=bool(display["fullscreen"]), units="height",
            color=self.gray, colorSpace="rgb", allowGUI=not bool(display["fullscreen"]),
        )
        self.kb = keyboard.Keyboard()
        self.images = {}
        screen_width, screen_height = self.win.size
        for key, (path, img_width, img_height) in checked.items():
            scale = min(screen_width / img_width, screen_height / img_height)
            size = (img_width * scale / screen_height, img_height * scale / screen_height)
            self.images[key] = visual.ImageStim(self.win, image=str(path), size=size,
                                                pos=(0, 0), interpolate=True)
        # Lewy górny róg: na planszach z leśniczym nazwa lasu
        # jest w dymku w prawym górnym rogu — nie wolno go zasłonić.
        screen_ratio = screen_width / screen_height
        wallet_width = min(0.54, 0.32 * screen_ratio)
        wallet_height = wallet_width * wallet_dimensions[1] / wallet_dimensions[0]
        wallet_center = (-screen_ratio / 2 + 0.015 + wallet_width / 2,
                         0.5 - 0.014 - wallet_height / 2)
        self.wallet_center = wallet_center
        self.wallet_width = wallet_width
        self.wallet = visual.ImageStim(self.win, image=str(wallet_path),
                                       size=(wallet_width, wallet_height),
                                       pos=wallet_center, interpolate=True)

    def close(self):
        self.win.close()

    def _text(self, text, *, pos=(0, 0), size=0.044, color="black", width=1.5,
              bold=False, alignText="center"):
        visual.TextStim(self.win, text=str(text), font=self.font, height=size,
                        color=color, pos=pos, wrapWidth=width, bold=bold,
                        alignText=alignText).draw()

    def _rect(self, pos, size, color):
        visual.Rect(self.win, pos=pos, width=size[0], height=size[1],
                    fillColor=color, lineColor=color).draw()

    def _gray(self):
        self.win.color = self.gray

    def set_phase(self, phase):
        """None = instrukcja/quiz; training = bez portfela; experiment = portfel."""
        if phase not in (None, "training", "experiment"):
            raise ValueError(f"Nieznana faza: {phase}")
        self.phase = phase

    def set_balance(self, amount):
        """Przyjmuje aktualne saldo obliczone przez session_runner.py."""
        self.balance = round(float(amount), 2)

    def _draw_wallet(self):
        if self.phase != "experiment":
            return
        self.wallet.draw()
        wx, wy = self.wallet_center
        ww = self.wallet_width
        # Teksty dodajemy w PsychoPy na pustej prawej części grafiki.
        # Dzięki temu kwota nigdy nie jest wpisana na stałe w PNG.
        text_x = wx + 0.205 * ww
        self._text(self.wallet_caption, pos=(text_x, wy + 0.019),
                   size=0.024, color="#452514", bold=True, width=ww * 0.52)
        self._text(money(self.balance), pos=(text_x, wy - 0.032),
                   size=0.039, color="#452514", bold=True, width=ww * 0.52)

    def _listen(self, allowed: list[str], seconds: Optional[float]):
        keys = list(set(allowed) | {self.quit_key})
        while seconds is None or self.kb.clock.getTime() < float(seconds):
            for press in self.kb.getKeys(keyList=keys, waitRelease=False, clear=True):
                if press.rt is None or float(press.rt) < 0:
                    continue
                if press.name == self.quit_key:
                    raise KeyboardInterrupt("Przerwano przez Esc")
                if press.name in allowed:
                    return press.name, float(press.rt)
            core.wait(0.005)
        return None, None

    def _show(self):
        # First visible frame: clear old presses and start the response clock.
        self.win.callOnFlip(self.kb.clock.reset)
        self.win.callOnFlip(self.kb.clearEvents)
        self.win.flip()

    def choose_forester(self, cost: float, timeout: float, buy_cue_side: str = "right"):
        """Pierwszy wybór. Koszt jest DYNAMICZNY z konkretnej próby."""
        self._gray()
        training = self.phase == "training"
        self._text("Czy chcesz poprosić leśniczego\no wskazówkę?",
                   pos=(0, 0.30 if training else 0.23),
                   size=0.061, bold=True, width=1.72)
        if not training:
            self._rect((0, 0.04), (0.72, 0.16), "white")
            self._text(f"Koszt: {money(cost)}", pos=(0, 0.04), size=0.055, bold=True)
        if buy_cue_side not in {"left", "right"}:
            raise ValueError("buy_cue_side musi wynosić left lub right")
        button_y = -0.11 if training else -0.22
        for side, x, label in (("left", -0.28, "← strzałka w lewo"),
                               ("right", 0.28, "strzałka w prawo →")):
            is_yes = side == buy_cue_side
            self._rect((x, button_y), (0.45, 0.19), "#1DB44C" if is_yes else "#B10021")
            self._text("TAK" if is_yes else "NIE", pos=(x, button_y),
                       size=0.075, color="white", bold=True)
            self._text(label, pos=(x, button_y - 0.145), size=0.025)
        self._draw_wallet()
        self._show()
        return self._listen([self.left_key, self.right_key], float(timeout))

    def choose_path(self, forest: str, timeout: float):
        """Drugi wybór: GOTOWA ilustracja w tle; żadnych nakładanych etykiet."""
        self._gray()
        if forest not in self.images:
            raise ValueError(f"Nieznany obraz lasu: {forest}")
        self.images[forest].draw()
        self._draw_wallet()
        self._show()
        return self._listen([self.left_key, self.right_key], float(timeout))

    def show_found(self, reward: float, seconds: float):
        """Feedback: TYLKO wylosowana nagroda, nie koszt ani suma netto."""
        self._gray()
        self._rect((0, 0), (1.12, 0.25), "white")
        self._text(f"Znaleziono: {money(reward)}", pos=(0, 0),
                   size=0.075, bold=True, width=1.05)
        self._draw_wallet()
        self._show()
        self._listen([], float(seconds))

    def choice(self, title, left, right, timeout):
        """Pozostałe pytania (quiz) — nie jest to ekran leśniczego."""
        self._gray()
        lines = [line.strip() for line in str(title).splitlines() if line.strip()]
        title_text = lines[0] if lines else "Pytanie"
        question = " ".join(lines[1:]) if len(lines) > 1 else title_text
        self._text(title_text, pos=(0, 0.37), size=0.029, width=1.65)
        self._text(question, pos=(0, 0.19), size=0.047, width=1.60, bold=True)
        for x, label, sym in ((-0.30, left, "←"), (0.30, right, "→")):
            self._rect((x, -0.095), (0.54, 0.21), "white")
            self._text(str(label), pos=(x, -0.095), size=0.027, width=0.49)
            self._text(sym, pos=(x, -0.27), size=0.040)
        self._show()
        return self._listen([self.left_key, self.right_key], float(timeout))

    def message(self, text, seconds=None, continue_required=False):
        self._gray()
        if str(text).strip() == "+":
            self._text("+", size=0.067)
        else:
            lines = str(text).splitlines()
            heading = lines[0]
            body = "\n".join(lines[1:]).strip()
            self._text(heading, pos=(0, 0.275), size=0.042, bold=True)
            if body:
                size = 0.028 if len(body) > 200 else 0.034
                self._text(body, pos=(0, -0.04), size=size, width=1.56)
            if continue_required:
                self._text("Spacja - dalej", pos=(0, -0.405), size=0.025)
        self._draw_wallet()
        self._show()
        self._listen([self.continue_key] if continue_required else [],
                     None if continue_required else float(seconds if seconds is not None else 1))


def run_demo(config):
    if config["experiment"]["mode"] != "test":
        raise ValueError("Uruchom demo tylko przy experiment.mode: test")
    from session_runner import verify_demo_config
    verify_demo_config(config)
    generator = TrialGenerator(config)
    planned = generator.generate()[0]
    env = BanditEnvironment(config, seed=generator.seed + 1)
    cond = next(c for c in config["design"]["conditions"] if c["id"] == planned.condition_id)
    trial = env.prepare_trial(cond)
    screens = None
    try:
        screens = PreviewScreens(config)
        screens.set_phase(None)
        screens.message("Próba demonstracyjna\n\nWybieraj strzałkami w lewo lub w prawo.",
                        continue_required=True)
        screens.message("+", seconds=config["timing"]["fixation_seconds"])
        screens.set_phase("experiment")
        screens.set_balance(float(config["scoring"]["starting_points"]))
        key1, rt1 = screens.choose_forester(trial.cue_cost,
                         config["timing"]["cue_decision_timeout_seconds"],
                         buy_cue_side=planned.buy_cue_side)
        if key1 is None:
            screens.message("Brak odpowiedzi", seconds=1.5)
            return
        bought = key1 == planned.buy_cue_side
        forest = display_context(trial.cue_signal, config) if bought else "unknown"
        screens.set_balance(float(config["scoring"]["starting_points"]) -
                            (trial.cue_cost if bought else 0.0))
        key2, rt2 = screens.choose_path(forest,
                         config["timing"]["reward_decision_timeout_seconds"])
        if key2 is None:
            screens.message("Brak odpowiedzi", seconds=1.5)
            return
        result = env.resolve_trial(trial, buy_cue=bought,
                                    reward_choice="safe" if key2 == "left" else "risky")
        screens.set_balance(float(config["scoring"]["starting_points"]) + result["net_points"])
        screens.show_found(result["reward_points"], config["timing"]["reward_feedback_seconds"])
        print("DEMO — stan do kontroli (niewidoczny dla uczestnika):",
              "las", trial.context, "zapytano", bought, "nagroda", result["reward_points"],
              "koszt", result["cue_cost_paid"], "netto", result["net_points"],
              "RT", rt1, rt2)
    finally:
        if screens is not None:
            screens.close()
        core.quit()


if __name__ == "__main__":
    run_demo(load_config())

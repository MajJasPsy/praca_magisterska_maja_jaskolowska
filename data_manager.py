"""Bezpieczny zapis sesji testowej do CSV; bez zaleznosci od PsychoPy.

KAZDA sesja ma odrebny folder, wiec zadne poprzednie wyniki nie zostana nadpisane.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4


COLUMNS = [
    "participant_id", "session_id", "config_version", "phase",
    "trial_index", "block_index", "in_block_index", "condition_id",
    "cue_accuracy", "cue_cost_offered", "buy_cue_side", "risky_side",
    "planned_context", "true_context", "cue_signal_generated", "risky_reward_generated", "forest_image_displayed",
    "cue_choice", "cue_signal_observed", "rt_cue_seconds",
    "reward_choice", "rt_reward_seconds", "reward_points", "cue_cost_paid",
    "net_points", "running_total", "trial_status",
]


def safe_participant_id(value: str) -> str:
    """Identyfikator anonimowy, bez nazwisk, maili ani pelnych sciezek."""
    if not isinstance(value, str):
        raise ValueError("participant_id musi byc tekstem.")
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value):
        raise ValueError("Identyfikator moze zawierac tylko litery A-Z, cyfry, _ lub -, max 40 znakow.")
    return value


class SessionRecorder:
    def __init__(self, config: dict, participant_id: str, plan_seed: int,
                 environment_seed: int, schedule: list, output_dir=None,
                 training_schedule=None, training_plan_seed=None,
                 training_environment_seed=None):
        self.config = config
        training_schedule = list(training_schedule or [])
        self.participant_id = safe_participant_id(participant_id)
        self.session_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        root = Path(output_dir) if output_dir is not None else (
            Path(__file__).resolve().parent / config["data"]["output_directory"]
        )
        root.mkdir(parents=True, exist_ok=True)
        self.folder = root / f"{self.participant_id}_{self.session_id}"
        self.folder.mkdir(exist_ok=False)
        self.csv_path = self.folder / "trials.csv"
        self._handle = None
        self._writer = None

        if bool(config["data"].get("save_config_snapshot", True)):
            (self.folder / "config_snapshot.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        if bool(config["data"].get("save_generated_trial_schedule", True)):
            trial_rows = ([dict(phase="training", **asdict(x)) for x in training_schedule]
                          + [dict(phase="experiment", **asdict(x)) for x in schedule])
            if trial_rows:
                with (self.folder / "planned_trials.csv").open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=list(trial_rows[0]))
                    writer.writeheader()
                    writer.writerows(trial_rows)

        meta = {
            "participant_id": self.participant_id, "session_id": self.session_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "plan_seed": plan_seed, "environment_seed": environment_seed,
            "config_version": config["experiment"]["version"],
            "n_planned_trials": len(schedule),
            "n_planned_training_trials": len(training_schedule),
            "training_trials_per_context": config["training"].get("trials_per_context"),
            "training_plan_seed": training_plan_seed,
            "training_environment_seed": training_environment_seed,
            "comprehension_check_enabled": bool(config["training"]["comprehension_check"]["enabled"]),
            "note": "PROTOTYP TESTOWY; instrukcje i quiz sa robocze, brak pomiaru pewnosci.",
        }
        (self.folder / "session_metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._handle = self.csv_path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._handle, fieldnames=COLUMNS, extrasaction="raise")
        self._writer.writeheader()
        self._sync()

    def _sync(self):
        self._handle.flush()
        if bool(self.config["data"].get("save_each_trial", True)):
            os.fsync(self._handle.fileno())

    def save_trial(self, row: dict):
        if self._handle is None or self._handle.closed:
            raise RuntimeError("Plik wynikow jest zamkniety.")
        unknown = set(row) - set(COLUMNS)
        if unknown:
            raise ValueError(f"Nieznane kolumny w danych: {sorted(unknown)}")
        full_row = {key: row.get(key, None) for key in COLUMNS}
        full_row["participant_id"] = self.participant_id
        full_row["session_id"] = self.session_id
        full_row["config_version"] = self.config["experiment"]["version"]
        full_row["phase"] = row.get("phase", "experiment")
        if full_row["phase"] not in {"training", "experiment"}:
            raise ValueError("Nieznana faza: " + str(full_row["phase"]))
        self._writer.writerow(full_row)
        self._sync()

    def close(self):
        if self._handle is not None and not self._handle.closed:
            self._sync()
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

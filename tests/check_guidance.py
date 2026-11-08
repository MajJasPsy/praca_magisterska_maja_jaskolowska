"""Krok 7: kontrola instrukcji, quizu, zapisow i blokady przed czescia glowna.

Uruchamiaj w PyCharm: Run 'check_guidance'. Nie wymaga PsychoPy.
"""
from copy import deepcopy
from pathlib import Path
import csv
import json
import tempfile
import yaml

from session_runner import run_session
from participant_guidance import instruction_pages, quiz_items

ROOT = Path(__file__).resolve().parent
with (ROOT / 'config.yaml').open(encoding='utf-8') as f:
    config = yaml.safe_load(f)


def rows(folder, filename):
    with (folder / filename).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class FakeScreens:
    def __init__(self, config, quiz_strategy='correct'):
        self.left_key = config['controls']['left_key']
        self.right_key = config['controls']['right_key']
        self.strategy = quiz_strategy
        self.qa_count = 0
        self.messages = []
        self.answers = {q['prompt']:q['correct'] for q in quiz_items(config)}

    def message(self, text, seconds=None, continue_required=False):
        self.messages.append(str(text))

    def choice(self, title, left, right, timeout):
        if 'Decyzja 1' in title:
            return self.left_key, 0.31
        if 'Decyzja 2' in title:
            return self.right_key, 0.45
        if title.startswith('Pytanie '):
            question = next((key for key in self.answers if key in title), None)
            assert question is not None, title
            self.qa_count += 1
            attempt = int(title.split('Podejście ')[1].split('/')[0])
            correct = self.answers[question]
            if self.strategy == 'wrong' or (self.strategy == 'retry' and attempt == 1):
                chosen = 'right' if correct == 'left' else 'left'
            else:
                chosen = correct
            return (self.left_key if chosen == 'left' else self.right_key), 0.62
        raise AssertionError('Nieznany ekran w testach: ' + str(title))


assert len(instruction_pages(config)) == 4
assert len(quiz_items(config)) == 6
assert f"{float(config['rewards']['safe_points']):g} punktów" in instruction_pages(config)[1]

with tempfile.TemporaryDirectory() as temp:
    enabled = deepcopy(config)
    enabled['training']['comprehension_check']['enabled'] = True
    sc = FakeScreens(enabled)
    folder = run_session(enabled, sc, 'test_quiz_pass', output_dir=temp)
    trials = rows(folder, 'trials.csv')
    quiz = rows(folder, 'comprehension.csv')
    summary = json.loads((folder/'comprehension_summary.json').read_text(encoding='utf-8'))
    assert len(trials) == enabled['training']['n_trials'] + enabled['experiment']['n_trials']
    assert len(quiz) == 6 and all(int(r['is_correct']) == 1 for r in quiz)
    assert summary['status'] == 'passed' and summary['attempts'] == 1
    assert sc.qa_count == 6
    assert all(r['phase'] == 'training' for r in trials[:enabled['training']['n_trials']]); assert all(r['phase'] == 'experiment' for r in trials[enabled['training']['n_trials']:] )

    sf = FakeScreens(enabled, quiz_strategy='wrong')
    failed_folder = run_session(enabled, sf, 'test_quiz_fail', output_dir=temp)
    failed_trials = rows(failed_folder, 'trials.csv')
    failed_quiz = rows(failed_folder, 'comprehension.csv')
    failed_summary = json.loads((failed_folder/'comprehension_summary.json').read_text(encoding='utf-8'))
    assert len(failed_trials) == enabled['training']['n_trials']
    assert len(failed_quiz) == 12
    assert failed_summary['status'] == 'failed' and failed_summary['attempts'] == 2
    assert all(r['phase'] == 'training' for r in failed_trials)

    sr = FakeScreens(enabled, quiz_strategy='retry')
    retried_folder = run_session(enabled, sr, 'test_quiz_retry', output_dir=temp)
    retried_trials = rows(retried_folder, 'trials.csv')
    retried_summary = json.loads((retried_folder/'comprehension_summary.json').read_text(encoding='utf-8'))
    assert len(retried_trials) == enabled['training']['n_trials'] + enabled['experiment']['n_trials']
    assert retried_summary['status'] == 'passed' and retried_summary['attempts'] == 2
    assert len(rows(retried_folder, 'comprehension.csv')) == 12

    disabled = deepcopy(config)
    disabled["training"]["comprehension_check"]["enabled"] = False
    sd = FakeScreens(disabled)
    old_folder = run_session(disabled, sd, 'test_quiz_disabled', output_dir=temp)
    assert len(rows(old_folder, 'trials.csv')) == disabled['training']['n_trials'] + disabled['experiment']['n_trials']
    assert not (old_folder/'comprehension.csv').exists()
    # Wlaczony quiz nie zmienia ani planu, ani nagrod w eksperymencie.
    with (folder/'planned_trials.csv').open('rb') as f:
        plan_enabled = f.read()
    with (old_folder/'planned_trials.csv').open('rb') as f:
        plan_disabled = f.read()
    assert plan_enabled == plan_disabled
    for a,b in zip(rows(folder,'trials.csv'), rows(old_folder,'trials.csv')):
        for key in ('phase','condition_id','true_context','cue_signal_generated',
                    'risky_reward_generated','buy_cue_side','risky_side'):
            assert a[key] == b[key], (key, a[key], b[key])

print('OK - instrukcje korzystaja z parametrow YAML.')
print('OK - quiz zaliczony: 6 pytan; czesc glowna rozpoczeta.')
print('OK - quiz niezaliczony: 2 podejscia; czesc glowna zablokowana.')
print('OK - ponowne podejscie, zapis quizu i tryb bez quizu.')
print('OK - plan prob i generowanie nagrod nie zmieniaja sie po wlaczeniu quizu.')
print('KROK 7 - TESTY ZAKONCZONE POMYSLNIE.')

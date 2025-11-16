import importlib.util
from pathlib import Path
import types

PAGE_PATH = Path(__file__).resolve().parents[1] / 'finance_dashboard' / 'pages' / '8_🔁_Recurring_Payments.py'


def _load_page_module():
    spec = importlib.util.spec_from_file_location('recurring_page_test', PAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_plan_data_fallback(monkeypatch):
    module = _load_page_module()
    monkeypatch.setattr(module, '_plan_load_impl', None, raising=False)
    data = module._load_plan_data()
    assert data == {'selection': [], 'overrides': {}}


def test_save_plan_data_fallback(monkeypatch):
    module = _load_page_module()
    monkeypatch.setattr(module, '_plan_save_impl', None, raising=False)
    module._save_plan_data(['A'], {'A': {'avg': 10.0}})  # should not raise


def test_plan_state_initialization_uses_storage(monkeypatch):
    module = _load_page_module()
    dummy_state = {}
    monkeypatch.setattr(module, 'st', types.SimpleNamespace(session_state=dummy_state))
    monkeypatch.setattr(
        module,
        '_plan_load_impl',
        lambda: {'selection': ['A'], 'overrides': {'A': {'avg': 12.0}}},
        raising=False,
    )
    module._ensure_plan_state()
    assert dummy_state['confirmed_recurring'] == ['A']
    assert dummy_state['plan_overrides'] == {'A': {'avg': 12.0}}


def test_plan_state_persistence(monkeypatch):
    module = _load_page_module()
    captured = {}
    dummy_state = {'confirmed_recurring': ['B'], 'plan_overrides': {'B': {'avg': 5.0}}}
    monkeypatch.setattr(module, 'st', types.SimpleNamespace(session_state=dummy_state))

    def fake_save(selection, overrides):
        captured['selection'] = selection
        captured['overrides'] = overrides

    monkeypatch.setattr(module, '_plan_save_impl', fake_save, raising=False)
    module._persist_plan_state()
    assert captured['selection'] == ['B']
    assert captured['overrides'] == {'B': {'avg': 5.0}}


def test_rerun_prefers_streamlit_rerun(monkeypatch):
    module = _load_page_module()
    called = {}

    def fake_rerun():
        called['method'] = 'rerun'

    st_mock = types.SimpleNamespace(rerun=fake_rerun)
    monkeypatch.setattr(module, 'st', st_mock, raising=False)
    module._rerun()
    assert called['method'] == 'rerun'


def test_rerun_falls_back_to_experimental(monkeypatch):
    module = _load_page_module()
    called = {}

    def fake_experimental():
        called['method'] = 'experimental'

    st_mock = types.SimpleNamespace(experimental_rerun=fake_experimental)
    monkeypatch.setattr(module, 'st', st_mock, raising=False)
    module._rerun()
    assert called['method'] == 'experimental'

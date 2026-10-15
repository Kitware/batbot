"""Execution-provider selection and inference progress regressions."""

from types import SimpleNamespace

import numpy as np
import pytest

from batbot.classifier import inference


@pytest.fixture
def runtime(monkeypatch):
    import onnxruntime as ort

    calls = []
    session = SimpleNamespace(
        get_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    state = SimpleNamespace(
        available=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        calls=calls,
        session=session,
        options=None,
        provider_options=None,
    )

    def preload():
        calls.append('preload')
        print('Runtime library diagnostics')

    def create(path, *, sess_options, providers, provider_options=None):
        assert isinstance(sess_options, ort.SessionOptions)
        state.options = sess_options
        state.provider_options = provider_options
        calls.append(('session', path, providers, sess_options))
        return session

    monkeypatch.delenv('BATBOT_REDUCED', raising=False)
    monkeypatch.setattr(ort, 'disable_telemetry_events', lambda: None)
    monkeypatch.setattr(ort, 'get_available_providers', lambda: state.available)
    monkeypatch.setattr(ort, 'preload_dlls', preload, raising=False)
    monkeypatch.setattr(ort, 'InferenceSession', create)
    return state


def test_cuda_libraries_are_preloaded_before_creating_session(runtime, capsys):
    session = inference._create_session('model.onnx')

    assert session is runtime.session
    assert runtime.calls == [
        'preload',
        (
            'session',
            'model.onnx',
            ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            runtime.options,
        ),
    ]
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'Runtime library diagnostics' in captured.err


def test_cpu_selection_skips_gpu_preloading(runtime):
    runtime.session.get_providers = lambda: ['CPUExecutionProvider']

    inference._create_session('model.onnx', providers=['CPUExecutionProvider'])

    assert runtime.calls == [('session', 'model.onnx', ['CPUExecutionProvider'], runtime.options)]


def test_automatic_cpu_selection_needs_no_gpu_runtime(runtime):
    runtime.available = ['CPUExecutionProvider']
    runtime.session.get_providers = lambda: ['CPUExecutionProvider']

    inference._create_session('model.onnx')

    assert runtime.calls == [('session', 'model.onnx', ['CPUExecutionProvider'], runtime.options)]
    assert runtime.options.intra_op_num_threads == 0
    assert runtime.options.inter_op_num_threads == 0


def test_explicit_cuda_rejects_cpu_only_installation(runtime):
    runtime.available = ['CPUExecutionProvider']

    with pytest.raises(RuntimeError, match=r'not available.*onnxruntime-gpu\[cuda,cudnn\]'):
        inference._create_session('model.onnx', providers=['CUDAExecutionProvider'])

    assert runtime.calls == []


@pytest.mark.parametrize('providers', [None, ['CUDAExecutionProvider', 'CPUExecutionProvider']])
def test_automatic_cuda_fallback_warns_with_installation_guidance(runtime, providers):
    runtime.session.get_providers = lambda: ['CPUExecutionProvider']

    with pytest.warns(RuntimeWarning, match='CUDAExecutionProvider failed.*Falling back to CPU'):
        session = inference._create_session('model.onnx', providers=providers)

    assert session is runtime.session


def test_explicit_provider_priority_allows_cpu_fallback_when_cuda_is_absent(runtime):
    runtime.available = ['CPUExecutionProvider']
    runtime.session.get_providers = lambda: ['CPUExecutionProvider']

    with pytest.warns(RuntimeWarning, match='not available.*Falling back to CPU'):
        session = inference._create_session(
            'model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

    assert session is runtime.session
    assert runtime.calls == [('session', 'model.onnx', ['CPUExecutionProvider'], runtime.options)]


def test_explicit_cuda_never_silently_falls_back(runtime):
    runtime.session.get_providers = lambda: ['CPUExecutionProvider']

    with pytest.raises(RuntimeError, match='CUDAExecutionProvider failed to initialize'):
        inference._create_session('model.onnx', providers=['CUDAExecutionProvider'])


def test_older_runtime_without_preloader_can_use_system_libraries(monkeypatch, runtime):
    import onnxruntime as ort

    monkeypatch.delattr(ort, 'preload_dlls')

    assert inference._create_session('model.onnx') is runtime.session
    assert len(runtime.calls) == 1


def test_reduced_threads_do_not_print_into_cli_json(monkeypatch, runtime, capsys):
    monkeypatch.setenv('BATBOT_REDUCED', '1')
    runtime.session.get_providers = lambda: ['CPUExecutionProvider']

    inference._create_session('model.onnx', providers=['CPUExecutionProvider'])

    options = runtime.calls[0][3]
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 2
    assert capsys.readouterr().out == ''


def test_coreml_selection_uses_native_provider_without_cuda_preloading(runtime):
    runtime.available = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    runtime.session.get_providers = lambda: runtime.available

    session = inference._create_session('model.onnx', providers=['CoreMLExecutionProvider'])

    assert session is runtime.session
    assert runtime.calls == [
        ('session', 'model.onnx', ['CoreMLExecutionProvider'], runtime.options)
    ]
    assert runtime.provider_options == [
        {'ModelFormat': 'MLProgram', 'RequireStaticInputShapes': '1'}
    ]


def test_coreml_selection_reports_unsupported_platform(runtime):
    runtime.available = ['CPUExecutionProvider']
    with pytest.raises(RuntimeError, match='CoreMLExecutionProvider is unavailable'):
        inference._create_session('model.onnx', providers=['CoreMLExecutionProvider'])


def test_coreml_selection_rejects_silent_cpu_fallback(runtime):
    runtime.available = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    runtime.session.get_providers = lambda: ['CPUExecutionProvider']
    with pytest.raises(RuntimeError, match='CoreMLExecutionProvider failed to initialize'):
        inference._create_session('model.onnx', providers=['CoreMLExecutionProvider'])


def test_prediction_respects_fixed_coreml_batch_shape():
    calls = []

    def run(_, inputs):
        values = inputs['input']
        calls.append(len(values))
        return [values[:, 0, 0, :2].astype(np.float32)]

    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name='input', shape=[1, 224, 224, 3])],
        run=run,
    )
    windows = np.stack([np.full((224, 224, 3), value, dtype=np.uint8) for value in range(3)])

    scores = inference._predict_windows(windows, session, batch_size=10)

    assert calls == [1, 1, 1]
    np.testing.assert_array_equal(scores, [[1, 1]])


@pytest.mark.parametrize('num_workers', [1, 2])
@pytest.mark.parametrize('show_progress', [True, False])
def test_inference_progress_only_counts_completed_predictions(
    monkeypatch, num_workers, show_progress
):
    calls = []
    bars = []

    class Progress:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.count = 0
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, count):
            self.count += count
            assert self.count <= len(calls)

    def run(_, inputs):
        calls.append(True)
        return [np.ones((len(inputs['input']), 2), dtype=np.float32)]

    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name='input')],
        run=run,
    )
    windows = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    monkeypatch.setattr(inference.tqdm, 'tqdm', Progress)

    outputs = list(
        inference.predict(
            [(windows, 'mobilenet')] * 3,
            sessions={'mobilenet': session},
            num_workers=num_workers,
            total=3,
            progress=show_progress,
        )
    )

    assert len(outputs) == 3
    assert bars[0].count == bars[0].options['total'] == 3
    assert bars[0].options['disable'] is not show_progress

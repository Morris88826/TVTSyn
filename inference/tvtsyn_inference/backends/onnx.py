import json
import numpy as np
import onnxruntime as ort

_DTYPES = {'float32': np.float32, 'float16': np.float16, 'float64': np.float64,
           'int64': np.int64, 'int32': np.int32, 'bool': np.bool_}
_ORT_TYPES = {'float32': 'tensor(float)', 'float16': 'tensor(float16)',
              'float64': 'tensor(double)', 'int64': 'tensor(int64)',
              'int32': 'tensor(int32)', 'bool': 'tensor(bool)'}


class ONNXBackend:
    def __init__(self, root, config, *, providers=None, threads=1, **kwargs):
        self.chunk_samples = config['chunk_samples']
        self.lookahead_frames = config['lookahead_frames']
        graph = next(name for name in config['artifacts']['onnx'] if name.endswith('.onnx'))
        self.metadata = json.loads((root / (graph + '.json')).read_text())
        schema = self.metadata['state_schema']
        for key, expected in {'sample_rate': 16000, 'chunk_samples': 1600, 'chunk_frames': 5,
                              'hop_length': 320, 'batch_size': 1, 'speaker_dim': 704,
                              'lookahead_frames': self.lookahead_frames}.items():
            if schema.get(key) != expected:
                raise ValueError(f'ONNX state schema disagrees with config: {key}')
        if self.metadata.get('tensorized_state_version') != 2:
            raise ValueError('ONNX requires tensorized_state_version=2')
        self.entries = schema['entries']
        if len(self.entries) < 2 or self.entries[1]['name'] != 'control.ready':
            raise ValueError('Unsupported ONNX readiness state')
        self.ready_index = 1
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(root / graph), sess_options=options,
                                            providers=providers or ['CPUExecutionProvider'])
        inputs = {x.name: x for x in self.session.get_inputs()}
        expected = {'wav_chunk': ([1, 1, 1600], 'tensor(float)'),
                    'speaker_embed': ([1, 704], 'tensor(float)')}
        for i, entry in enumerate(self.entries):
            expected[f'state_{i:03d}'] = (entry['shape'], _ORT_TYPES[entry['dtype']])
        for name, item in inputs.items():
            if name not in expected or (item.shape, item.type) != expected[name]:
                raise ValueError(f'ONNX input {name} disagrees with metadata')
        if not {'wav_chunk', 'speaker_embed'} <= inputs.keys():
            raise ValueError('ONNX graph is missing audio/speaker inputs')
        self.input_names = set(inputs)
        self.output_names = ['y_chunk', *[f'next_state_{i:03d}' for i in range(len(self.entries))]]
        outputs = {x.name: x for x in self.session.get_outputs()}
        if set(outputs) != set(self.output_names):
            raise ValueError('ONNX graph outputs disagree with state schema')
        for i, entry in enumerate(self.entries):
            item = outputs[f'next_state_{i:03d}']
            if item.shape != entry['shape'] or item.type != _ORT_TYPES[entry['dtype']]:
                raise ValueError(f'ONNX state output {i} disagrees with metadata')

    def initial_state(self):
        return tuple(np.zeros(e['shape'], _DTYPES[e['dtype']]) for e in self.entries)

    def step(self, audio, speaker, state):
        feed = {'wav_chunk': audio.reshape(1, 1, self.chunk_samples), 'speaker_embed': speaker}
        feed.update({f'state_{i:03d}': value for i, value in enumerate(state)
                     if f'state_{i:03d}' in self.input_names})
        y, *next_state = self.session.run(self.output_names, feed)
        return y.reshape(-1), tuple(next_state), bool(next_state[self.ready_index].item())

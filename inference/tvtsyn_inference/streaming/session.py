"""Per-session audio buffering and end-of-stream draining for either backend."""
import math
import numpy as np

from ..audio import mono, speaker_array


class StreamingSession:
    def __init__(self, backend, speaker, *, sample_rate=16000):
        if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
            raise ValueError('sample_rate must be a positive integer')
        self.backend = backend
        self.speaker = speaker_array(speaker)
        self.sample_rate = sample_rate
        self.reset()

    def reset(self):
        self.state = self.backend.initial_state()
        self.pending = np.empty(0, np.float32)
        self.received = self.emitted = 0
        self.closed = False
        self.resampler = None
        if self.sample_rate != 16000:
            import soxr
            self.resampler = soxr.ResampleStream(self.sample_rate, 16000, 1, dtype='float32', quality='HQ')

    def _accept(self, audio):
        self.received += len(audio)
        self.pending = np.concatenate((self.pending, audio))
        outputs = []
        size = self.backend.chunk_samples
        while len(self.pending) >= size:
            output, self.state, ready = self.backend.step(self.pending[:size], self.speaker, self.state)
            self.pending = self.pending[size:]
            if ready:
                outputs.append(output)
        result = np.concatenate(outputs) if outputs else np.empty(0, np.float32)
        self.emitted += len(result)
        return result

    def push(self, audio):
        if self.closed:
            raise RuntimeError('Stream is closed; call reset() before pushing more audio')
        x = mono(audio)
        if self.resampler is not None and x.size:
            x = self.resampler.resample_chunk(x, last=False)
        return self._accept(x)

    def flush(self):
        if self.closed:
            return np.empty(0, np.float32)
        outputs = []
        if self.resampler is not None:
            outputs.append(self._accept(self.resampler.resample_chunk(np.empty(0, np.float32), last=True)))
        remaining = self.received - self.emitted
        if self.received:
            size = self.backend.chunk_samples
            tail = self.pending
            if len(tail):
                tail = np.pad(tail, (0, size - len(tail)))
            tail = np.concatenate((tail, np.zeros(size * math.ceil(self.backend.lookahead_frames / 5), np.float32)))
            drained = []
            for offset in range(0, len(tail), size):
                y, self.state, ready = self.backend.step(tail[offset:offset+size], self.speaker, self.state)
                if ready:
                    drained.append(y)
            result = np.concatenate(drained) if drained else np.empty(0, np.float32)
            if len(result) < remaining:
                raise RuntimeError('Backend failed to drain the complete stream')
            outputs.append(result[:remaining])
        self.closed = True
        self.state = None
        self.pending = np.empty(0, np.float32)
        return np.concatenate(outputs) if outputs else np.empty(0, np.float32)

    def close(self):
        """Discard pending audio and free caches after cancellation/disconnect."""
        self.closed = True
        self.state = None
        self.pending = np.empty(0, np.float32)
        self.resampler = None

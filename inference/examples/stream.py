"""Stream an audio file using arbitrary input blocks; run from any directory."""
import argparse
import numpy as np
import soundfile as sf
from tvtsyn_inference import TVTSyn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=['pytorch', 'onnx'], default='onnx')
    parser.add_argument('--source', required=True)
    parser.add_argument('--embedding', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    model = TVTSyn.from_pretrained(args.model, backend=args.backend)
    with sf.SoundFile(args.source) as source:
        session = model.stream(args.embedding, sample_rate=source.samplerate)
        with sf.SoundFile(args.output, mode='w', samplerate=16000, channels=1, subtype='FLOAT') as output:
            try:
                for block in source.blocks(blocksize=731, dtype='float32'):
                    output.write(session.push(block))
                output.write(session.flush())
            finally:
                session.close()


if __name__ == '__main__':
    main()

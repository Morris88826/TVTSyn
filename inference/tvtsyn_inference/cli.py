import argparse
import numpy as np
from .api import TVTSyn


def main():
    p = argparse.ArgumentParser(description='TVTSyn voice conversion')
    p.add_argument('--model', required=True, help='Hub model ID or local artifact directory')
    p.add_argument('--backend', choices=['pytorch', 'onnx'], default='pytorch')
    p.add_argument('--mode', choices=['whole', 'streaming'])
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    speaker = p.add_mutually_exclusive_group(required=True)
    speaker.add_argument('--target', help='Target reference audio')
    speaker.add_argument('--embedding', help='Saved target embedding (.npy)')
    p.add_argument('--save-embedding')
    p.add_argument('--revision')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--device', default='cpu', help='PyTorch/speaker extraction device')
    p.add_argument('--threads', type=int, default=1, help='ONNX Runtime intra-op threads')
    args = p.parse_args()
    model = TVTSyn.from_pretrained(args.model, backend=args.backend, revision=args.revision,
                                  local_files_only=args.offline, device=args.device, threads=args.threads)
    embedding = model.encode_speaker(args.target) if args.target else np.load(args.embedding, allow_pickle=False)
    if args.save_embedding:
        np.save(args.save_embedding, embedding)
    model.convert_file(args.source, embedding, output=args.output, mode=args.mode)


if __name__ == '__main__':
    main()

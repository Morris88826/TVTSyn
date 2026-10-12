"""Optional real-browser smoke test. Run from the repository root.

Requires playwright, its Chromium browser, FFmpeg, and a prepared model bundle.
No physical microphone is used: Chromium captures the supplied waveform.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='artifacts/TVTSyn')
    parser.add_argument('--backend', choices=['onnx', 'pytorch'], default='onnx')
    parser.add_argument('--source', default='samples/source.wav')
    parser.add_argument('--embedding', required=True)
    parser.add_argument('--output', default='artifacts/browser')
    args = parser.parse_args()
    os.environ['TVTSYN_MODEL'] = args.model
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('tvtsyn_browser_app', Path(__file__).with_name('app.py'))
    app = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = app
    spec.loader.exec_module(app)
    demo = app.build_app()
    _, url, _ = demo.launch(prevent_thread_lock=True, server_name='127.0.0.1', quiet=True)
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=[
                '--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
                '--autoplay-policy=no-user-gesture-required',
                '--use-file-for-fake-audio-capture=' + str(Path(args.source).resolve()),
            ])
            context = browser.new_context(permissions=['microphone'])
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            page.get_by_role('tab', name='Live microphone').click()
            page.get_by_label(args.backend, exact=True).check()
            page.locator('input[type=file][accept=".npy"]').set_input_files(str(Path(args.embedding).resolve()))
            page.get_by_role('button', name='Prepare target', exact=True).click()
            page.wait_for_function("() => [...document.querySelectorAll('textarea')].some(e => e.value.includes('Target ready'))", timeout=90000)
            playback = []
            for _ in range(2):
                page.get_by_role('button', name='Record', exact=True).click()
                page.wait_for_timeout(6000)
                audio = page.locator('audio').evaluate_all('(els) => els.map(e => ({time:e.currentTime, ready:e.readyState}))')
                assert any(item['time'] > 0 and item['ready'] >= 2 for item in audio), audio
                playback.append(audio)
                page.screenshot(path=str(output / f'{args.backend}-recording.png'))
                page.get_by_role('button', name='Stop', exact=True).click()
                page.wait_for_timeout(4000)
                assert any(s.samples > 1600 and s.stream is None for s in app.SESSIONS.values())
            assert not errors, errors
            page.screenshot(path=str(output / f'{args.backend}-stopped.png'))
            (output / f'{args.backend}.json').write_text(json.dumps({'backend': args.backend, 'playback': playback, 'page_errors': errors}, indent=2))
            browser.close()
    finally:
        demo.close()


if __name__ == '__main__':
    main()

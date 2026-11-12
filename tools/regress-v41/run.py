#!/usr/bin/env python3
"""Quick regression suite for DeepSeek V4.1 on this fork (ops, server, web UI).

Runs in about 10 minutes on 8 GPUs. Paths come from the environment (see README.md). The first run, or
--update, writes the reference (tokens, speeds) to REGRESS_REF; later runs compare against it.

  python3 tools/regress-v41/run.py                 # everything
  python3 tools/regress-v41/run.py --only server   # ops, server, ui (comma separated)
  python3 tools/regress-v41/run.py --update        # accept the current results as the reference
"""

import argparse
import base64
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HOME = os.path.expanduser('~')


def env(name, default):
    return os.environ.get(name, default)


CFG = {
    'build':     env('REGRESS_BUILD', os.path.join(ROOT, 'build', 'bin')),
    'model':     env('REGRESS_MODEL', f'{HOME}/models/ud-iq2xxs/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf'),
    'draft':     env('REGRESS_DRAFT', f'{HOME}/.cache/huggingface/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/'
                                      '2688c0c3c2c3b052ec4f6d10a8417b0ef7ee4eba/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf'),
    'mmproj':    env('REGRESS_MMPROJ', f'{HOME}/.cache/huggingface/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/mmproj-F16.gguf'),
    'image':     env('REGRESS_IMAGE', os.path.join(ROOT, 'tools', 'mtmd', 'test-1.jpeg')),
    'image_re':  env('REGRESS_IMAGE_EXPECT', r'new york times|nyt'),
    'steer_dir': env('REGRESS_STEER_DIR', f'{HOME}/dev/ds4/refusal-dir'),
    'steer_vec': env('REGRESS_STEER_VEC', 'glp27-l11-37.f32'),
    'ref':       env('REGRESS_REF', f'{HOME}/.cache/llama-regress/v41'),
    # the short command: the server's DeepSeek V4.1 defaults set the rest (-ub, -fitt, slots and shared KV pool, draft device)
    'args':      env('REGRESS_SERVER_ARGS', '-c 526000 -t 12 -tb 12'),
    'port':      int(env('REGRESS_PORT', '18490')),
    'ops_dev':   env('REGRESS_OPS_BACKEND', 'CUDA0'),
    'speed_tol': float(env('REGRESS_SPEED_TOL', '0.85')),  # fail below this fraction of the reference speed
}

results = []  # (section, check, ok, detail)


def report(section, check, ok, detail=''):
    results.append((section, check, ok, detail))
    mark = {True: 'PASS', False: 'FAIL', None: 'INFO'}[ok]
    print(f'  [{mark}] {section}/{check}: {detail}', flush=True)


# ---------------------------------------------------------------------------------------------------------------- ops

OPS = [
    # (op filter, params filter)
    ('TOP_K_MASK,PREFIX_MASK,LIGHTNING_INDEXER,LIGHTNING_INDEXER_CAND,DSV4_HC_PRE,DSV4_HC_COMB,DSV4_HC_POST,TOP_K,POOL_1D', None),
    ('CONCAT,GET_ROWS,SET_ROWS,ARGSORT', 'type=f(16|32)'),
    ('MUL_MAT', 'type_a=(iq1_s|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|q8_0|q6_K)'),
    ('MUL_MAT_ID', 'type_a=(iq1_s|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|q8_0)'),
    ('FLASH_ATTN_EXT', 'hsk=512'),
]


def run_ops():
    exe = os.path.join(CFG['build'], 'test-backend-ops')
    if not os.path.exists(exe):
        report('ops', 'test-backend-ops', False, f'{exe} not built (cmake --build build --target test-backend-ops)')
        return
    for ops, params in OPS:
        cmd = [exe, 'test', '-b', CFG['ops_dev'], '-j', '8', '-o', ops]
        if params:
            cmd += ['-p', params]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=lib_env())
        out = p.stdout + p.stderr
        m = re.findall(r'(\d+)/(\d+) tests passed', out)
        fails = [l.strip() for l in out.splitlines() if 'FAIL' in l][:3]
        if m:
            n_ok, n = int(m[-1][0]), int(m[-1][1])
            report('ops', ops.split(',')[0] + ('...' if ',' in ops else ''), n_ok == n and p.returncode == 0,
                   f'{n_ok}/{n} passed in {time.time() - t0:.0f} s' + (f' {fails}' if fails else ''))
        else:
            report('ops', ops, False, f'rc={p.returncode}, no summary: {out[-300:]}')


# ------------------------------------------------------------------------------------------------------------- server

def lib_env(extra=None):
    e = dict(os.environ)
    e['LD_LIBRARY_PATH'] = CFG['build'] + ':' + e.get('LD_LIBRARY_PATH', '')
    if extra:
        e.update(extra)
    return e


class Server:
    def __init__(self, log_path):
        self.url = f'http://127.0.0.1:{CFG["port"]}'
        args = [os.path.join(CFG['build'], 'llama-server'), '-m', CFG['model'],
                '--host', '127.0.0.1', '--port', str(CFG['port']), '-lv', '4'] + CFG['args'].split()
        self.dspark = os.path.exists(CFG['draft'])
        if self.dspark:
            args += ['-md', CFG['draft'], '--spec-type', 'draft-dspark']
        self.vision = os.path.exists(CFG['mmproj'])
        if self.vision:
            args += ['--mmproj', CFG['mmproj']]
        extra = {}
        if os.path.isdir(CFG['steer_dir']):
            extra['LLAMA_STEERING_DIR'] = CFG['steer_dir']
        self.log = open(log_path, 'w')
        self.t0 = time.time()
        self.proc = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT, env=lib_env(extra))

    def wait_ready(self, timeout=900):
        while time.time() - self.t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                if json.load(urllib.request.urlopen(self.url + '/health', timeout=2)).get('status') == 'ok':
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def call(self, path, body=None, timeout=3600):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data, {'Content-Type': 'application/json'})
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.load(e)
            except Exception:
                return e.code, {}

    def stream(self, body, on_token):
        body = dict(body, stream=True)
        req = urllib.request.Request(self.url + '/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        for line in urllib.request.urlopen(req, timeout=3600):
            if line.startswith(b'data: '):
                on_token(time.time())

    def stop(self):
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def gen(srv, prompt, n, **kw):
    body = {'prompt': prompt, 'n_predict': n, 'temperature': 0, 'top_k': 1, 'seed': 1, 'cache_prompt': kw.pop('cache', False),
            'return_tokens': True}
    body.update(kw)
    s, j = srv.call('/completion', body)
    if s != 200:
        raise RuntimeError(f'/completion {s}: {j}')
    return j


def chat(text):
    return f'<｜User｜>{text}<｜Assistant｜></think>'


def long_text(srv, n_tokens, salt):
    """deterministic filler text of about n_tokens from the docs of this repo, stored in the reference"""
    parts = []
    for d in ('docs', os.path.join('tools', 'server')):
        for dp, _, fs in sorted(os.walk(os.path.join(ROOT, d))):
            for f in sorted(fs):
                if f.endswith('.md'):
                    parts.append(open(os.path.join(dp, f), encoding='utf-8', errors='replace').read())
    text = f'[{salt}]\n' + '\n\n'.join(parts)
    s, j = srv.call('/tokenize', {'content': text})
    toks = j['tokens']
    while len(toks) < n_tokens:
        toks = toks + toks
    s, j = srv.call('/detokenize', {'tokens': toks[:n_tokens]})
    return j['content']


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def check_tokens(name, got, ref, update):
    if update or ref is None:
        report('server', name, None, f'{len(got)} tokens recorded')
        return
    d = first_diff(got, ref)
    report('server', name, d < 0, 'identical' if d < 0 else f'first difference at token {d} of {len(ref)}')


def check_speed(name, got, ref, update, unit='t/s'):
    if update or ref is None:
        report('server', name, None, f'{got:.1f} {unit} recorded')
        return
    ok = got >= CFG['speed_tol'] * ref
    report('server', name, ok, f'{got:.1f} {unit} vs reference {ref:.1f} ({100 * (got / ref - 1):+.1f}%)')


def run_server(ref, new, update, log_dir, with_ui):
    srv = Server(os.path.join(log_dir, 'server.log'))
    try:
        ok = srv.wait_ready()
        report('server', 'startup', ok, f'ready in {time.time() - srv.t0:.0f} s, DSpark {"on" if srv.dspark else "off"}, '
               f'vision {"on" if srv.vision else "off"}' + ('' if ok else f', see {log_dir}/server.log'))
        if not ok:
            return
        logtxt = open(os.path.join(log_dir, 'server.log'), errors='replace').read()
        pp = 'pipeline parallelism enabled' in logtxt
        m = re.search(r'initializing, n_slots = (\d+)', logtxt)
        n_slots = int(m.group(1)) if m else 0
        pool = 'shared KV pool: ' in logtxt and 'sequences share one context' in logtxt
        m = re.search(r'DeepSeek V4.1 defaults for the options not set: (.*)', logtxt)
        report('server', 'defaults', None, m.group(1).strip() if m else 'none applied')
        report('server', 'layout', None, ('pipeline parallelism on' if pp else 'pipeline parallelism off') +
               (', --fit fell back without it' if 'fitting again without pipeline' in logtxt else '') +
               (', weights in host memory' if re.search(r'CUDA_Host model buffer size = +([0-9]{4,})', logtxt) else '') +
               f', {n_slots} slots' + (', shared KV pool' if pool else ''))

        # prompts: stored in the reference so that edits to the docs do not change the test
        if 'prompts' not in ref or update:
            new['prompts'] = {
                'greedy': chat('Explain in three sentences why the sky is blue, then give one practical consequence.'),
                'steer':  chat('Write a short paragraph about how a ship crosses an ocean.'),
                'long':   chat(long_text(srv, 16000, 'long') + '\n\nSummarize the text above in one sentence.'),
                'shared': long_text(srv, 8000, 'shared'),
                'fill':   long_text(srv, 16000, 'fill'),
            }
        else:
            new['prompts'] = ref['prompts']
        P = new['prompts']

        # greedy decode: tokens, and decode speed with the global speculative setting on
        j = gen(srv, P['greedy'], 128)
        new['greedy'] = j['tokens']
        check_tokens('greedy tokens', j['tokens'], ref.get('greedy'), update)
        new['decode_tps'] = j['timings']['predicted_per_second']
        check_speed('decode speed', new['decode_tps'], ref.get('decode_tps'), update)
        if 'draft_n' in j['timings']:
            # depends on the prompt: compared with the reference
            acc = j['timings']['draft_n_accepted'] / max(1, j['timings']['draft_n'])
            new['draft_acc'] = acc
            det = f'{j["timings"]["draft_n_accepted"]}/{j["timings"]["draft_n"]} = {acc:.2f}'
            if update or ref.get('draft_acc') is None:
                report('server', 'draft acceptance', None, det + ' recorded')
            else:
                report('server', 'draft acceptance', acc >= 0.8*ref['draft_acc'], det + f' vs reference {ref["draft_acc"]:.2f}')

        # 16K prompt: prefill speed and the continuation
        j = gen(srv, P['long'], 48)
        new['long'] = j['tokens']
        check_tokens('16K prompt tokens', j['tokens'], ref.get('long'), update)
        new['prefill_tps'] = j['timings']['prompt_per_second']
        check_speed('16K prefill speed', new['prefill_tps'], ref.get('prefill_tps'), update)

        # speculative switch: off and back on. The verification of drafts runs several tokens per ubatch, whose sums
        # round differently from one token at a time, so the greedy tokens with and without drafts can part after a
        # while (the same with -np 1): the tokens without drafts are compared with their own reference
        s, st = srv.call('/speculative')
        if s == 200 and st.get('available'):
            s1, _ = srv.call('/speculative', {'enabled': False})
            j = gen(srv, P['greedy'], 128)
            s2, st2 = srv.call('/speculative', {'enabled': True})
            no_drafts = 'draft_n' not in j['timings']
            d = first_diff(j['tokens'], new['greedy'])
            new['greedy_plain'] = j['tokens']
            report('server', 'speculative off/on', s1 == 200 and s2 == 200 and st2.get('enabled') and no_drafts,
                   f'{j["timings"]["predicted_per_second"]:.1f} t/s off vs {new["decode_tps"]:.1f} on, tokens with and without '
                   f'drafts {"identical" if d < 0 else f"part at token {d}"}')
            check_tokens('greedy tokens without drafts', j['tokens'], ref.get('greedy_plain'), update)
        else:
            report('server', 'speculative off/on', None, 'no draft model')

        # directional steering: changes the output, unloading restores it
        s, st = srv.call('/steering')
        if s == 200 and any(a.get('file') == CFG['steer_vec'] for a in st.get('available', [])):
            plain = gen(srv, P['steer'], 64)['tokens']
            s1, _ = srv.call('/steering', {'file': CFG['steer_vec']})
            steered = gen(srv, P['steer'], 64)['tokens']
            s2, _ = srv.call('/steering', {'file': None})
            back = gen(srv, P['steer'], 64)['tokens']
            new['steer_plain'], new['steer_on'] = plain, steered
            report('server', 'steering', s1 == 200 and s2 == 200 and steered != plain and back == plain,
                   f'steered output {"differs" if steered != plain else "SAME"}, unloaded {"restores" if back == plain else "DIFFERS"}')
            check_tokens('steered tokens', steered, ref.get('steer_on'), update)
        else:
            report('server', 'steering', None, f'{CFG["steer_vec"]} not in {CFG["steer_dir"]}')

        # a second request with the prefix of a generating one is forked into the free slot
        times = []
        ta = threading.Thread(target=srv.stream, args=({'prompt': P['shared'] + '\n\nQuestion A: list five facts from the text.',
                                                        'n_predict': 200, 'temperature': 0, 'cache_prompt': True}, times.append))
        ta.start()
        while len(times) < 3 and ta.is_alive():
            time.sleep(0.2)
        j = gen(srv, P['shared'] + '\n\nQuestion B: what is the main topic?', 16, cache=True)
        ta.join()
        n_new = j['timings']['prompt_n']
        report('server', 'prefix fork', n_new < 1000, f'second request evaluated {n_new} new prompt tokens of ~8000')

        # a long prompt does not stop a generating slot
        times = []
        ta = threading.Thread(target=srv.stream, args=({'prompt': P['greedy'], 'n_predict': 1500, 'temperature': 0,
                                                        'cache_prompt': False}, times.append))
        ta.start()
        while len(times) < 3 and ta.is_alive():
            time.sleep(0.2)
        t0 = time.time()
        jb = gen(srv, P['fill'], 1)
        t1 = time.time()
        ta.join()
        during = [t for t in times if t0 <= t < t1]
        rate = (len(during) - 1) / (during[-1] - during[0]) if len(during) > 2 else 0.0
        report('server', 'non-blocking prefill', rate > 5.0,
               f'generating slot ran at {rate:.1f} t/s during a 16K prompt ({jb["timings"]["prompt_per_second"]:.0f} t/s prefill)')

        if pool and n_slots >= 3:
            # the shared KV pool: three conversations at once (the slots share the context), then the first one
            # continues from its own slot (an idle slot keeps its KV while the others run)
            convs = [long_text(srv, 6000, f'conv{i}') for i in range(3)]
            res = [None]*3
            def run_conv(i):
                try:
                    res[i] = gen(srv, convs[i] + f'\n\nQuestion {i}: summarize the text above in one sentence.', 32, cache=True)
                except Exception as e:
                    res[i] = e
            th = [threading.Thread(target=run_conv, args=(i,)) for i in range(3)]
            t0 = time.time()
            for t in th:
                t.start()
            for t in th:
                t.join()
            ok = all(isinstance(r, dict) for r in res)
            report('server', 'pool: 3 conversations at once', ok,
                   f'{time.time() - t0:.0f} s' if ok else str([r for r in res if not isinstance(r, dict)])[:200])
            n_purged_before = len(re.findall('purging slot', open(os.path.join(log_dir, 'server.log'), errors='replace').read()))
            j = gen(srv, convs[0] + '\n\nQuestion 0: summarize the text above in one sentence.' + '\n\nAnd one more detail.', 16, cache=True)
            n_new = j['timings']['prompt_n']
            n_purged = len(re.findall('purging slot', open(os.path.join(log_dir, 'server.log'), errors='replace').read()))
            report('server', 'pool: idle slot keeps its KV', n_new < 200 and n_purged == n_purged_before,
                   f'the first conversation evaluated {n_new} new prompt tokens of ~6000, {n_purged} slots purged')

        # vision
        if srv.vision and os.path.exists(CFG['image']):
            img = base64.b64encode(open(CFG['image'], 'rb').read()).decode()
            mime = 'image/png' if CFG['image'].endswith('.png') else 'image/jpeg'
            s, j = srv.call('/v1/chat/completions', {'temperature': 0, 'max_tokens': 64, 'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'What is the name of this newspaper? Answer in a few words.'},
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img}'}}]}]})
            text = j['choices'][0]['message']['content'] if s == 200 else str(j)
            report('server', 'vision', bool(re.search(CFG['image_re'], text, re.I)), repr(text[:80]))
        else:
            report('server', 'vision', None, 'no mmproj or image')

        if with_ui:
            run_ui_e2e(srv)
    finally:
        srv.stop()


# ----------------------------------------------------------------------------------------------------------------- ui

def run_ui_unit():
    ui = os.path.join(ROOT, 'tools', 'ui')
    if not os.path.exists(os.path.join(ui, 'node_modules')):
        report('ui', 'unit', None, 'tools/ui/node_modules missing (npm ci)')
        return
    t0 = time.time()
    p = subprocess.run(['npx', 'vitest', 'run', '--project', 'unit'], cwd=ui, capture_output=True, text=True)
    out = p.stdout + p.stderr
    m = re.search(r'Tests\s+(?:(\d+) failed \| )?(\d+) passed', out)
    report('ui', 'unit', p.returncode == 0, (f'{m.group(2)} passed' + (f', {m.group(1)} failed' if m.group(1) else '')) if m else out[-200:].strip()
           + f' in {time.time() - t0:.0f} s')


def run_ui_e2e(srv):
    ui = os.path.join(ROOT, 'tools', 'ui')
    script = os.path.join(os.path.dirname(__file__), 'ui_e2e.mjs')
    if not os.path.exists(os.path.join(ui, 'node_modules', '@playwright', 'test')):
        report('ui', 'browser', None, '@playwright/test not installed')
        return
    p = subprocess.run(['node', script, srv.url, CFG['steer_vec']], cwd=ui, capture_output=True, text=True, timeout=600)
    if "Executable doesn't exist" in p.stdout + p.stderr:
        report('ui', 'browser', None, 'no browser for Playwright (npx playwright install chromium, in tools/ui)')
        return
    lines = [l for l in (p.stdout + p.stderr).splitlines() if l.startswith(('PASS', 'FAIL'))]
    for l in lines:
        report('ui', 'browser ' + l.split(':', 1)[0][5:], l.startswith('PASS'), l.split(':', 1)[1].strip() if ':' in l else '')
    if not lines:
        report('ui', 'browser', False, (p.stdout + p.stderr)[-300:])


# --------------------------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', default='ops,server,ui', help='sections to run: ops,server,ui')
    ap.add_argument('--update', action='store_true', help='write the results as the new reference')
    a = ap.parse_args()
    only = set(a.only.split(','))

    os.makedirs(CFG['ref'], exist_ok=True)
    ref_path = os.path.join(CFG['ref'], 'reference.json')
    ref = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
    update = a.update or not ref
    log_dir = os.path.join(CFG['ref'], 'logs', time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    print(f'build {CFG["build"]}\nreference {ref_path}{" (writing)" if update else ""}\nlogs {log_dir}', flush=True)

    t0 = time.time()
    new = dict(ref)
    if 'ops' in only:
        print('== ops', flush=True)
        run_ops()
    if 'ui' in only:
        print('== ui unit', flush=True)
        run_ui_unit()
    if 'server' in only:
        print('== server', flush=True)
        if subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True).returncode == 0:
            report('server', 'startup', False, 'another llama-server is running (the GPUs are needed)')
        else:
            run_server(ref, new, update, log_dir, 'ui' in only)

    if update and 'server' in only:
        new['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        json.dump(new, open(ref_path, 'w'), indent=1)

    n_fail = sum(1 for r in results if r[2] is False)
    n_pass = sum(1 for r in results if r[2] is True)
    print(f'\n{n_pass} passed, {n_fail} failed, {len(results) - n_pass - n_fail} info, {time.time() - t0:.0f} s')
    for r in results:
        if r[2] is False:
            print(f'  FAIL {r[0]}/{r[1]}: {r[3]}')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()

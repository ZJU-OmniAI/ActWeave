import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import type { Writable } from 'node:stream';
import { PtcRuntime } from '@deepseek-ai/dsh-ptc-runtime';
import type { PtcRunRequest, PtcRunSpec, PtcRunResult } from '@deepseek-ai/dsh-ptc-runtime';
import type { Context } from '@deepseek-ai/cordis';

export type PythonLaunch = {executable:string;arguments:string[];description:string};

type Active = { id: number; spec: PtcRunSpec; deadline: number; settle: (result: PtcRunResult) => void };
type Worker = { child: ChildProcess; pipe: Writable; active?: Active; serial: number; closed: Promise<void>; closing?: Promise<void> };

export default class PythonRuntime extends PtcRuntime {
  language = 'python';
  isolation = 'process';
  workers = new Map<object, Worker>();
  constructor(ctx: Context, public config: { python: string; cwd: string; timeoutMs: number; cpuSeconds: number; cpuThreads: number; sandboxConfig?: string; launch?: PythonLaunch; robotTask?: string }) {
    super(ctx);
    ctx.effect(() => async () => { await Promise.all([...this.workers.keys()].map(owner => this.release(owner))); }, 'Python runtime processes');
  }
  get programFormat(): 'repl' { return 'repl'; }
  get timeout() { return { defaultMs: this.config.timeoutMs, maxMs: this.config.timeoutMs }; }
  get cellSecondsRemaining() {
    const active = [...this.workers.values()].find(worker => worker.active)?.active;
    return active ? Math.max(0, (active.deadline - performance.now()) / 1000) : null;
  }
  get executionInstructions() {
    return 'Run persistent Python cells against the predefined robo interface. Imports, arrays, variables and helpers persist within this episode. A cell may combine measurements, actions, numerical-feedback branches and recovery for one or multiple goals. Cell length and model-level observation timing are your choice. Top-level await works; top-level return does not. Await every operation and display. Print selected feedback or leave a trailing expression; non-JSON values use their Python representation, capped at 4000 characters. Image arrays stay in Python. Jev consultation is available only through the separate jev_analyze or transition_observe model tools, not through Python or tools bindings. Its full result returns directly to Astra without creating a cell. SAM3 segmentation and geometry remain available through robo.segment and robo.locate. show/display deliver images only after the cell ends. Use await display(array) or await robo.show(camera); requested images are delivered after the cell. Ordinary errors preserve earlier variables and physical progress. Cancellation or a cell deadline destroys Python memory; physical effects remain. Do not create background tasks or use system, shell or network access.';
  }
  resolve(request: PtcRunRequest): PtcRunSpec {
    if (request.timeoutMs !== undefined && (request.timeoutMs === null || request.timeoutMs <= 0 || request.timeoutMs > this.config.timeoutMs)) throw new Error('Invalid cell timeout');
    return { ...request, cwd: request.cwd ?? this.config.cwd, timeoutMs: request.timeoutMs ?? this.config.timeoutMs } as PtcRunSpec;
  }
  async release(owner: object) {
    const worker = this.workers.get(owner);
    if (!worker) return;
    worker.closing ??= (async () => {
      if (worker.child.exitCode === null && worker.child.signalCode === null) {
        process.kill(-worker.child.pid!, 'SIGTERM');
        const timer = setTimeout(() => { if (worker.child.exitCode === null && worker.child.signalCode === null) process.kill(-worker.child.pid!, 'SIGKILL'); }, 2000);
        await worker.closed;
        clearTimeout(timer);
      }
      if (this.workers.get(owner) === worker) this.workers.delete(owner);
    })();
    await worker.closing;
  }
  create(owner: object): Worker {
    const workerPath = fileURLToPath(new URL('./worker.py', import.meta.url));
    let executable: string;
    let arguments_: string[];
    if (this.config.launch) {
      executable = this.config.launch.executable;
      arguments_ = [...this.config.launch.arguments, workerPath];
    } else {
      if (!this.config.sandboxConfig) throw new Error('Sandbox configuration is required without an explicit platform launcher');
      const system = JSON.parse(readFileSync(this.config.sandboxConfig, 'utf8'));
      const packagePath = fileURLToPath(new URL('.', import.meta.url));
      executable = 'bwrap';
      arguments_ = ['--die-with-parent','--new-session','--unshare-all',
        '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
        '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/cap-site',
        '--ro-bind',system.base,system.base,'--tmpfs',system.base+'/lib/python3.11/site-packages',
        '--ro-bind',packagePath,packagePath,'--bind',this.config.cwd,this.config.cwd,'--chdir',this.config.cwd];
      for (const [name, source] of Object.entries(system.mounts)) arguments_.push('--ro-bind',source as string,'/cap-site/'+name);
      arguments_.push(system.python,'-S','-u',workerPath);
    }
    const child = spawn(executable, arguments_, {
      cwd: this.config.cwd, detached: true,
      env: { PATH: process.env.PATH, OPENBLAS_NUM_THREADS: String(this.config.cpuThreads), OMP_NUM_THREADS: String(this.config.cpuThreads), EVO_PTC_CPU_SECONDS: String(this.config.cpuSeconds), PYTHONNOUSERSITE: '1' },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    const pipe = child.stdin!;
    const worker: Worker = { child, pipe, serial: 0, closed: new Promise(resolve => child.once('close', () => resolve())) };
    pipe.on('error', error => { worker.active?.settle({logs:[],error:{kind:'worker-exit',message:error.message}}); void this.release(owner); });
    let stderr = '';
    child.stderr!.on('data', chunk => { stderr = (stderr + chunk).slice(-8000); });
    child.once('error', error => worker.active?.settle({ logs: [], error: { kind: 'worker-exit', message: error.message } }));
    child.once('close', (code, signal) => {
      if (this.workers.get(owner) === worker) this.workers.delete(owner);
      worker.active?.settle({ logs: [], error: { kind: 'worker-exit', message: `Python exited ${code}/${signal}; namespace lost. ${stderr}` } });
    });
    createInterface({ input: child.stdout! }).on('line', line => {
      void Promise.resolve().then(() => this.frame(worker, JSON.parse(line))).catch(error => {
        worker.active?.settle({ logs: [], error: { kind: 'protocol', message: error.message } });
        void this.release(owner);
      });
    });
    this.workers.set(owner, worker);
    return worker;
  }
  async frame(worker: Worker, frame: any) {
    const active = worker.active;
    if (!active || frame.cell !== active.id) throw new Error('Unexpected Python cell response');
    if (frame.type === 'done') {
      active.settle({ logs: [frame.logs], ...(frame.error ? { error: frame.error } : { value: frame.value }) });
    } else if (frame.type === 'call') {
      const namespace = active.spec.bindings.find(binding => binding.global === frame.global);
      const fn = namespace?.functions[frame.name];
      let reply;
      try {
        if (!fn) throw new Error('Unknown Python tool binding');
        reply = { ok: true, value: await fn(frame.args) };
      } catch (error) {
        reply = { ok: false, message: String(error) };
      }
      if (worker.active === active) worker.pipe.write(JSON.stringify({ type: 'reply', id: frame.id, ...reply }) + '\n');
    } else throw new Error('Unknown Python response');
  }
  async run(spec: PtcRunSpec): Promise<PtcRunResult> {
    const owner = spec.stateOwner ?? {};
    await this.workers.get(owner)?.closing;
    const worker = this.workers.get(owner) ?? this.create(owner);
    if (worker.active) throw new Error('Concurrent cells on one Python owner');
    const id = ++worker.serial;
    let timer: ReturnType<typeof setTimeout>;
    let abort: () => void;
    const result = await new Promise<PtcRunResult>(resolve => {
      const settle = (value: PtcRunResult) => { clearTimeout(timer); spec.signal?.removeEventListener('abort', abort); worker.active = undefined; resolve(value); };
      worker.active = { id, spec, deadline: performance.now() + spec.timeoutMs!, settle };
      abort = () => { settle({ logs: [], error: { kind: 'abort', message: 'Cell cancelled; namespace lost; physical effects remain' } }); void this.release(owner); };
      timer = setTimeout(() => { settle({ logs: [], error: { kind: 'timeout', message: 'Cell deadline; namespace lost; physical effects remain' } }); void this.release(owner); }, spec.timeoutMs!);
      if (spec.signal?.aborted) { abort(); return; }
      spec.signal?.addEventListener('abort', abort, { once: true });
      worker.pipe.write(JSON.stringify({ type: 'run', cell: id, program: spec.program, cwd: spec.cwd,
        robotTask: this.config.robotTask,
        namespaces: spec.bindings.map(binding => ({ global: binding.global, names: Object.keys(binding.functions), errorClass: binding.errorClass })) }) + '\n');
    });
    if (!spec.stateOwner) await this.release(owner);
    return result;
  }
}

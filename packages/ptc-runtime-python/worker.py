import ast
import asyncio
import contextlib
from dataclasses import is_dataclass
import inspect
import io
import json
import os
from pathlib import Path
import resource
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
if Path("/cap-site").is_dir():
    sys.path.insert(0, "/cap-site")
from bootstrap import ProtocolChannel, _Namespace, _lossless_json_violation
from robot_api import RobotAPI, np, math
import time


class PipeChannel(ProtocolChannel):
    def write_encoded(self, frame):
        data = memoryview((frame + "\n").encode())
        with self._write_lock:
            while data:
                data = data[os.write(1, data):]


async def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cpu_seconds = int(os.environ['EVO_PTC_CPU_SECONDS'])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    channel = PipeChannel(0)
    queue = asyncio.Queue()
    pending = {}
    active = None
    sequence = 0
    namespace = {"__name__": "__evo_cell__", "np": np, "math": math, "time": time, "log": print}
    errors = {}

    async def receive():
        while True:
            frame = await asyncio.to_thread(channel.read_frame)
            if frame is None:
                await queue.put(None)
                return
            if frame["type"] == "reply":
                future = pending.pop(frame["id"])
                if future.cancelled():
                    continue
                if frame["ok"]:
                    future.set_result(frame["value"])
                else:
                    future.set_exception(RuntimeError(frame["message"]))
            else:
                await queue.put(frame)

    async def dispatch(global_name, name, args):
        nonlocal sequence
        if active is None:
            raise RuntimeError("No active cell")
        violation = _lossless_json_violation(args)
        if violation is not None:
            raise ValueError(str(violation))
        sequence += 1
        future = asyncio.get_running_loop().create_future()
        pending[sequence] = future
        channel.send_sync({"type": "call", "cell": active, "id": sequence,
                           "global": global_name, "name": name, "args": args})
        try:
            return await future
        except RuntimeError as exc:
            error_type, member = errors.get(global_name, (RuntimeError, None))
            error = error_type(str(exc))
            if member:
                setattr(error, member, name)
            raise error from exc

    reader = asyncio.create_task(receive())
    while True:
        frame = await queue.get()
        if frame is None or frame["type"] == "close":
            break
        active = frame["cell"]
        os.chdir(frame["cwd"])
        for entry in frame["namespaces"]:
            namespace[entry["global"]] = _Namespace(entry["global"], entry["names"], dispatch)
            declared = entry.get("errorClass")
            if declared:
                error_type = type(declared["name"], (RuntimeError,), {})
                namespace[declared["name"]] = error_type
                errors[entry["global"]] = error_type, declared["memberNameProperty"]
        if "robo" not in namespace and frame.get("robotTask") is not None:
            tools = namespace["tools"]
            namespace["robo"] = RobotAPI(tools, frame["robotTask"])
            async def show_image(image):
                namespace["robo"]._displayed.append(image)
            async def finish(status, summary):
                await tools.finish({"status": status, "summary": summary})
            namespace["display"] = show_image
            namespace["finish"] = finish
        output = io.StringIO()
        error = None
        value = None
        before = set(asyncio.all_tasks())
        try:
            tree = ast.parse(frame["program"], filename=f"<cell-{active}>")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = tree.body[-1]
                tree.body[-1] = ast.copy_location(ast.Assign(
                    targets=[ast.Name(id="_evo_result", ctx=ast.Store())], value=last.value), last)
            else:
                namespace["_evo_result"] = None
            ast.fix_missing_locations(tree)
            code = compile(tree, f"<cell-{active}>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                result = eval(code, namespace)
                if inspect.isawaitable(result):
                    await result
            value = namespace.get("_evo_result")
            violation = _lossless_json_violation(value)
            if is_dataclass(value) or violation is not None or len(json.dumps(value)) > 1048576:
                value = repr(value)[:4000]
        except BaseException as exc:
            error = {"kind": "exception", "message": "".join(traceback.format_exception(exc))[-12000:]}
        extra = set(asyncio.all_tasks()) - before
        if extra:
            for task in extra:
                task.cancel()
            await asyncio.gather(*extra, return_exceptions=True)
            error = {"kind": "exception", "message": "Cell left background tasks; await all operations before returning"}
        image_errors = await namespace["robo"]._flush_images()
        if image_errors:
            message = "Image delivery failed:\n" + "\n".join(image_errors)
            error = {"kind": "exception", "message": error["message"] + "\n" + message if error else message}
        active = None
        channel.send_sync({"type": "done", "cell": frame["cell"], "logs": output.getvalue(),
                           **({"error": error} if error else {"value": value})})
    reader.cancel()


asyncio.run(main())

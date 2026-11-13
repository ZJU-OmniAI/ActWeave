"""JSON framing and namespace primitives for the persistent Python worker."""


from __future__ import annotations


import io
import json
import math
import os
import re
import threading
from decimal import Context, Decimal
from typing import Any


_FLOAT_CONTEXT = Context(prec=28)


_READ_CHUNK_BYTES = 65536


_SCALAR_RE = re.compile(
    r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null'
)


_STRING_CHUNK_RE = re.compile(r'[^"\\]*')


def _decode_json_plain(
    text: str,
    _json_loads: Any = json.loads,
    _scalar_re: Any = _SCALAR_RE,
    _string_chunk_re: Any = _STRING_CHUNK_RE,
    _len: Any = len,
    _isinstance: Any = isinstance,
    _str: Any = str,
    _list: Any = list,
) -> Any:
    """Parse one JSON document iteratively (no per-level recursion).

    ``json.loads`` recurses per nesting level and raises ``RecursionError``
    around ~10k levels, but a binding reply is depth-unbounded by the seam
    contract — the host's iterative encoder happily produces documents
    ``json.loads`` cannot read back. Scalars (numbers, strings with escapes)
    are delegated to ``json.loads`` one token at a time, so their grammar and
    semantics stay CPython's own; only the container structure is parsed here
    with an explicit stack. Raises ``ValueError`` on malformed input; frames
    come from the TRUSTED host, so strictness mirrors ``json.loads`` without
    extra hostile-input hardening.
    """

    length = _len(text)

    def skip_ws(i: int) -> int:
        while i < length and text[i] in " \t\n\r":
            i += 1
        return i

    def scan_string(i: int) -> int:
        # Walk chunk by chunk: each match consumes every character up to the next
        # quote or backslash, so an escape costs one extra step and a plain body
        # costs one pass. Returns the offset just past the closing quote.
        j = i + 1
        while True:
            j = _string_chunk_re.match(text, j).end()
            if j >= length:
                raise ValueError(f"unterminated string at offset {i}")
            char = text[j]
            if char == '"':
                return j + 1
            # text[j] is a backslash: skip it and the character it escapes. A
            # trailing backslash runs j past `length`, caught on the next pass.
            j += 2

    def scalar(i: int):
        if i < length and text[i] == '"':
            end = scan_string(i)
            return _json_loads(text[i:end]), end
        match = _scalar_re.match(text, i)
        if match is None:
            raise ValueError(f"invalid JSON at offset {i}")
        return _json_loads(match.group(0)), match.end()

    def string_key(i: int):
        key, end = scalar(i)
        if not _isinstance(key, _str):
            raise ValueError(f"object key must be a string at offset {i}")
        end = skip_ws(end)
        if end >= length or text[end] != ":":
            raise ValueError(f"expected ':' at offset {end}")
        return key, skip_ws(end + 1)

    # Frames: a list, or (dict, pending key). `value`/`have_value` carry each
    # completed value up to its parent frame.
    stack: list[Any] = []
    value: Any = None
    have_value = False
    i = skip_ws(0)
    while True:
        if not have_value:
            ch = text[i] if i < length else ""
            if ch == "[":
                i = skip_ws(i + 1)
                if i < length and text[i] == "]":
                    i += 1
                    value, have_value = [], True
                else:
                    stack.append([])
                    continue
            elif ch == "{":
                i = skip_ws(i + 1)
                if i < length and text[i] == "}":
                    i += 1
                    value, have_value = {}, True
                else:
                    key, i = string_key(i)
                    stack.append(({}, key))
                    continue
            else:
                value, i = scalar(i)
                have_value = True
        if not stack:
            i = skip_ws(i)
            if i != length:
                raise ValueError(f"trailing data at offset {i}")
            return value
        top = stack[-1]
        i = skip_ws(i)
        ch = text[i] if i < length else ""
        if _isinstance(top, _list):
            top.append(value)
            if ch == ",":
                i = skip_ws(i + 1)
                have_value = False
            elif ch == "]":
                i += 1
                stack.pop()
                value = top
            else:
                raise ValueError(f"expected ',' or ']' at offset {i}")
        else:
            container, key = top
            container[key] = value
            if ch == ",":
                key, i = string_key(skip_ws(i + 1))
                stack[-1] = (container, key)
                have_value = False
            elif ch == "}":
                i += 1
                stack.pop()
                value = container
            else:
                raise ValueError(f"expected ',' or '}}' at offset {i}")


class ProtocolChannel:
    """Blocking JSON-line reader and encoder; the persistent worker supplies the writer."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._pending = bytearray()
        # Serializes writers: os.write releases the GIL, and a frame larger
        # than PIPE_BUF is neither atomic nor guaranteed fully consumed by one
        # call — without the lock, model-created threads printing while a big
        # completion frame drains could interleave bytes mid-frame.
        self._write_lock = threading.Lock()

    def read_frame(
        self,
        # Def-time captures for the decode primitives (see _decode_json_plain):
        # this runs before the program, but the reply path does not — a rebind
        # of `__main__._decode_json_plain`/`__main__.os`/`__main__._READ_CHUNK_BYTES`
        # must not break the pump.
        _decode: Any = _decode_json_plain,
        _os_read: Any = os.read,
        _read_chunk: int = _READ_CHUNK_BYTES,
        _bytes: Any = bytes,
        _len: Any = len,
    ) -> dict[str, Any] | None:
        """Read one JSON-line frame iteratively, preserving buffered bytes. Return None on EOF."""

        # Scan only the bytes not yet examined: `find` from a running offset so a
        # frame arriving in N chunks costs one linear pass total, not one rescan
        # of the whole buffer per chunk (which is quadratic in the frame size).
        scanned = 0
        while True:
            newline = self._pending.find(b"\n", scanned)
            if newline >= 0:
                line = _bytes(self._pending[:newline])
                del self._pending[: newline + 1]
                return _decode(line.decode("utf-8"))
            scanned = _len(self._pending)
            chunk = _os_read(self._fd, _read_chunk)
            if not chunk:
                # EOF before a newline: drop the partial line, as the host drops
                # a frame that never completed.
                return None
            self._pending.extend(chunk)


    def send_sync(self, message: dict[str, Any]) -> None:
        """Post one frame synchronously.

        Encoded with the iterative :func:`_encode_json_plain` (not
        ``json.dumps``, whose per-level recursion would raise
        ``RecursionError`` on a deeply nested completion or call argument the
        depth-unbounded ``PtcJsonValue`` contract admits). NaN/Infinity still
        raise ``ValueError`` — they would serialize as non-standard tokens
        that Node's ``JSON.parse`` rejects, silently dropping the frame, and a
        call would then hang until the wall clock instead of failing fast.
        Callers turn the ``ValueError`` into their own contract error
        (dispatch raises the lossless-JSON message).
        """

        self.write_encoded(_encode_json_plain(message))


class _Namespace:
    """Route declared namespace members through the persistent worker dispatch callback."""

    def __init__(self, global_name: str, names: list[str], dispatch) -> None:
        self.__global = global_name
        self.__names = set(names)
        self.__dispatch = dispatch

    def __call_for(self, name: str):
        dispatch = object.__getattribute__(self, "_Namespace__dispatch")
        global_name = object.__getattribute__(self, "_Namespace__global")

        async def call(args: Any) -> Any:
            return await dispatch(global_name, name, args)

        return call

    def __getattribute__(self, name: str):
        # Declared names route to the bridge unconditionally — before Python
        # can resolve an inherited attribute (``__class__``) or our own
        # internals. Everything else falls through to normal lookup so the
        # proxy machinery itself keeps working.
        names = object.__getattribute__(self, "_Namespace__names")
        if name in names:
            return object.__getattribute__(self, "_Namespace__call_for")(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str):
        # Reached only when normal lookup found nothing (declared names were
        # already intercepted above), so this is always an undeclared tool.
        raise AttributeError(
            f"tool {name!r} is not declared in namespace "
            f"{object.__getattribute__(self, '_Namespace__global')!r}"
        )

    def __getitem__(self, name: str):
        names = object.__getattribute__(self, "_Namespace__names")
        if name not in names:
            raise KeyError(
                f"tool {name!r} is not declared in namespace "
                f"{object.__getattribute__(self, '_Namespace__global')!r}"
            )
        return object.__getattribute__(self, "_Namespace__call_for")(name)


def _encode_json_plain(value: Any) -> str:
    """Encode validated JSON-plain values iteratively using host-compatible scalar spelling."""

    # O(DEPTH) auxiliary space, not O(width). A container pushes ONE cursor frame
    # that pulls its children one at a time and writes each into the shared buffer,
    # rather than one stack entry (plus a separator marker) per child: a flat
    # `[0] * 6_000_000` encodes to ~12 MB but per-element frames are ~400 MB — an
    # RLIMIT_AS death on a value `_check_done_value` already admitted (it walks by
    # depth as well). The output string is the only width-proportional
    # allocation, and its size the caller metered within budget. `io.StringIO`
    # accumulates without the intermediate `"".join(chunks)` second copy. A cursor
    # frame is [kind, iterator, wrote_any]; a visit frame is (VISIT, value).
    buffer = io.StringIO()
    exhausted = object()
    visit, list_cursor, dict_cursor = 0, 1, 2
    stack: list[Any] = [(visit, value)]
    while stack:
        frame = stack.pop()
        kind = frame[0]
        if kind == list_cursor:
            iterator, wrote_any = frame[1], frame[2]
            child = next(iterator, exhausted)
            if child is exhausted:
                buffer.write("]")
                continue
            if wrote_any:
                buffer.write(",")
            else:
                frame[2] = True
            stack.append(frame)
            stack.append((visit, child))
            continue
        if kind == dict_cursor:
            iterator, wrote_any = frame[1], frame[2]
            entry = next(iterator, exhausted)
            if entry is exhausted:
                buffer.write("}")
                continue
            key, item = entry
            if wrote_any:
                buffer.write(",")
            else:
                frame[2] = True
            buffer.write(_dump_scalar(key))
            buffer.write(":")
            stack.append(frame)
            stack.append((visit, item))
            continue
        current = frame[1]
        current_type = type(current)
        if current_type is list or current_type is tuple:
            buffer.write("[")
            stack.append([list_cursor, iter(current), False])
        elif current_type is dict:
            buffer.write("{")
            stack.append([dict_cursor, iter(dict.items(current)), False])
        else:
            buffer.write(_dump_scalar(current))
    return buffer.getvalue()


def _dump_scalar(value: Any) -> str:
    """One scalar as compact JSON, byte-compatible with the host's encoder.

    ``ensure_ascii=False`` keeps non-ASCII text as raw UTF-8 — the default
    backslash-u escaping would make the child count ``"é"`` as 8 bytes where
    the host meter (and the worker backend) count its UTF-8 JSON form as 4,
    splitting the budget the two sides are supposed to share. Strings route
    through :func:`_dump_string`, which restores the escaping for the one class
    of character UTF-8 cannot hold. Floats route through :func:`_dump_float`
    because CPython's ``repr`` and ECMAScript's Number-to-String disagree on
    spelling.

    Dispatch is on EXACT type, matching the validators: a ``float`` subclass
    reaching :func:`_dump_float` would have its overridden ``__repr__`` read as
    the number's digits, so ``F(2.5)`` whose ``__repr__`` returns ``"1.0"``
    would serialize as ``1``. ``json.dumps`` then refuses any subclass by
    ``TypeError`` instead of emitting a value nothing validated; the callers
    reject subclasses first, so this is the encoder refusing to be the place a
    validation gap turns into corrupted output.
    """

    if type(value) is float:
        return _dump_float(value)
    if type(value) is str:
        return _dump_string(value)
    if value is None or type(value) is bool or type(value) is int:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise TypeError(f"unsupported type ({type(value).__name__})")


_SURROGATE = re.compile("[\ud800-\udfff]")


_SURROGATE_PAIR = re.compile("[\ud800-\udbff][\udc00-\udfff]")


def _combine_surrogate_pair(match: re.Match[str]) -> str:
    """Fold one spelled-out high-low pair into the astral code point it names."""

    high, low = match.group(0)
    return chr(0x10000 + ((ord(high) - 0xD800) << 10) + (ord(low) - 0xDC00))


def _dump_string(text: str) -> str:
    """One string as compact JSON, byte-identical to the host's ``JSON.stringify``.

    ``ensure_ascii=False`` cannot render a surrogate code unit: UTF-8 has no
    encoding for one, so the frame write would raise and the run would strand
    until the wall clock. JSON carries it as the ASCII escape ``\\ud800``, which
    the host's ``JSON.parse`` reads back as the same UTF-16 code unit and its
    ``JSON.stringify`` re-emits identically — so the shared seam
    (``PtcJsonValue``, ``snapshotJsonValue``, the worker backend) keeps a
    lone-surrogate string instead of failing the value. An adjacent high-low
    pair is folded into its astral code point FIRST: the host holds strings as
    UTF-16, where those two code units and the single character are the same
    string, and the raw 4-byte form is what the host would emit — escaping the
    halves separately would charge 12 bytes against a budget the host meters at
    4. Every remaining surrogate is lone and becomes six ASCII bytes, matching
    the host exactly.
    @param text: the string to encode.
    @return: its compact JSON form, always UTF-8-encodable.
    """

    rendered = json.dumps(text, ensure_ascii=False)
    if _SURROGATE.search(rendered) is None:
        return rendered
    return _SURROGATE.sub(
        lambda match: "\\u%04x" % ord(match.group(0)),
        _SURROGATE_PAIR.sub(_combine_surrogate_pair, rendered),
    )


def _dump_float(value: float) -> str:
    """One finite float in ECMAScript ``Number::toString`` spelling.

    CPython's ``repr`` and the host's ``String(number)`` name the same double
    differently: ``1.0`` is ``"1.0"`` here but ``"1"`` there, ``1e-07`` pads the
    exponent the host writes as ``1e-7``, and ``1e+21``/``2**60`` differ again.
    Since the child meters the completion value against ``maxValueBytes`` and
    the host re-meters the frame it parses, any spelling difference splits the
    shared budget: ``return 1.0`` under ``maxValueBytes: 1`` used to be reported
    as ``output-limit`` by the child while the host would have counted the
    one-byte ``1`` it actually receives. Both sides also emit these bytes (the
    child through :func:`_encode_json_plain`, the host through
    ``encodeJsonPlain``), so the fix has to be in the shared speller, not in the
    meter.

    Implements ECMA-262 ``Number::toString`` radix 10 directly: ``repr``
    already yields the shortest round-tripping decimal digits, and ``Decimal``
    splits them into the significand ``s`` (``digits``, ``k`` of them) and
    decimal exponent ``n`` the spec's cases select on. The integral values above
    the JS safe range take the host's BigInt branch, whose exact digits differ
    from the shortest-round-trip form (``2**60`` prints ``...846976``, not
    ``...847000``).
    """

    if value != value or value in (float("inf"), float("-inf")):
        # json.dumps(allow_nan=False) raises the same way; the callers reject
        # non-finite floats before metering, so this is unreachable defense.
        raise ValueError("Out of range float values are not JSON compliant")
    if value == 0.0:
        # Covers -0.0 too; callers reject it as non-lossless before this point.
        return "0"
    if value < 0:
        return "-" + _dump_float(-value)
    if value.is_integer() and value > float(2**53 - 1):
        # The host's BigInt branch: exact digits, not shortest-round-trip.
        return str(int(value))
    parts = Decimal(repr(value)).normalize(context=_FLOAT_CONTEXT).as_tuple()
    digits = "".join(str(digit) for digit in parts.digits)
    k = len(digits)
    n = parts.exponent + k
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * -n + digits
    exponent = ("+" if n - 1 >= 0 else "-") + str(abs(n - 1))
    return (digits if k == 1 else digits[0] + "." + digits[1:]) + "e" + exponent


def _lossless_json_violation(value: Any) -> str | None:
    """Return why a value cannot round-trip as JSON through the host, or None."""

    # The canonical boundary accepts every JS-double-exact value: an int
    # outside +-2**53-1 is fine IFF the double round-trip is exact (2**53 or
    # 2**60 survive; 2**53+1 rounds), matching the worker backend.
    js_safe = 2**53 - 1
    # Post-order walk with an explicit "leave" marker: a container's id is added
    # to `on_path` when entered and removed when left, so a back-edge to an
    # ancestor (a cycle) is detected without rejecting a legitimately shared
    # acyclic subtree.
    on_path: set[int] = set()
    # O(DEPTH) auxiliary space, not O(width), for the reason
    # :func:`_lossless_json_violation` documents: this walk runs in ``dispatch`` on
    # MODEL-CONSTRUCTED binding arguments, which no child-side byte budget
    # bounds first (the frame ceiling is the host's, and it applies after this
    # returns). Enqueueing one frame per member would let a legitimate
    # ``[0] * 6_000_000`` argument -- ~17 MB of JSON -- allocate ~366 MB of
    # traversal tuples and die as the program's own MemoryError instead of
    # round-tripping. A container therefore pushes ONE cursor frame holding its
    # iterator; children are pulled one at a time.
    exhausted = object()
    visit, container_cursor = 0, 1
    # A visit frame is (visit, value, None, None); a cursor frame is
    # (cursor, container, iterator, seen-keys-for-dicts).
    stack: list[tuple[int, Any, Any, Any]] = [(visit, value, None, None)]
    while stack:
        kind = stack[-1][0]
        if kind == container_cursor:
            _, container, iterator, seen = stack[-1]
            child = next(iterator, exhausted)
            if child is exhausted:
                # Leaving the container: it is no longer on the current path, so
                # a legitimately shared acyclic subtree is not mistaken for a cycle.
                on_path.discard(id(container))
                stack.pop()
                continue
            if type(container) is dict:
                # The dict cursor yields (key, value): check the key as it is
                # pulled. Only an EXACT str key survives -- int, float, None, and
                # tuple keys coerce or raise, and a str subclass can carry
                # overrides the encoder does not honor.
                key, child = child
                if type(key) is not str:
                    return f"non-string dict key ({type(key).__name__})"
                # The key's JSON form folds a spelled-out surrogate pair into its
                # astral code point (`_dump_string`), so two DIFFERENT Python
                # keys -- the two code units and the single character -- encode
                # to the same JSON member, and the host's JSON.parse silently
                # drops one of them. A lossless-JSON violation, rejected here
                # before any encoding.
                combined_key = _SURROGATE_PAIR.sub(_combine_surrogate_pair, key)
                if combined_key in seen:
                    return "duplicate dict key after surrogate-pair combining"
                seen.add(combined_key)
            stack.append((visit, child, None, None))
            continue
        _, current, _unused, _unused2 = stack.pop()
        if current is None or type(current) is bool:
            continue
        if type(current) is str:
            # Every string is lossless JSON. A lone surrogate has no UTF-8 form,
            # but JSON carries the code unit as its ASCII ``\uXXXX`` escape and
            # :func:`_dump_string` emits exactly that, so the host receives the
            # same code unit the program passed — the same acceptance
            # ``PtcJsonValue``, ``snapshotJsonValue``, and the worker backend
            # already give it.
            continue
        if type(current) is int:
            if current > js_safe or current < -js_safe:
                try:
                    exact = int(float(current)) == current
                except OverflowError:
                    exact = False
                if not exact:
                    return "integer not exactly representable as a JavaScript number"
            continue
        if type(current) is float:
            if current != current or current in (float("inf"), float("-inf")):
                return "non-finite float"
            # JSON serialization turns -0.0 into 0 (or "-0.0" text that the
            # host parses to JS -0), silently changing the sign bit either
            # way; the repository's canonical lossless-JSON boundary and the
            # worker backend both reject it, so this side must too.
            if current == 0.0 and math.copysign(1.0, current) < 0:
                return "negative zero"
            continue
        if type(current) is list or type(current) is dict:
            if id(current) in on_path:
                return "circular reference"
            on_path.add(id(current))
            if type(current) is dict:
                # Keys are checked as the cursor pulls each entry, not in a
                # separate pass: ``current.values()`` would need a second walk,
                # and materializing ``items()`` up front allocates one tuple per
                # member -- the very spike the cursor removes. The seen-set holds
                # one combined key per member -- O(keys), the same order as the
                # dict itself -- for the surrogate-collision check.
                stack.append((container_cursor, current, iter(current.items()), set()))
            else:
                stack.append((container_cursor, current, iter(current), None))
            continue
        return f"unsupported type ({type(current).__name__})"
    return None

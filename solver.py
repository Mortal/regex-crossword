import ast
import string
from typing import Any, Tuple, Optional, List, Iterable, Dict


EMPTY: Tuple[str, Any] = ("concat", [])


def parse_pattern(s: str) -> Tuple[str, Any]:
    alphabet = string.ascii_uppercase

    def parse_atom(i: int) -> Tuple[Tuple[str, Any], int]:
        c = s[i]
        if c == ".":
            return (".", i), i + 1
        if c == "[":
            if s[i + 1] == "^":
                neg = True
                j = i + 2
            else:
                neg = False
                j = i + 1
            k = s.find("]", j)
            assert k != -1, "Parse error"
            return ("chargroup", (neg, s[j:k])), k + 1
        if c == "(":
            x, j = parse_alternates(i + 1)
            assert s[j] == ")", "Parse error"
            return ("capture", x), j + 1
        if c == "\\":
            e = s[i + 1]
            if e in string.digits:
                return ("backref", int(e)), i + 2
            raise Exception(f"Parse error: Unexpected escape sequence '{c}{e}' at {i}")
        if c in alphabet:
            return ("atom", (s[i], i)), i + 1
        raise Exception(f"Parse error: Unexpected character '{c}' at {i}")

    def parse_alternates(i: int) -> Tuple[Tuple[str, Any], int]:
        xs = []
        x, i = parse_concat(i)
        while i < len(s) and s[i] == "|":
            xs.append(x)
            x, i = parse_concat(i + 1)
        if xs:
            return ("alternates", xs + [x]), i
        return x, i

    def parse_concat(i: int) -> Tuple[Tuple[str, Any], int]:
        xs = []
        x, j = parse_rep(i)
        while j < len(s) and s[j] not in (")", "|"):
            xs.append(x)
            x, j = parse_concat(j)
        assert i != j
        if xs:
            return ("concat", xs + [x]), j
        return x, j

    def parse_rep(i: int) -> Tuple[Tuple[str, Any], int]:
        x, j = parse_atom(i)
        assert j > i, (x, i, s[i:])
        if j < len(s) and s[j] in ("?", "*", "+"):
            return ("rep", (s[j], x)), j + 1
        return x, j

    return parse_alternates(0)[0]


def parse_crossword() -> Tuple[List[str], List[str], List[str], int]:
    group: Optional[List[str]] = None
    x: List[str] = []
    y: List[str] = []
    z: List[str] = []
    size: Optional[int] = None
    with open("crossword.js") as fp:
        for line in fp:
            if line == "  x: [\n":
                group = x
            elif line == "  y: [\n":
                group = y
            elif line == "  z: [\n":
                group = z
            elif line == "  ],\n":
                group = None
            elif line == "};\n":
                break
            elif line.startswith("  size: "):
                size = int(line.split()[1].strip(","))
            elif group is not None:
                group.append(ast.literal_eval(line.strip(" ,\n")))
    assert size is not None
    return x, y, z, size


def enumerate_pattern(tree: Tuple[str, Any], char_vars: List[str], constraints: Dict[str, str]) -> Iterable[str]:
    r'''
    >>> def help(p, c):
    ...     tree = parse_pattern(p)
    ...     char_vars = [str(i) for i in range(len(c))]
    ...     constraints = {str(i): b for i, b in enumerate(c) if b != "_"}
    ...     seen = set()
    ...     for x in enumerate_pattern(tree, char_vars, constraints):
    ...         if x in seen:
    ...             continue
    ...         seen.add(x)
    ...         print(x)
    >>> help(".", "_")
    .
    >>> help(".", "A")
    A
    >>> help(r"(RR|HHH)*.?", "RRRRHHHRRU")
    RRRRHHHRRU
    >>> help(r"[^C]*[^R]*III.*", "HRXRCMIIIHXLS")
    HRXRCMIIIHXLS
    >>> help(r"C*MC(CCC|MM)*", "CMCCCCMMMMMM")
    CMCCCCMMMMMM
    >>> help(r"N.*X.X.X.*E", "NCXDXEXLE")
    NCXDXEXLE
    >>> help("R*D*", "RRRRRRR_")
    RRRRRRRR
    RRRRRRRD
    >>> help("R*D*M*", "RRRRRRR_")
    RRRRRRRR
    RRRRRRRD
    RRRRRRRM
    '''

    prefix: List[str] = []
    stack = [tree]
    captured = []

    def visit() -> Iterable[str]:
        pos = len(prefix)
        if not stack:
            if pos == len(char_vars):
                # Match!
                yield "".join(prefix)
            return
        if stack[-1][0] == "rep":
            kind, inner = stack[-1][1]
            # print("Rep %s at %r" % (kind, "".join(prefix)))
            if kind == "?":
                tree = stack.pop()
                stack.append(inner)
                yield from visit()
                stack.pop()
                yield from visit()
                stack.append(tree)
            elif kind == "*":
                stack.append(inner)
                yield from visit()
                stack.pop()
                tree = stack.pop()
                yield from visit()
                stack.append(tree)
            elif kind == "+":
                tree = stack.pop()
                stack.append(inner)
                stack.append(("rep", ("*", inner)))
                yield from visit()
                stack.pop()
                yield from visit()
                stack.pop()
                stack.append(tree)
            else:
                raise Exception("Unknown rep kind %r" % (kind,))
            return
        if stack[-1][0] == "noanchor":
            tree = stack.pop()
            yield "".join(prefix)
            stack.append(tree)
            return
        if stack[-1][0] == "capture":
            save = stack[:]
            stack[:] = [("noanchor", None), save[-1][1]]
            for p in visit():
                assert not stack
                captured.append(p[pos:])
                stack[:] = save[:-1]
                yield from visit()
                del stack[:]
                captured.pop()
            stack[:] = save
            return
        if stack[-1][0] == "backref":
            n = stack[-1][1]
            if n - 1 >= len(captured):
                raise Exception("Backreference \\%s not captured" % n)
            s = captured[n - 1]
            save = stack[:]
            stack.pop()
            stack.extend([("atom", c) for c in s[::-1]])
            yield from visit()
            stack[:] = save
            return
        if stack[-1][0] == "alternates":
            tree = stack.pop()
            for x in tree[1]:
                stack.append(x)
                yield from visit()
                stack.pop()
            stack.append(tree)
            return
        if stack[-1][0] == "concat":
            save = stack[:]
            tree = stack.pop()
            stack.extend(tree[1][::-1])
            yield from visit()
            stack[:] = save
            return
        if pos == len(char_vars):
            # print("No match: end of string, but pattern is not done", stack)
            return
        assert pos < len(char_vars)
        if stack[-1][0] == ".":
            tree = stack.pop()
            if char_vars[pos] in constraints:
                prefix.append(constraints[char_vars[pos]])
            else:
                prefix.append(".")
            yield from visit()
            stack.append(tree)
            prefix.pop()
            return
        if stack[-1][0] == "atom":
            tree = stack.pop()
            if constraints.get(char_vars[pos]) in (None, tree[1][0]):
                prefix.append(tree[1][0])
                yield from visit()
                prefix.pop()
            stack.append(tree)
            return
        if stack[-1][0] == "chargroup":
            tree = stack.pop()
            neg, chars = tree[1]
            constraint = constraints.get(char_vars[pos])
            if neg:
                if constraint is None or constraint not in chars:
                    prefix.append(constraint or "^")
                    yield from visit()
                    prefix.pop()
            else:
                if constraint is None:
                    for c in chars:
                        prefix.append(c)
                        yield from visit()
                        prefix.pop()
                elif constraint in chars:
                    prefix.append(constraint or "^")
                    yield from visit()
                    prefix.pop()
            stack.append(tree)
            return
        raise Exception("Unknown tree type %r" % (stack,))

    return visit()


def make_xvars(size: int) -> List[List[str]]:
    v = []
    for i in range(size):
        if i < size // 2:
            v.append(["r%sc%s" % (j, i) for j in range(size // 2 + i + 1)])
        else:
            v.append(["r%sc%s" % (j, i) for j in range(i - size // 2, size)])
    return v


def make_yvars(size: int) -> List[List[str]]:
    v = []
    for i in range(size):
        if i < size // 2:
            v.append(["r%sc%s" % (i, j) for j in range(size // 2 + i + 1)])
        else:
            v.append(["r%sc%s" % (i, j) for j in range(i - size // 2, size)])
    return v


def make_zvars(size: int) -> List[List[str]]:
    v = []
    for i in range(size):
        if i < size // 2:
            v.append(["r%sc%s" % (j, size // 2 - i + j) for j in range(size // 2 + i + 1)])
        else:
            v.append(["r%sc%s" % (j, size // 2 - i + j) for j in range(i - size // 2, size)])
    return v


def main() -> None:
    x, y, z, size = parse_crossword()
    assert size % 2 == 1
    assert len(x) == len(y) == len(z) == size
    xvars = make_xvars(size)
    yvars = make_yvars(size)
    zvars = make_zvars(size)
    plen = max(max(map(len, g)) for g in (x, y, z))
    for g, vs in ((x, xvars), (y, yvars), (z, zvars)):
        assert len(g) == len(vs)
        for i, (char_vars, p) in enumerate(zip(vs, g)):
            n = size - abs(i - len(g)//2)
            assert len(set(char_vars)) == len(char_vars) == n, (len(char_vars), n)
            try:
                tree = parse_pattern(p)
                match = next(iter(enumerate_pattern(tree, char_vars, {})))
                print(p.ljust(plen), match)
            except Exception:
                print(p)
                raise


if __name__ == "__main__":
    main()

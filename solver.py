import ast
import string
from typing import Any, Tuple, Optional, List


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


def parse_crossword() -> Tuple[List[str], List[str], List[str]]:
    group: Optional[List[str]] = None
    x: List[str] = []
    y: List[str] = []
    z: List[str] = []
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
            elif group is not None:
                group.append(ast.literal_eval(line.strip(" ,\n")))
    return x, y, z


def main() -> None:
    x, y, z = parse_crossword()
    for p in x + y + z:
        try:
            print(parse_pattern(p))
        except Exception:
            print(p)
            raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Complete educational SIC/XE two-pass assembler.

Supports instruction formats 1-4, SIC/XE addressing modes, literals,
program blocks, control sections/external references, macros, and H/D/R/T/M/E
object records.  It intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


OPTAB = {
    "ADD": (0x18,3), "ADDF": (0x58,3), "ADDR": (0x90,2), "AND": (0x40,3),
    "CLEAR": (0xB4,2), "COMP": (0x28,3), "COMPF": (0x88,3), "COMPR": (0xA0,2),
    "DIV": (0x24,3), "DIVF": (0x64,3), "DIVR": (0x9C,2), "FIX": (0xC4,1),
    "FLOAT": (0xC0,1), "HIO": (0xF4,1), "J": (0x3C,3), "JEQ": (0x30,3),
    "JGT": (0x34,3), "JLT": (0x38,3), "JSUB": (0x48,3), "LDA": (0x00,3),
    "LDB": (0x68,3), "LDCH": (0x50,3), "LDF": (0x70,3), "LDL": (0x08,3),
    "LDS": (0x6C,3), "LDT": (0x74,3), "LDX": (0x04,3), "LPS": (0xD0,3),
    "MUL": (0x20,3), "MULF": (0x60,3), "MULR": (0x98,2), "NORM": (0xC8,1),
    "OR": (0x44,3), "RD": (0xD8,3), "RMO": (0xAC,2), "RSUB": (0x4C,3),
    "SHIFTL": (0xA4,2), "SHIFTR": (0xA8,2), "SIO": (0xF0,1), "SSK": (0xEC,3),
    "STA": (0x0C,3), "STB": (0x78,3), "STCH": (0x54,3), "STF": (0x80,3),
    "STI": (0xD4,3), "STL": (0x14,3), "STS": (0x7C,3), "STSW": (0xE8,3),
    "STT": (0x84,3), "STX": (0x10,3), "SUB": (0x1C,3), "SUBF": (0x5C,3),
    "SUBR": (0x94,2), "SVC": (0xB0,2), "TD": (0xE0,3), "TIO": (0xF8,1),
    "TIX": (0x2C,3), "TIXR": (0xB8,2), "WD": (0xDC,3),
}
REGISTERS = {"A":0, "X":1, "L":2, "B":3, "S":4, "T":5, "F":6, "PC":8, "SW":9}
DIRECTIVES = {"START","END","BYTE","WORD","RESB","RESW","BASE","NOBASE",
              "EQU","ORG","LTORG","USE","CSECT","EXTDEF","EXTREF","MACRO","MEND"}


def is_number(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?(?:\d+|0[xX][0-9A-Fa-f]+|[0-9A-Fa-f]+[Hh])", text.strip()))


def number(text: str, radix: int = 10) -> int:
    value = text.strip()
    sign = -1 if value.startswith("-") else 1
    if value[:1] in "+-":
        value = value[1:]
    if value.lower().startswith("0x"):
        return sign * int(value[2:], 16)
    if value.lower().endswith("h"):
        return sign * int(value[:-1], 16)
    return sign * int(value, radix)


def constant_hex(operand: str) -> str:
    value = operand.strip()
    match = re.fullmatch(r"([CXcx])'([^']*)'", value)
    if not match:
        raise ValueError(f"invalid character/hex constant: {operand}")
    kind, contents = match.groups()
    if kind.upper() == "C":
        return contents.encode("latin-1").hex().upper()
    if len(contents) % 2 or not re.fullmatch(r"[0-9A-Fa-f]*", contents):
        raise ValueError(f"hex constant needs an even number of hex digits: {operand}")
    return contents.upper()


def split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def split_comment(raw: str) -> tuple[str, str]:
    quoted = False
    for index, char in enumerate(raw):
        if char == "'":
            quoted = not quoted
        elif char == "." and not quoted and (index == 0 or raw[index-1].isspace()):
            return raw[:index].rstrip(), raw[index:].strip()
    return raw.rstrip(), ""


@dataclass
class Line:
    source_no: int
    raw: str
    label: str = ""
    op: str = ""
    operand: str = ""
    comment: str = ""
    section: str = ""
    block: str = ""
    offset: int = 0
    address: int = 0
    object_code: str = ""
    generated: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def comment_only(self) -> bool:
        return not self.op


def parse_line(raw: str, source_no: int, macro_names: set[str] | None = None) -> Line:
    macro_names = macro_names or set()
    code, comment = split_comment(raw.rstrip("\n"))
    if not code.strip():
        return Line(source_no, raw.rstrip("\n"), comment=comment or raw.rstrip("\n"))
    begins_space = bool(code[:1].isspace())
    fields = code.strip().split(None, 2)
    first = fields[0].upper()
    test = first[1:] if first.startswith("+") else first
    operation_first = begins_space or test in OPTAB or first in DIRECTIVES or first in macro_names
    if operation_first:
        op = first
        operand = " ".join(fields[1:]) if len(fields) > 1 else ""
        return Line(source_no, raw.rstrip("\n"), op=op, operand=operand, comment=comment)
    label = first
    op = fields[1].upper() if len(fields) > 1 else ""
    operand = fields[2].strip() if len(fields) > 2 else ""
    return Line(source_no, raw.rstrip("\n"), label=label, op=op, operand=operand, comment=comment)


@dataclass
class Macro:
    name: str
    parameters: list[str]
    defaults: dict[str, str]
    body: list[str]


class MacroProcessor:
    """Collect MACRO/MEND definitions and expand their invocations."""

    def __init__(self) -> None:
        self.macros: dict[str, Macro] = {}
        self.counter = 0

    def process(self, source: list[str]) -> list[str]:
        ordinary: list[str] = []
        index = 0
        while index < len(source):
            line = parse_line(source[index], index + 1, set(self.macros))
            if line.op == "MACRO":
                name = line.label
                if not name:
                    raise ValueError(f"line {index+1}: MACRO needs a name in the label field")
                parameters, defaults = [], {}
                for item in split_csv(line.operand):
                    key, sep, default = item.partition("=")
                    key = key.strip().upper()
                    if not key.startswith("&"):
                        key = "&" + key
                    parameters.append(key)
                    if sep:
                        defaults[key] = default.strip()
                body, depth = [], 1
                index += 1
                while index < len(source) and depth:
                    candidate = parse_line(source[index], index + 1, set(self.macros))
                    if candidate.op == "MACRO": depth += 1
                    if candidate.op == "MEND": depth -= 1
                    if depth: body.append(source[index])
                    index += 1
                if depth:
                    raise ValueError(f"macro {name} has no MEND")
                self.macros[name] = Macro(name, parameters, defaults, body)
                continue
            ordinary.append(source[index])
            index += 1

        expanded: list[str] = []
        for source_line in ordinary:
            expanded.extend(self._expand(source_line, 0))
        return expanded

    def _expand(self, raw: str, depth: int) -> list[str]:
        if depth > 20:
            raise ValueError("macro expansion nested more than 20 levels")
        line = parse_line(raw, 0, set(self.macros))
        macro = self.macros.get(line.op)
        if not macro:
            return [raw]
        actual = split_csv(line.operand)
        values = dict(macro.defaults)
        positional = 0
        for item in actual:
            key, sep, val = item.partition("=")
            if sep:
                formal = key.strip().upper()
                if not formal.startswith("&"): formal = "&" + formal
                values[formal] = val.strip()
            elif positional < len(macro.parameters):
                values[macro.parameters[positional]] = item
                positional += 1
        missing = [p for p in macro.parameters if p not in values]
        if missing:
            raise ValueError(f"macro {macro.name} is missing argument(s): {', '.join(missing)}")
        self.counter += 1
        unique = f"M{self.counter:04d}"
        result: list[str] = [f". expanded {macro.name} {line.operand}".rstrip()]
        first = True
        for body_line in macro.body:
            replaced = body_line
            for formal, value in values.items():
                replaced = re.sub(re.escape(formal) + r"\b", value, replaced, flags=re.I)
            replaced = re.sub(r"\$([A-Za-z0-9_]+)", lambda m: f"{unique}_{m.group(1)}", replaced)
            if first and line.label:
                parsed = parse_line(replaced, 0, set(self.macros))
                if not parsed.label and parsed.op:
                    replaced = f"{line.label}\t{parsed.op}\t{parsed.operand}".rstrip()
                first = False
            for nested in self._expand(replaced, depth + 1):
                result.append(nested)
        return result


@dataclass
class Symbol:
    block: str
    offset: int
    value: int = 0
    absolute: bool = False


@dataclass
class Block:
    name: str
    number: int
    location: int = 0
    maximum: int = 0
    start: int = 0


@dataclass
class Literal:
    text: str
    data: str
    block: str = ""
    offset: Optional[int] = None
    address: Optional[int] = None

    @property
    def length(self) -> int:
        return len(self.data) // 2


@dataclass
class Section:
    name: str
    lines: list[Line]
    start: int = 0
    length: int = 0
    end_operand: str = ""
    symbols: dict[str, Symbol] = field(default_factory=dict)
    blocks: dict[str, Block] = field(default_factory=dict)
    literals: dict[str, Literal] = field(default_factory=dict)
    extdef: list[str] = field(default_factory=list)
    extref: list[str] = field(default_factory=list)
    modifications: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def error(self, line: Line, message: str) -> None:
        text = f"line {line.source_no}: {message}"
        line.errors.append(message)
        self.errors.append(text)

    def resolve(self, token: str) -> int:
        key = token.strip().upper()
        if is_number(key): return number(key)
        if key.startswith("="):
            literal = self.literals.get(key)
            if not literal or literal.address is None: raise ValueError(f"undefined literal {token}")
            return literal.address
        symbol = self.symbols.get(key)
        if not symbol: raise ValueError(f"undefined symbol {token}")
        return symbol.value

    def expression(self, expression: str, allow_external: bool = False) -> tuple[int, list[tuple[int,str]]]:
        compact = expression.replace(" ", "").upper()
        if not compact: raise ValueError("missing expression")
        total, externals = 0, []
        parts = re.findall(r"([+-]?)([^+-]+)", compact)
        if not parts: raise ValueError(f"invalid expression {expression}")
        for sign_text, term in parts:
            sign = -1 if sign_text == "-" else 1
            if term in self.extref and allow_external:
                externals.append((sign, term))
            elif term == "*":
                raise ValueError("'*' is only valid in EQU/ORG during pass 1")
            else:
                total += sign * self.resolve(term)
        return total, externals


class Assembler:
    def __init__(self) -> None:
        self.macro_processor = MacroProcessor()
        self.expanded_source: list[str] = []
        self.sections: list[Section] = []

    def assemble(self, source_path: str | Path, output_dir: str | Path = "output") -> dict[str, Path]:
        source_path = Path(source_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = source_path.read_text(encoding="utf-8").splitlines()
        self.expanded_source = self.macro_processor.process(raw)
        parsed = [parse_line(text, index + 1) for index, text in enumerate(self.expanded_source)]
        self.sections = self._split_sections(parsed, source_path.stem.upper()[:6])
        for section in self.sections:
            self._pass_one(section)
            self._pass_two(section)

        stem = source_path.stem
        paths = {
            "expanded": output_dir / f"{stem}.expanded.asm",
            "listing": output_dir / f"{stem}.lst",
            "object": output_dir / f"{stem}.obj",
            "symbols": output_dir / f"{stem}.sym",
        }
        paths["expanded"].write_text("\n".join(self.expanded_source) + "\n", encoding="utf-8")
        paths["listing"].write_text(self._listing(), encoding="utf-8")
        paths["object"].write_text(self._object_program(), encoding="utf-8")
        paths["symbols"].write_text(self._symbol_report(), encoding="utf-8")
        return paths

    def _split_sections(self, lines: list[Line], fallback: str) -> list[Section]:
        groups: list[tuple[str,list[Line]]] = []
        current: list[Line] = []
        current_name = fallback or "NONAME"
        for line in lines:
            if line.op == "CSECT":
                if current: groups.append((current_name, current))
                current_name = line.label or f"CSECT{len(groups)+1}"
                current = [line]
            else:
                if line.op == "START" and line.label: current_name = line.label
                current.append(line)
        if current: groups.append((current_name, current))
        return [Section(name[:6].upper(), group) for name, group in groups]

    def _known_expression(self, section: Section, expression: str, block: str, offset: int) -> int:
        compact = expression.replace(" ", "").upper()
        total = 0
        for sign_text, term in re.findall(r"([+-]?)([^+-]+)", compact):
            sign = -1 if sign_text == "-" else 1
            if term == "*": value = offset
            elif is_number(term): value = number(term)
            elif term in section.symbols:
                symbol = section.symbols[term]
                value = symbol.value if symbol.absolute else symbol.offset
            else: raise ValueError(f"undefined symbol in expression: {term}")
            total += sign * value
        return total

    def _pass_one(self, section: Section) -> None:
        section.blocks = {"": Block("", 0)}
        current_block = ""
        org_stack: list[int] = []
        index = 0
        while index < len(section.lines):
            line = section.lines[index]
            index += 1
            if line.comment_only: continue
            block = section.blocks[current_block]
            line.section, line.block, line.offset = section.name, current_block, block.location
            op = line.op
            if op == "START":
                try: section.start = number(line.operand or "0", 16)
                except ValueError: section.error(line, "START needs a hexadecimal address")
                line.address = section.start
                continue
            if op == "CSECT":
                line.address = 0
                continue
            if op == "USE":
                current_block = line.operand.strip().upper()
                section.blocks.setdefault(current_block, Block(current_block, len(section.blocks)))
                continue
            if line.label and line.label != "*":
                if line.label in section.symbols:
                    section.error(line, f"duplicate symbol {line.label}")
                elif op == "EQU":
                    try:
                        value = self._known_expression(section, line.operand, current_block, block.location)
                        section.symbols[line.label] = Symbol("", 0, value, True)
                    except ValueError as exc: section.error(line, str(exc))
                else:
                    section.symbols[line.label] = Symbol(current_block, block.location)
            if op in {"EQU","BASE","NOBASE","EXTDEF","EXTREF"}:
                if op == "EXTDEF": section.extdef.extend(x.upper() for x in split_csv(line.operand))
                if op == "EXTREF": section.extref.extend(x.upper() for x in split_csv(line.operand))
                continue
            if op == "ORG":
                if not line.operand.strip():
                    if org_stack: block.location = org_stack.pop()
                    else: section.error(line, "ORG restore has no saved address")
                else:
                    try:
                        org_stack.append(block.location)
                        block.location = self._known_expression(section, line.operand, current_block, block.location)
                    except ValueError as exc: section.error(line, str(exc))
                block.maximum = max(block.maximum, block.location)
                continue
            if op in {"LTORG","END"}:
                if op == "END": section.end_operand = line.operand.strip().upper()
                inserted = self._emit_literals(section, index, current_block, block)
                index += inserted
                continue
            try: size = self._statement_size(line)
            except ValueError as exc:
                section.error(line, str(exc)); size = 0
            self._collect_literal(section, line)
            block.location += size
            block.maximum = max(block.maximum, block.location)

        cursor = 0
        for block in section.blocks.values():
            block.start = cursor
            cursor += block.maximum
        section.length = cursor
        for symbol in section.symbols.values():
            if not symbol.absolute:
                symbol.value = section.start + section.blocks[symbol.block].start + symbol.offset
        for line in section.lines:
            if line.comment_only: continue
            if line.op == "START": line.address = section.start
            else: line.address = section.start + section.blocks.get(line.block, section.blocks[""]).start + line.offset
        for literal in section.literals.values():
            if literal.offset is not None:
                literal.address = section.start + section.blocks[literal.block].start + literal.offset

    def _emit_literals(self, section: Section, at: int, block_name: str, block: Block) -> int:
        generated: list[Line] = []
        for literal in section.literals.values():
            if literal.offset is not None: continue
            literal.block, literal.offset = block_name, block.location
            literal_body = literal.text[1:]
            literal_op = "WORD" if literal_body.upper().startswith("W'") else "BYTE"
            literal_operand = literal_body[2:-1] if literal_op == "WORD" else literal_body
            line = Line(0, f"* {literal.text}", label="*", op=literal_op,
                        operand=literal_operand, section=section.name, block=block_name,
                        offset=block.location, generated=True)
            generated.append(line)
            block.location += literal.length
            block.maximum = max(block.maximum, block.location)
        section.lines[at:at] = generated
        return len(generated)

    def _collect_literal(self, section: Section, line: Line) -> None:
        operand = line.operand.strip()
        if operand[:1] in "#@": operand = operand[1:].strip()
        if operand.upper().endswith(",X"): operand = operand[:-2].strip()
        if not operand.startswith("="): return
        key = operand.upper()
        try:
            if key.startswith("=W'"):
                value = number(key[3:-1]); data = f"{value & 0xFFFFFF:06X}"
            elif key == "=*":
                data = f"{line.offset & 0xFFFFFF:06X}"
            else: data = constant_hex(key[1:])
            section.literals.setdefault(key, Literal(key, data))
        except ValueError as exc: section.error(line, str(exc))

    def _statement_size(self, line: Line) -> int:
        op = line.op
        if op.startswith("+"):
            if op[1:] not in OPTAB or OPTAB[op[1:]][1] != 3:
                raise ValueError(f"{op} cannot use format 4")
            return 4
        if op in OPTAB: return OPTAB[op][1]
        if op == "WORD": return 3
        if op == "RESW": return 3 * number(line.operand)
        if op == "RESB": return number(line.operand)
        if op == "BYTE": return len(constant_hex(line.operand)) // 2
        if op in DIRECTIVES: return 0
        raise ValueError(f"unknown operation {op}")

    def _pass_two(self, section: Section) -> None:
        base: Optional[int] = None
        for line in section.lines:
            if line.comment_only: continue
            try:
                if line.op == "BASE": base = section.resolve(line.operand)
                elif line.op == "NOBASE": base = None
                elif line.op == "BYTE": line.object_code = constant_hex(line.operand)
                elif line.op == "WORD":
                    value, external = section.expression(line.operand, True)
                    line.object_code = f"{value & 0xFFFFFF:06X}"
                    for sign, name in external:
                        section.modifications.append(f"M{line.address:06X}06{'+' if sign > 0 else '-'}{name[:6]}")
                elif line.op in DIRECTIVES: pass
                else: line.object_code = self._instruction(section, line, base)
            except ValueError as exc: section.error(line, str(exc))

    def _format_two(self, line: Line, opcode: int) -> str:
        args = [x.strip().upper() for x in line.operand.split(",") if x.strip()]
        def nibble(token: str) -> int:
            if token in REGISTERS: return REGISTERS[token]
            value = number(token)
            if not 0 <= value <= 15: raise ValueError(f"format-2 operand out of range: {token}")
            return value
        r1 = nibble(args[0]) if args else 0
        r2 = nibble(args[1]) if len(args) > 1 else 0
        if line.op in {"SHIFTL","SHIFTR"} and len(args) > 1:
            r2 = (r2 - 1) & 0xF
        if len(args) > 2: raise ValueError("too many operands for format 2")
        return f"{opcode:02X}{r1:X}{r2:X}"

    def _instruction(self, section: Section, line: Line, base: Optional[int]) -> str:
        extended = line.op.startswith("+")
        mnemonic = line.op[1:] if extended else line.op
        if mnemonic not in OPTAB: raise ValueError(f"unknown instruction {line.op}")
        opcode, standard = OPTAB[mnemonic]
        fmt = 4 if extended else standard
        if extended and standard != 3: raise ValueError("only format-3 instructions may use '+'")
        if fmt == 1: return f"{opcode:02X}"
        if fmt == 2: return self._format_two(line, opcode)
        operand = line.operand.strip()
        n = i = 1; x = b = p = 0; e = int(extended)
        if operand.startswith("#"): n, i, operand = 0, 1, operand[1:].strip()
        elif operand.startswith("@"): n, i, operand = 1, 0, operand[1:].strip()
        if operand.upper().endswith(",X"):
            x, operand = 1, operand[:-2].strip()
        numeric = is_number(operand) if operand else False
        external = operand.upper() in section.extref
        target = 0 if not operand or external else section.resolve(operand)
        first = (opcode & 0xFC) | (n << 1) | i
        if fmt == 4:
            field = target & 0xFFFFF
            if external:
                section.modifications.append(f"M{line.address+1:06X}05+{operand.upper()[:6]}")
            elif operand and not numeric:
                section.modifications.append(f"M{line.address+1:06X}05+{section.name[:6]}")
            flags = (x<<3)|(b<<2)|(p<<1)|e
            return f"{((first<<24)|(flags<<20)|field):08X}"
        if external: raise ValueError(f"external reference {operand} requires format 4")
        if not operand: field = 0
        elif numeric and n == 0:
            if not -2048 <= target <= 4095: raise ValueError("immediate value does not fit format 3")
            field = target & 0xFFF
        else:
            displacement = target - (line.address + 3)
            if -2048 <= displacement <= 2047:
                p, field = 1, displacement & 0xFFF
            elif base is not None and 0 <= target - base <= 4095:
                b, field = 1, target - base
            elif 0 <= target <= 0xFFF:
                field = target
            else: raise ValueError("target out of range; set BASE or use format 4")
        flags = (x<<3)|(b<<2)|(p<<1)|e
        return f"{((first<<16)|(flags<<12)|field):06X}"

    @staticmethod
    def _text_records(lines: list[Line]) -> list[str]:
        records, data, start, expected = [], "", None, None
        def flush() -> None:
            nonlocal data, start, expected
            if data and start is not None: records.append(f"T{start:06X}{len(data)//2:02X}{data}")
            data, start, expected = "", None, None
        for line in lines:
            if not line.object_code:
                if line.op in {"RESB","RESW","USE","ORG"}: flush()
                continue
            size = len(line.object_code)//2
            if start is None or line.address != expected or len(data)//2 + size > 30:
                flush(); start = line.address
            data += line.object_code
            expected = line.address + size
        flush()
        return records

    def _object_program(self) -> str:
        output: list[str] = []
        for index, section in enumerate(self.sections):
            output.append(f"H{section.name[:6]:<6}{section.start:06X}{section.length:06X}")
            if section.extdef:
                data = "".join(f"{name[:6]:<6}{section.symbols[name].value:06X}"
                               for name in section.extdef if name in section.symbols)
                if data: output.append("D" + data)
            if section.extref:
                output.append("R" + "".join(f"{name[:6]:<6}" for name in section.extref))
            output.extend(self._text_records(section.lines))
            output.extend(section.modifications)
            execution = ""
            if index == 0:
                try: execution = f"{section.resolve(section.end_operand):06X}" if section.end_operand else f"{section.start:06X}"
                except ValueError: execution = f"{section.start:06X}"
            output.append("E" + execution)
        return "\n".join(output) + "\n"

    def _listing(self) -> str:
        out = ["LINE  SEC    LOC    BLOCK      LABEL      OPERATION  OPERAND                  OBJECT CODE",
               "----  ------ ------ ---------- ---------- ---------- ------------------------ ----------"]
        for section in self.sections:
            for line in section.lines:
                if line.comment_only:
                    out.append(f"{line.source_no:4}                         {line.raw}")
                else:
                    out.append(f"{('*' if line.generated else line.source_no)!s:>4}  {section.name:<6} {line.address:06X} "
                               f"{(line.block or 'DEFAULT'):<10} {line.label:<10} {line.op:<10} "
                               f"{line.operand:<24} {line.object_code}")
                for error in line.errors: out.append(f"**** ERROR: {error}")
            out.append("")
        errors = [error for section in self.sections for error in section.errors]
        out.append(f"Assembly completed with {len(errors)} error(s).")
        return "\n".join(out) + "\n"

    def _symbol_report(self) -> str:
        out = ["SECTION SYMBOL       ADDRESS TYPE  BLOCK", "------- ------------ ------- ----- ----------"]
        for section in self.sections:
            for name, symbol in section.symbols.items():
                out.append(f"{section.name:<7} {name:<12} {symbol.value:06X} "
                           f"{'ABS' if symbol.absolute else 'REL':<5} {symbol.block or 'DEFAULT'}")
            out.append(f"{section.name:<7} {'LENGTH':<12} {section.length:06X}")
        return "\n".join(out) + "\n"

    @property
    def errors(self) -> list[str]:
        return [f"{section.name}: {error}" for section in self.sections for error in section.errors]


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-pass SIC/XE assembler")
    parser.add_argument("sources", nargs="+", help="one or more SIC/XE .txt/.asm source files")
    parser.add_argument("-o", "--output", default="output", help="output directory (default: output)")
    args = parser.parse_args()
    status = 0
    for source in args.sources:
        assembler = Assembler()
        try:
            paths = assembler.assemble(source, args.output)
            print(f"Assembled {source}")
            for label, path in paths.items(): print(f"  {label:8}: {path}")
            if assembler.errors:
                status = 2
                print(f"  errors  : {len(assembler.errors)} (see listing)")
            else: print("  status  : success")
        except Exception as exc:
            status = 2
            print(f"Assembly failed for {source}: {exc}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

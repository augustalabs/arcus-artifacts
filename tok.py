"""Byte-level UTF-8 tokenizer with greedy longest-match special tokens (ode.pt)."""

SPECIALS = [
    ('<|fernando_pessoa|>', 256),
    ('<|alberto_caeiro|>', 257),
    ('<|ricardo_reis|>', 258),
    ('<|bernardo_soares|>', 259),
    ('_', 260),
    ('{', 261),
]
# longest-first for greedy matching
_SPECIALS_SORTED = sorted(SPECIALS, key=lambda kv: -len(kv[0]))
_ID2SPECIAL = {i: s for s, i in SPECIALS}

def encode(text: str):
    b = text.encode('utf-8')
    out = []
    i = 0
    while i < len(b):
        matched = False
        for s, sid in _SPECIALS_SORTED:
            sb = s.encode('utf-8')
            if b[i:i+len(sb)] == sb:
                out.append(sid); i += len(sb); matched = True; break
        if not matched:
            out.append(b[i]); i += 1
    return out

def decode(ids):
    out = bytearray()
    parts = []
    for i in ids:
        if i in _ID2SPECIAL:
            if out:
                parts.append(out.decode('utf-8', errors='replace')); out = bytearray()
            parts.append(_ID2SPECIAL[i])
        else:
            out.append(i)
    if out:
        parts.append(out.decode('utf-8', errors='replace'))
    return ''.join(parts)

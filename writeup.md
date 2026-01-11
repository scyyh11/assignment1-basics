# Problem (unicode1):
(a) `chr(0)` returns `'\x00'`, which is the null character.

(b) Its string representation (\_\_repr\_\_()) returns `'\x00'`, while the printed representation is nothing.

(c) When this character in text, it becomes an invisible embedded null byte but the string still contains it.

# Problem (unicode2):
(a) Because using UTF-16 or UTF-32 will have a longer sequence compared to UTF-8.

(b) Japanese character "こんにちは". Because not all characters are stored in one byte, the method used in the function may cut off a character when decoding.

(c) Example: b"\xC0\xAF"; this is an invalid UTF-8 sequence (an overlong encoding), so it does not decode to any Unicode character(s) under standard UTF-8 decoding rules.

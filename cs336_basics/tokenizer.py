"""
BPE Tokenizer Implementation

This module implements Byte Pair Encoding (BPE) tokenization,
similar to GPT-2's tokenizer.
"""

from __future__ import annotations

import os
import regex as re
import heapq
from functools import lru_cache
from collections import Counter, defaultdict
from typing import Any


def _get_pretokenization_pattern() -> str:
    r"""
    GPT-2 style pre-tokenization regex pattern.
    
    This pattern splits text into tokens that will be used for BPE training.
    Pattern matches:
    - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
    - Words (with optional leading space): ?\p{L}+
    - Numbers (with optional leading space): ?\p{N}+
    - Punctuation (with optional leading space): ?[^\s\p{L}\p{N}]+
    - Whitespace sequences: \s+(?!\S)|\s+
    """
    return r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretokenize(text: str) -> list[str]:
    """
    Pre-tokenize text using GPT-2 style regex pattern.
    
    Args:
        text: Input text string
        
    Returns:
        List of pre-token strings
    """
    pattern = _get_pretokenization_pattern()
    return re.findall(pattern, text)


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer.
    
    This tokenizer encodes text into token IDs using BPE merges,
    and can decode token IDs back to text.
    """
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab: Dictionary mapping token ID to bytes
            merges: List of BPE merges (ordered by creation time)
            special_tokens: List of special token strings
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Build reverse vocabulary: bytes -> token_id
        self.bytes_to_id: dict[bytes, int] = {bytes_val: token_id for token_id, bytes_val in vocab.items()}
        
        # Build merge lookup for efficient encoding
        # This will help us quickly find which merges to apply
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            merge: rank for rank, merge in enumerate(merges)
        }

        # Pre-compile pretoken regex once (speed)
        self._pretoken_re = re.compile(_get_pretokenization_pattern())

        # Pre-compile special token regex + prefixes once (speed)
        self._special_regex = None
        self._special_prefixes: set[str] = set()
        self._max_special_len = 0

        if self.special_tokens:
            special_pattern = "|".join(
                re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_regex = re.compile(special_pattern)

            for token in self.special_tokens:
                self._max_special_len = max(self._max_special_len, len(token))
                # store all proper prefixes to handle chunk boundary in encode_iterable
                for i in range(1, len(token)):
                    self._special_prefixes.add(token[:i])

    
    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        if text == "":
            return []

        bytes_to_id = self.bytes_to_id
        pre_re = self._pretoken_re
        special_re = self._special_regex

        def encode_segment(segment: str) -> list[int]:
            ids: list[int] = []
            for m in pre_re.finditer(segment):
                token_bytes = m.group(0).encode("utf-8")
                ids.extend(self._merge_token_bytes(token_bytes))
            return ids

        # no special tokens
        if special_re is None:
            return encode_segment(text)

        encoded_ids: list[int] = []
        last_idx = 0

        for match in special_re.finditer(text):
            start, end = match.span()
            if start > last_idx:
                encoded_ids.extend(encode_segment(text[last_idx:start]))
            # special token as single id
            encoded_ids.append(bytes_to_id[match.group(0).encode("utf-8")])
            last_idx = end

        if last_idx < len(text):
            encoded_ids.extend(encode_segment(text[last_idx:]))

        return encoded_ids
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        if not token_ids:
            return ""

        vocab = self.vocab
        decoded_bytes = bytearray()
        for token_id in token_ids:
            try:
                decoded_bytes.extend(vocab[token_id])
            except KeyError as exc:
                raise ValueError(f"Unknown token id: {token_id}") from exc

        return decoded_bytes.decode("utf-8", errors="replace")

    @lru_cache(maxsize=200_000)
    def _merge_token_bytes(self, token_bytes: bytes) -> tuple[int, ...]:
        """
        Faster cached BPE merge using heap + linked list.
        token_bytes (UTF-8 bytes) -> tuple of token ids.
        """
        if not token_bytes:
            return ()

        bytes_to_id = self.bytes_to_id
        merge_rank = self.merge_rank

        # start from byte-level tokens
        parts = [bytes([b]) for b in token_bytes]
        n = len(parts)
        if n == 1:
            return (bytes_to_id[parts[0]],)

        # doubly linked list over indices
        prev = list(range(-1, n - 1))      # prev[i] = i-1, prev[0] = -1
        nxt = list(range(1, n)) + [-1]     # nxt[i] = i+1, nxt[n-1] = -1
        alive = [True] * n

        # versioning: bump when a node's bytes changes, so heap entries can be invalidated
        ver = [0] * n

        heap: list[tuple[int, int, int, int]] = []
        # heap entry: (rank, left_i, ver_left, ver_right)
        def push_pair(i: int) -> None:
            j = nxt[i]
            if j == -1 or (not alive[i]) or (not alive[j]):
                return
            r = merge_rank.get((parts[i], parts[j]))
            if r is None:
                return
            heapq.heappush(heap, (r, i, ver[i], ver[j]))

        for i in range(n - 1):
            push_pair(i)

        while heap:
            r, i, vi, vj = heapq.heappop(heap)
            j = nxt[i]
            # validate adjacency & freshness
            if j == -1:
                continue
            if not (alive[i] and alive[j]):
                continue
            if vi != ver[i] or vj != ver[j]:
                continue
            # validate pair still has same rank
            cur_rank = merge_rank.get((parts[i], parts[j]))
            if cur_rank is None or cur_rank != r:
                continue

            # merge i + j into i (keep left node)
            parts[i] = parts[i] + parts[j]
            ver[i] += 1

            # remove j from list
            alive[j] = False
            nj = nxt[j]
            nxt[i] = nj
            if nj != -1:
                prev[nj] = i

            # after merging, only neighbor pairs around i can change
            pi = prev[i]
            if pi != -1:
                push_pair(pi)   # (pi, i)
            push_pair(i)        # (i, nxt[i])

        # collect final token bytes by traversing linked list from head
        head = 0
        while prev[head] != -1:
            head = prev[head]

        out_ids: list[int] = []
        k = head
        while k != -1:
            if alive[k]:
                out_ids.append(bytes_to_id[parts[k]])
            k = nxt[k]

        return tuple(out_ids)
    
    def encode_iterable(self, iterable):
        """
        Memory-efficient encoding for large files.
        
        Args:
            iterable: Iterable of text chunks (e.g., file object)
            
        Yields:
            Token IDs one at a time
        """
        bytes_to_id = self.bytes_to_id
        pretoken_regex = self._pretoken_re

        special_regex = self._special_regex
        special_prefixes = self._special_prefixes if self._special_regex is not None else None
        max_special_len = self._max_special_len

        def emit_pretokens(text: str, keep_last: bool) -> str:
            if not text:
                return ""

            last_start = None
            last_end = None
            for match in pretoken_regex.finditer(text):
                last_start = match.start()
                last_end = match.end()
                if keep_last and last_end == len(text):
                    break
                token_bytes = match.group(0).encode("utf-8")
                for token_id in self._merge_token_bytes(token_bytes):
                    yield token_id


            if keep_last and last_end == len(text):
                return text[last_start:]
            if last_end is None:
                return text
            if last_end < len(text):
                return text[last_end:]
            return ""

        special_tokens = self.special_tokens
        if special_tokens:
            special_pattern = "|".join(
                re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
            )
            special_regex = re.compile(special_pattern)
            special_prefixes = set()
            for token in special_tokens:
                max_special_len = max(max_special_len, len(token))
                for i in range(1, len(token)):
                    special_prefixes.add(token[:i])

        def special_prefix_len(text: str) -> int:
            if not special_prefixes:
                return 0
            max_len = min(max_special_len - 1, len(text))
            for i in range(max_len, 0, -1):
                if text[-i:] in special_prefixes:
                    return i
            return 0

        buffer = ""
        for chunk in iterable:
            if not chunk:
                continue
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            buffer += chunk

            while buffer:
                if special_regex is not None:
                    match = special_regex.search(buffer)
                    if match:
                        before = buffer[:match.start()]
                        if before:
                            leftover = yield from emit_pretokens(before, keep_last=False)
                            if leftover:
                                buffer = leftover + buffer[match.start():]
                                continue
                        yield bytes_to_id[match.group(0).encode("utf-8")]
                        buffer = buffer[match.end():]
                        continue

                    tail_len = special_prefix_len(buffer)
                    segment = buffer[:-tail_len] if tail_len else buffer
                    if segment:
                        leftover = yield from emit_pretokens(segment, keep_last=True)
                    else:
                        leftover = ""
                    buffer = leftover + buffer[len(segment):]
                    break

                leftover = yield from emit_pretokens(buffer, keep_last=True)
                buffer = leftover
                break

        while buffer:
            if special_regex is not None:
                match = special_regex.search(buffer)
                if match:
                    before = buffer[:match.start()]
                    if before:
                        yield from emit_pretokens(before, keep_last=False)
                    yield bytes_to_id[match.group(0).encode("utf-8")]
                    buffer = buffer[match.end():]
                    continue
                yield from emit_pretokens(buffer, keep_last=False)
                buffer = ""
                break

            yield from emit_pretokens(buffer, keep_last=False)
            buffer = ""
    
    def _apply_bpe_merges(self, word_bytes: list[bytes]) -> list[bytes]:
        """
        Apply BPE merges to a word (list of bytes).
        
        Args:
            word_bytes: List of bytes representing a word
            
        Returns:
            List of merged tokens (as bytes)
        """
        if len(word_bytes) == 1:
            return word_bytes

        merge_rank = self.merge_rank
        word = word_bytes[:]

        while True:
            best_pair = None
            best_rank = None
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            if len(word) == 1:
                break

        return word


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # -------------------------
    # 1) init vocab: 256 bytes + special tokens
    # -------------------------
    vocab: dict[int, bytes] = {}
    next_id = 0

    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    # then 256 single-byte tokens
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    target_merges = vocab_size - next_id
    if target_merges < 0:
        raise ValueError("vocab_size too small for 256 bytes + special tokens")

    # -------------------------
    # 2) split on special tokens (hard boundaries)
    # -------------------------
    split_re = None
    if special_tokens:
        delim = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        split_re = re.compile(delim)

    pre_re = re.compile(_get_pretokenization_pattern())

    # -------------------------
    # 3) words as counts: word is tuple[bytes,...] (start as single bytes)
    # -------------------------
    words: dict[tuple[bytes, ...], int] = {}
    wget = words.get

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    segments = [text] if split_re is None else split_re.split(text)
    for seg in segments:
        if not seg:
            continue
        for m in pre_re.finditer(seg):
            bs = m.group(0).encode("utf-8")
            w = tuple(bytes([b]) for b in bs)
            words[w] = wget(w, 0) + 1

    merges: list[tuple[bytes, bytes]] = []

    # helpers (use plain dict for speed)
    def pair_multiset(w: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
        d: dict[tuple[bytes, bytes], int] = {}
        L = len(w)
        if L < 2:
            return d
        prev = w[0]
        for i in range(1, L):
            cur = w[i]
            p = (prev, cur)
            d[p] = d.get(p, 0) + 1
            prev = cur
        return d

    def merge_word(w: tuple[bytes, ...], a: bytes, b: bytes, merged: bytes) -> tuple[bytes, ...]:
        out: list[bytes] = []
        L = len(w)
        i = 0
        while i < L:
            if i < L - 1 and w[i] == a and w[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(w[i])
                i += 1
        return tuple(out)

    # -------------------------
    # 4) build pair_counts and pair_to_words (inverted index)
    # -------------------------
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    pc_get = pair_counts.get
    for w, c in words.items():
        pm = pair_multiset(w)
        for p, k in pm.items():
            pair_counts[p] = pc_get(p, 0) + k * c
            pair_to_words[p].add(w)

    # -------------------------
    # 5) merge loop with incremental updates
    # -------------------------
    for _ in range(target_merges):
        if not pair_counts:
            break

        # IMPORTANT: tie-break exactly like reference: max by (freq, pair_bytes)
        best_pair, best_freq = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        if best_freq <= 0:
            break

        a, b = best_pair
        merged_tok = a + b
        merges.append(best_pair)

        vocab[next_id] = merged_tok
        next_id += 1

        affected = list(pair_to_words.get(best_pair, ()))
        if not affected:
            # should be rare; just delete and continue
            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)
            continue

        # process each affected word
        for w in affected:
            freq = words.get(w)
            if freq is None:
                continue  # may have been merged into another word already

            # old pairs (with multiplicity)
            old_pm = pair_multiset(w)

            # produce new word
            w2 = merge_word(w, a, b, merged_tok)

            # update words dict (merge duplicates)
            words.pop(w, None)
            words[w2] = words.get(w2, 0) + freq

            # new pairs (with multiplicity)
            new_pm = pair_multiset(w2)

            # ---- update pair_counts: subtract old, add new (weighted by freq) ----
            for p, k in old_pm.items():
                nv = pair_counts.get(p, 0) - k * freq
                if nv > 0:
                    pair_counts[p] = nv
                else:
                    pair_counts.pop(p, None)

            for p, k in new_pm.items():
                pair_counts[p] = pair_counts.get(p, 0) + k * freq

            # ---- update pair_to_words (membership sets) ----
            for p in old_pm.keys():
                s = pair_to_words.get(p)
                if s is not None:
                    s.discard(w)
                    if not s:
                        pair_to_words.pop(p, None)

            for p in new_pm.keys():
                pair_to_words[p].add(w2)

        # ensure best_pair mapping cleaned
        s = pair_to_words.get(best_pair)
        if s is not None and not s:
            pair_to_words.pop(best_pair, None)
        if best_pair in pair_counts and pair_counts[best_pair] <= 0:
            pair_counts.pop(best_pair, None)

    return vocab, merges

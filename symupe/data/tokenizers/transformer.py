"""SyMuPeTransformer module that transforms the SyMuPe tokenized sequence between different encoding types."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .symupe import SyMuPe
from .classes import TokSequence, SequenceType, EncodingType, SEQUENCE_TRANSFORMS
from .constants import (
    SCORE_KEYS,
    PLAIN_SCORE_KEYS,
    REL_PERFORMANCE_KEYS,
    TIME_PERFORMANCE_KEYS,
    MASK_TOKEN,
    IGNORE_TOKEN,
    BAR_LINE_TOKEN,
    PEDAL_ON_TOKEN,
    PEDAL_OFF_TOKEN,
)


class SyMuPeTransformer:
    """Utility class to transform :class:`TokSequence` objects between different encoding types.

    Facilitates conversion between metrical score representations and various performance-aligned
    formats by masking irrelevant dimensions and injecting deadpan timing when necessary.
    """

    def __init__(self, tokenizer: SyMuPe):
        """Initializes transformer with a specific SyMuPe tokenizer instance.

        Args:
            tokenizer: :class:`SyMuPe` tokenizer used for encoding/decoding logic.
        """
        self.tokenizer = tokenizer

    def __call__(
        self,
        seq: TokSequence,
        encoding: EncodingType | None = None,
        seq_type: SequenceType | None = None,
        clean_tokens: bool = True,
    ) -> TokSequence:
        """Transforms sequence to target encoding and sequence type.

        Handles decompression, conversion of score tokens to deadpan performance
        representations, and selective removal of special tokens.

        Args:
            seq: :class:`TokSequence` to transform.
            encoding: Target :class:`EncodingType`.
            seq_type: Optional override for :class:`SequenceType`.
            clean_tokens: If ``True``, removes special tokens irrelevant to target encoding.

        Returns:
            Transformed :class:`TokSequence` object.
        """
        seq_type = seq_type or seq.type
        assert encoding in SEQUENCE_TRANSFORMS[seq_type]

        if len(seq.ids) and len(seq.ids[0]) != len(self.tokenizer.performance_sizes):
            seq = self.tokenizer.decompress(seq)

        if seq_type in (SequenceType.TIME_PERFORMANCE, SequenceType.TIME_PERFORMANCE_SUSTAIN):
            seq.encoding = EncodingType.TIME_PERFORMANCE
            return seq

        if (
            seq_type in (SequenceType.SCORE, SequenceType.SYNC_PERFORMANCE)
            and encoding != EncodingType.SCORE
        ):
            seq = self.tokenizer.score_tokens_as_performance(seq)

        seq.encoding = encoding

        if clean_tokens:
            seq = self.remove_special_tokens(seq)

        if encoding == EncodingType.SCORE:
            seq = self.to_score_encoding(seq)
        elif encoding == EncodingType.PLAIN_SCORE:
            seq = self.to_plain_score_encoding(seq)
        elif encoding == EncodingType.REL_PERFORMANCE:
            seq = self.to_relative_performance_encoding(seq)
        elif encoding == EncodingType.TIME_PERFORMANCE:
            seq = self.to_time_performance_encoding(seq)
        elif encoding == EncodingType.SCORE_TIME_PERFORMANCE:
            seq = self.to_score_time_performance_encoding(seq)

        return seq

    def _to_sub_encoding(self, seq: TokSequence, enc_keys: list[str]) -> TokSequence:
        """Internal helper to mask dimensions not present in specified key list.

        Sets IDs and values of excluded dimensions to ignore tokens/values.

        Args:
            seq: :class:`TokSequence` to mask.
            enc_keys: List of dimension names to preserve.

        Returns:
            Masked :class:`TokSequence`.
        """
        for key in self.tokenizer.performance_sizes:
            if key not in enc_keys:
                seq.ids[:, self.tokenizer.vocab_types_idx[key]] = self.tokenizer.ignore_token
                if seq.values is not None:
                    seq.values[:, self.tokenizer.vocab_types_idx[key]] = self.tokenizer.ignore_value

        return seq

    def to_score_encoding(self, seq: TokSequence) -> TokSequence:
        """Converts sequence to standard score-metrical representation.

        Args:
            seq: :class:`TokSequence` to convert.

        Returns:
            Score-encoded :class:`TokSequence`.
        """
        return self._to_sub_encoding(seq, enc_keys=SCORE_KEYS)

    def to_plain_score_encoding(self, seq: TokSequence) -> TokSequence:
        """Converts sequence to plain score representation (without `Tempo` and `Velocity`).

        Args:
            seq: :class:`TokSequence` to convert.

        Returns:
            Plain-score-encoded :class:`TokSequence`.
        """
        return self._to_sub_encoding(seq, enc_keys=PLAIN_SCORE_KEYS)

    def to_deadpan_performance_encoding(self, seq: TokSequence) -> TokSequence:
        """Generates performance representation with zero timing/articulation deviations.

        Args:
            seq: :class:`TokSequence` to convert.

        Returns:
            Deadpan performance :class:`TokSequence`.
        """
        score_seq = self.to_score_encoding(seq)
        score_seq = self.tokenizer.compress(score_seq)
        return self.tokenizer.score_tokens_as_performance(score_seq)

    def to_relative_performance_encoding(self, seq: TokSequence) -> TokSequence:
        """Converts sequence to relative performance representation (`OnsetDev`/`PerfDuration`).

        Args:
            seq: :class:`TokSequence` to convert.

        Returns:
            Relative-performance-encoded :class:`TokSequence`.
        """
        assert self.tokenizer.config.additional_params["use_onset_tokens"]

        return self._to_sub_encoding(seq, enc_keys=REL_PERFORMANCE_KEYS)

    def to_time_performance_encoding(self, seq: TokSequence) -> TokSequence:
        """Converts sequence to absolute time performance representation (`TimeShift`/`TimeDuration`).

        Args:
            seq: :class:`TokSequence` to convert.

        Returns:
            Time-performance-encoded :class:`TokSequence`.
        """
        assert self.tokenizer.config.additional_params["use_time_tokens"]

        return self._to_sub_encoding(seq, enc_keys=TIME_PERFORMANCE_KEYS)

    def to_score_time_performance_encoding(self, seq: TokSequence) -> TokSequence:
        """Converts sequence to a hybrid score and absolute time performance representation.

        Args:
            seq: :class:`TokSequence` to convert.

        Returns:
            Score-time-performance-encoded :class:`TokSequence`.
        """
        assert self.tokenizer.config.additional_params["use_time_tokens"]

        return self._to_sub_encoding(seq, enc_keys=list(set(SCORE_KEYS + TIME_PERFORMANCE_KEYS)))

    def remove_special_tokens(self, seq: TokSequence, force: bool = False):
        """Removes `PEDAL_ON/OFF` or `BAR_LINE` special tokens based on current encoding requirements.

        Args:
            seq: :class:`TokSequence` to clean.
            force: If ``True``, removes both pedal and bar line tokens regardless of encoding.

        Returns:
            Cleaned :class:`TokSequence`.
        """
        if force or seq.encoding in (
            EncodingType.SCORE,
            EncodingType.PLAIN_SCORE,
            EncodingType.REL_PERFORMANCE,
            EncodingType.SCORE_TIME_PERFORMANCE,
        ):
            seq = self.tokenizer.remove_pedal_tokens(seq)

        if force or seq.encoding in (
            EncodingType.TIME_PERFORMANCE,
            EncodingType.SCORE_TIME_PERFORMANCE,
        ):
            seq = self.tokenizer.remove_bar_line_tokens(seq)

        return seq

    def adjust_seq_len_for_special_tokens(
        self,
        seq: TokSequence,
        encodings: Sequence[EncodingType] | None = None,
        offset: int = 0,
        seq_len: int = 256,
    ):
        """Calculates required sequence length to contain a specific number of notes.

        Accounts for special tokens (pedals, bar lines) which vary by encoding.

        Args:
            seq: :class:`TokSequence` to analyze.
            encodings: Optional list of :class:`EncodingType` values to check.
            offset: Starting index in sequence.
            seq_len: Target number of musical notes.

        Returns:
            Dictionary mapping :class:`EncodingType` to total sequence length (in tokens).
        """
        encodings = encodings or (seq.encoding,)

        pitches = seq.ids[:, seq.vocab["Pitch"]][offset:]

        note_mask = pitches >= self.tokenizer.zero_token

        pedal_ids, pedal_mask = self.tokenizer.pedal_ids, None
        if pedal_ids[0] is not None:
            pedal_mask = (pitches == pedal_ids[0]) | (pitches == pedal_ids[1])

        bar_line_id, bar_mask = self.tokenizer.bar_line_id, None
        if bar_line_id is not None:
            bar_mask = pitches == bar_line_id

        seq_len_adjusted = {}
        for encoding in encodings:
            if encoding in seq_len_adjusted:
                continue

            mask = note_mask
            if pedal_mask is not None and encoding not in (
                EncodingType.SCORE,
                EncodingType.PLAIN_SCORE,
                EncodingType.REL_PERFORMANCE,
                EncodingType.SCORE_TIME_PERFORMANCE,
            ):
                mask = mask | pedal_mask

            if bar_mask is not None and encoding not in (
                EncodingType.TIME_PERFORMANCE,
                EncodingType.SCORE_TIME_PERFORMANCE,
            ):
                mask = mask | bar_mask

            ids = np.where(np.cumsum(mask) > seq_len)[0]
            seq_len_adjusted[encoding] = ids[0] if len(ids) > 0 else len(pitches)

        return seq_len_adjusted

    def get_encoding_template(
        self,
        encoding: EncodingType,
        token_types: list[str] | None = None,
        bar_line_token: bool = True,
        pedal_token: bool = True,
        full_ignore: bool = True,
        minimal: bool = False,
    ):
        """Generates template tokens (placeholders) for a specific encoding.

        Useful for creating masked prompts or initializing generation buffers.

        Args:
            encoding: Target :class:`EncodingType` for template.
            token_types: Optional list of dimensions to keep after compression.
            bar_line_token: Whether to include bar line placeholder.
            pedal_token: Whether to include pedal placeholders.
            full_ignore: If ``True``, uses ignore tokens for irrelevant dimensions.
            minimal: If ``True``, omits placeholder special tokens if not required by encoding.

        Returns:
            :class:`TokSequence` containing template tokens.
        """
        vocab = self.tokenizer.vocab_types_idx

        tokens = []

        def _base_token(_token_name):
            token = np.full(len(vocab), fill_value=self.tokenizer.ignore_token)

            token[vocab["Pitch"]] = self.tokenizer[0, _token_name]
            if self.tokenizer.config.additional_params["use_pitch_classes"]:
                token[vocab["PitchClass"]] = self.tokenizer[0, _token_name]
                token[vocab["PitchOctave"]] = self.tokenizer[0, _token_name]

            return token

        if bar_line_token and BAR_LINE_TOKEN in self.tokenizer.special_tokens:
            token = _base_token(BAR_LINE_TOKEN)

            if encoding not in (EncodingType.TIME_PERFORMANCE, EncodingType.SCORE_TIME_PERFORMANCE):
                token[vocab["Bar"]] = self.tokenizer[0, MASK_TOKEN]
                token[vocab["Position"]] = self.tokenizer[0, MASK_TOKEN]
                if self.tokenizer.config.additional_params["use_position_shifts"]:
                    token[vocab["PositionShift"]] = self.tokenizer[0, MASK_TOKEN]
                tokens.append(token)
            elif not minimal:
                tokens.append(_base_token(IGNORE_TOKEN) if full_ignore else token)

        if (
            pedal_token
            and PEDAL_ON_TOKEN in self.tokenizer.special_tokens
            and PEDAL_OFF_TOKEN in self.tokenizer.special_tokens
        ):
            for token_name in [PEDAL_ON_TOKEN, PEDAL_OFF_TOKEN]:
                token = _base_token(token_name)

                if encoding not in (
                    EncodingType.SCORE,
                    EncodingType.PLAIN_SCORE,
                    EncodingType.REL_PERFORMANCE,
                    EncodingType.SCORE_TIME_PERFORMANCE,
                ):
                    if self.tokenizer.config.additional_params["use_time_tokens"]:
                        token[vocab["TimeShift"]] = self.tokenizer[0, MASK_TOKEN]
                    tokens.append(token)
                elif not minimal:
                    tokens.append(_base_token(IGNORE_TOKEN) if full_ignore else token)

        token = np.full(len(vocab), fill_value=self.tokenizer[0, MASK_TOKEN])
        token = self(
            TokSequence(ids=token[None], type=SequenceType.PERFORMANCE),
            encoding=encoding,
            clean_tokens=False,
        )[0]
        tokens.append(token)

        tokens = TokSequence(
            ids=np.stack(tokens, axis=0), type=SequenceType.PERFORMANCE, encoding=encoding
        )
        tokens.values = self.tokenizer.decode_values(tokens.ids)

        if token_types is not None:
            tokens = self.tokenizer.compress(tokens, token_types=token_types)

        return tokens

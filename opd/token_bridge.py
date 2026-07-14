"""Bridging the student (ChatML, vocab 128262) and teacher (Llama-3, vocab 128256).

Both tokenizers share the Llama-3 base vocabulary (ids 0..128255). The student
adds 6 ChatML/tool tokens at 128256..128261 which the teacher cannot embed.
Prompts are therefore rendered per-model with each model's own chat template;
only completion tokens (base vocab) are aligned across the two models.

The student's <|im_end|> (128256) and the teacher's <|eot_id|> (128009) both
mean "end of turn", so for scoring we merge the student's <|im_end|> probability
mass into the <|eot_id|> slot — that way the teacher also supervises *when to
stop*, not just what to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SHARED_VOCAB_SIZE = 128256

STUDENT_IM_END = 128256  # <|im_end|>
TEACHER_EOT_ID = 128009  # <|eot_id|>
END_OF_TEXT = 128001     # <|end_of_text|> (shared, valid for both)


@dataclass
class Bridge:
    shared_vocab_size: int = SHARED_VOCAB_SIZE
    # student id -> teacher id, applied when a completion token is fed to the
    # teacher or used as a scoring target in the shared vocab.
    swap: dict[int, int] = field(default_factory=lambda: {STUDENT_IM_END: TEACHER_EOT_ID})
    # completion is truncated at the first of these (kept, as terminal target)
    stop_ids: tuple[int, ...] = (STUDENT_IM_END, END_OF_TEXT)

    def clean_completion(self, ids: list[int]) -> list[int]:
        """Truncate a raw sampled completion for scoring.

        Cuts at the first stop token (inclusive: stopping is supervised too).
        Any other out-of-shared-vocab token (tool tokens etc.) truncates the
        completion *before* it — the teacher has no equivalent to score.
        """
        out = []
        for t in ids:
            if t in self.stop_ids:
                out.append(t)
                break
            if t >= self.shared_vocab_size:
                break
            out.append(t)
        return out

    def to_teacher(self, ids: list[int]) -> list[int]:
        return [self.swap.get(t, t) for t in ids]


def assert_compatible(student_tok, teacher_tok, verbose: bool = True) -> None:
    """Hard preconditions for exact per-token KL. Raises on violation."""
    probes = [
        "Rispondi alla seguente domanda a scelta multipla sull'argomento 'storia'.",
        "La Divina Commedia è composta da tre cantiche: Inferno, Purgatorio e Paradiso.",
        "Perché l'ossigeno è più elettronegativo dell'azoto? Risposta: A",
        "x = [n**2 for n in range(10)]  # città, però, così",
    ]
    for text in probes:
        s_ids = student_tok.encode(text, add_special_tokens=False)
        t_ids = teacher_tok.encode(text, add_special_tokens=False)
        if s_ids != t_ids:
            raise AssertionError(
                f"Tokenizers diverge on shared text!\n{text!r}\nstudent={s_ids}\nteacher={t_ids}"
            )
    t_vocab = len(teacher_tok)
    if t_vocab != SHARED_VOCAB_SIZE:
        raise AssertionError(f"Teacher vocab {t_vocab} != expected {SHARED_VOCAB_SIZE}")
    im_end = student_tok.convert_tokens_to_ids("<|im_end|>")
    if im_end != STUDENT_IM_END:
        raise AssertionError(f"<|im_end|> id {im_end} != expected {STUDENT_IM_END}")
    eot = teacher_tok.convert_tokens_to_ids("<|eot_id|>")
    if eot != TEACHER_EOT_ID:
        raise AssertionError(f"<|eot_id|> id {eot} != expected {TEACHER_EOT_ID}")
    if verbose:
        print(f"[token_bridge] OK — shared base vocab ({SHARED_VOCAB_SIZE} ids), "
              f"swap map {{{STUDENT_IM_END}: {TEACHER_EOT_ID}}}")

"""HuggingFace ``transformers`` reference engine.

This is the trusted oracle for the L3/L4 differential tests: a plain, eager
``transformers`` forward pass with greedy decoding. vLLM's job is to produce the
*same* tokens far more efficiently; where it diverges, this reference is what
"correct" means.

``transformers`` is an optional dependency (the ``hf`` extra). The module imports
without it and :func:`transformers_available` reports whether it is present.
"""

from __future__ import annotations

import torch

from .types import GenResult

try:
    import transformers  # noqa: F401

    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False


def transformers_available() -> bool:
    """Whether the optional ``transformers`` dependency is importable.

    Returns
    -------
    bool
        ``True`` if ``transformers`` imported successfully at module load.
    """
    return _HAS_HF


_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def resolve_dtype(name: str) -> torch.dtype:
    """Map a short dtype name to its ``torch.dtype``.

    Parameters
    ----------
    name : {"fp32", "fp16", "bf16"}
        Short dtype name.

    Returns
    -------
    torch.dtype
        The corresponding torch dtype.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised dtype.
    """
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(_DTYPES)}")
    return _DTYPES[name]


class HFReference:
    """Greedy decoding and next-token log-probabilities from a HF model.

    Wraps a ``transformers`` causal-LM and its tokenizer to serve as the trusted
    reference engine in the differential harness.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier or local path passed to ``from_pretrained``.
    dtype : {"fp32", "fp16", "bf16"}, optional
        Parameter dtype to load the model in. Default ``"fp32"``.
    device : str, optional
        Torch device string the model and inputs are placed on. Default ``"cpu"``.
    seed : int, optional
        Seed for ``torch.manual_seed`` at construction. Default
        ``0x9E3779B9`` (the golden-ratio constant φ·2³², chosen for its
        well-distributed bit pattern rather than a low-entropy value like
        ``0``). Note that under greedy decoding over pretrained weights the
        seed does not affect outputs; it matters only if sampling is added.

    Attributes
    ----------
    model_id : str
        The model identifier the instance was built from.
    device : str
        Device the model runs on.
    torch_dtype : torch.dtype
        Resolved dtype the model was loaded in.
    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer paired with the model.
    model : transformers.PreTrainedModel
        The loaded causal-LM in eval mode.

    Raises
    ------
    RuntimeError
        If ``transformers`` is not installed.
    """

    def __init__(
        self,
        model_id: str,
        *,
        dtype: str = "fp32",
        device: str = "cpu",
        seed: int = 0x9E3779B9,
    ) -> None:
        if not _HAS_HF:
            raise RuntimeError("transformers is not installed (pip install '.[hf]')")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        self.model_id = model_id
        self.device = device
        self.torch_dtype = resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, dtype=self.torch_dtype).to(device).eval()
        )

    @torch.no_grad()
    def greedy_generate(self, prompts: list[str], *, max_new_tokens: int = 16) -> list[GenResult]:
        """Greedy-decode each prompt; return continuation ids, text, and step logprobs.

        Parameters
        ----------
        prompts : list[str]
            Prompts to decode independently.
        max_new_tokens : int, optional
            Maximum number of tokens to generate per prompt. Default ``16``.

        Returns
        -------
        list[GenResult]
            One result per prompt, in input order. ``step_logprobs`` holds the
            log-probability the model assigned to each emitted token.
        """
        results: list[GenResult] = []
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_len = enc["input_ids"].shape[1]
            out = self.model.generate(
                **enc,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            gen_ids = out.sequences[0, prompt_len:].tolist()
            # out.scores[i] are the next-token logits at generation step i.
            step_logprobs: list[float] = []
            for i, tok in enumerate(gen_ids):
                logits = out.scores[i][0].float()
                step_logprobs.append(float(torch.log_softmax(logits, dim=-1)[tok]))
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(GenResult(prompt, gen_ids, text, step_logprobs))
        return results

    @torch.no_grad()
    def next_token_logprobs(self, prompt: str) -> torch.Tensor:
        """Full ``(vocab,)`` log-probability distribution for the next token.

        Parameters
        ----------
        prompt : str
            Prompt to condition on.

        Returns
        -------
        torch.Tensor
            A 1-D float tensor of shape ``(vocab_size,)`` holding the
            ``log_softmax`` of the next-token logits.
        """
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits[0, -1, :].float()
        return torch.log_softmax(logits, dim=-1)

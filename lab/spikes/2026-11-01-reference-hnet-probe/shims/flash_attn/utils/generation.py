"""flash_attn.utils.generation.GenerationMixin: the reference LM inherits it only for `generate`, which this
harness does not shim (kv-cache decoding raises in flash_attn_with_kvcache)."""


class GenerationMixin:
    def generate(self, *args, **kwargs):
        raise NotImplementedError("generation is not shimmed; use full forward passes")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is derived from the Flux conditioner module.
# Original source: https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/conditioner.py
# Original license: Apache License 2.0

# pyre-unsafe

import torch
from torch import nn, Tensor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    def __init__(self, version: str, is_clip: bool, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = is_clip
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                version, max_length=max_length
            )
            # pyre-fixme[8]: Attribute has type `CLIPTextModel`; used as
            #  `PreTrainedModel`.
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                version, **hf_kwargs
            )
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                version, max_length=max_length
            )
            # pyre-fixme[8]: Attribute has type `T5EncoderModel`; used as
            #  `PreTrainedModel`.
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                version, **hf_kwargs
            )

        # pyre-fixme[16]: Item `CLIPTextModel` of `CLIPTextModel | T5EncoderModel`
        #  has no attribute `eval`.
        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        # pyre-fixme[29]: `Union[CLIPTokenizer, T5Tokenizer]` is not a function.
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        # pyre-fixme[29]: `Union[CLIPTextModel, T5EncoderModel]` is not a function.
        outputs = self.hf_module(
            # pyre-fixme[16]: Item `CLIPTextModel` of `CLIPTextModel |
            #  T5EncoderModel` has no attribute `device`.
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]


def load_t5(device, max_length=256):
    return HFEmbedder(
        "google/t5-v1_1-xl",
        is_clip=False,
        max_length=max_length,
        torch_dtype=torch.bfloat16,
    ).to(device)


def load_clip(device):
    return HFEmbedder(
        "openai/clip-vit-large-patch14",
        is_clip=True,
        max_length=77,
        torch_dtype=torch.bfloat16,
    ).to(device)


class TextFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.t5 = load_t5(device)
        self.clip = load_clip(device)

    def to(self, dtype):
        self.t5 = self.t5.to(dtype)
        self.clip = self.clip.to(dtype)
        return self

    def __call__(self, captions):
        t5_out = self.t5(captions).detach()
        clip_out = self.clip(captions).detach()
        return t5_out, clip_out


class DummyTextFeatureExtractor:
    def __init__(self, device):
        pass

    def __call__(self, captions):
        return None, None

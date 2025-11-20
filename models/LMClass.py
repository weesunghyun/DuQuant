import torch
import transformers
import torch.nn.functional as F
from torch import nn
from typing import Optional
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_config_with_rope_fix
from .models_utils import (
    BaseLM,
    find_layers,
    get_rolling_token_windows,
    make_disjoint_window,
)
import pdb


class LMClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model
        config = load_config_with_rope_fix(
            args.model, attn_implementation=args.attn_implementation
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
        # self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=config.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

        self.is_llama3 = self._is_llama3_tokenizer()
        self.llama3_system_prompt = getattr(
            self.tokenizer, "default_system_prompt", None
        )

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _is_llama3_tokenizer(self):
        model_name = self.model_name.lower()
        template = getattr(self.tokenizer, "chat_template", None) or ""
        if "llama-3" in model_name:
            return True
        return "llama-3" in template.lower()

    def _perplexity_prefix_token_id(self):
        if self.is_llama3 and self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.eot_token_id

    def _prepare_perplexity_tokens(self, text: str):
        if not self.is_llama3:
            return self.tok_encode(text)

        chat_template = getattr(self.tokenizer, "chat_template", None)

        if chat_template:
            system_prompt = []
            if self.llama3_system_prompt:
                system_prompt.append(
                    {"role": "system", "content": self.llama3_system_prompt}
                )
            messages = system_prompt + [{"role": "user", "content": text}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )

        tokens = []
        if self.tokenizer.bos_token_id is not None:
            tokens.append(self.tokenizer.bos_token_id)
        if self.llama3_system_prompt:
            tokens.extend(
                self.tokenizer.encode(
                    self.llama3_system_prompt, add_special_tokens=False
                )
            )
        tokens.extend(self.tok_encode(text))
        return tokens

    def loglikelihood_rolling(self, requests, disable_tqdm: Optional[bool] = None):
        if not self.is_llama3:
            return super().loglikelihood_rolling(
                requests, disable_tqdm=disable_tqdm
            )

        loglikelihoods = []
        outer_disable_tqdm = False if disable_tqdm is None else disable_tqdm
        inner_disable_tqdm = True if disable_tqdm is None else disable_tqdm

        for (string,) in tqdm(requests, disable=outer_disable_tqdm):
            tokenized = self._prepare_perplexity_tokens(string)
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=tokenized,
                        prefix_token=self._perplexity_prefix_token_id(),
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows, disable_tqdm=inner_disable_tqdm
            )

            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )

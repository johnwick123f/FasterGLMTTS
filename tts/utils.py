import argparse
import json
import logging
import os
import torch
import torchaudio
import tqdm

from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
from utils import file_utils, seed_util
from utils import tts_model_util, yaml_util
from transformers import AutoTokenizer, LlamaForCausalLM
from llm.glmtts import GLMTTS
from utils.audio import mel_spectrogram
from functools import partial
from utils.vocos_util import Vocos2DInference

def load_frontends(speech_tokenizer, sample_rate=24000, use_phoneme=False, frontend_dir="frontend", model_dir='ckpt'):
    
    feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=480, n_fft=1920, num_mels=80, win_size=1920, fmin=0, fmax=8000, center=False)

    glm_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, 'vq32k-phoneme-tokenizer'), trust_remote_code=True
    )

    tokenize_fn = lambda text: glm_tokenizer.encode(text)

    frontend = TTSFrontEnd(
        tokenize_fn,
        speech_tokenizer,
        feat_extractor,
        os.path.join(frontend_dir, "campplus.onnx"),
        os.path.join(frontend_dir, "spk2info.pt"),
        DEVICE,
    )
    text_frontend = TextFrontEnd(use_phoneme)
    return frontend, text_frontend

def local_flow_forward(flow, token_list, prompt_speech_tokens, speech_feat, embedding, nsteps):
    """
    Single Flow forward pass.
    """
    wav, full_mel = flow.token2wav_with_cache(
        token_list,
        n_timesteps=nsteps,
        prompt_token=prompt_speech_tokens,
        prompt_feat=speech_feat,
        embedding=embedding,
    )
    return wav.detach().cpu(), full_mel

def get_cached_prompt(cache, synth_text_token, device):
    """
    Constructs prompt tokens from the cache.
    Prunes the cache if the sequence length exceeds MAX_LLM_SEQ_INP_LEN.
    """
    cache_text = cache["cache_text"]
    cache_text_token = cache["cache_text_token"]
    cache_speech_token = cache["cache_speech_token"]

    def __len_cache_text_token():
        return sum(map(lambda x: x.shape[1], cache_text_token))

    def __len_cache_speech_token():
        return sum(map(len, cache_speech_token))

    # Estimate required length ratio
    # Avoid division by zero
    text_len = __len_cache_text_token()
    ta_ratio = __len_cache_speech_token() / (text_len if text_len > 0 else 1.0)

    __len_synth_text_token = synth_text_token.shape[1]
    __len_synth_audi_token_estim = int(ta_ratio * __len_synth_text_token)

    # Prune cache if too long.
    # Logic: Keep the first item (original prompt), remove from the second item onwards.
    while (
        __len_cache_speech_token() + __len_synth_audi_token_estim > MAX_LLM_SEQ_INP_LEN
    ):
        if len(cache_speech_token) <= 1:
            break  # Always keep at least the original prompt
        # logging.debug(f'[get_cached_prompt] Cache pop. Text count before: {len(cache_text)}')
        cache_text.pop(1)
        cache_text_token.pop(1)
        cache_speech_token.pop(1)

    # Construct Text Prompt
    prompt_text_token_from_cache = []
    for a_token in cache_text_token:
        prompt_text_token_from_cache.extend(a_token.squeeze().tolist())

    prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(device)

    # Construct Speech Prompt
    speech_tokens = []
    for a_cache_speech_token in cache_speech_token:
        speech_tokens.extend(a_cache_speech_token)

    llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(device)

    return prompt_text_token, llm_speech_token

def process_inputs(frontend, text_frontend, tokenizer, text_info, cache, device, embedding, seed=0, flow_prompt_token=None, speech_feat=None, use_phoneme=False):
    
    ats_token = 61498 ## token to convert inputs to real llm tokens
    boa_token = 59260 ## beginning of audio token
    outputs = []
    full_mels = []
    output_token_list = []
    all_input_texts = []
    uttid = text_info[0]
    syn_text = text_info[1]
    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }
    short_text_list = text_frontend.split_by_len(syn_text)
    
    for _, tts_text in enumerate(short_text_list):
        seed_util.set_seed(seed)
        tts_text_tn = text_frontend.text_normalize(tts_text)
        text_tn_dict["syn_text_tn"].append(tts_text_tn)
        
        if use_phoneme:
            tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
            text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
            
        tts_text_token = frontend._extract_text_token(tts_text_tn)

        # Access cache references
        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]

        # Determine Prompts
        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(
                cache, tts_text_token, device
            )
        else:
            # Initial prompt case
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor(
                [cache_speech_token[0]], dtype=torch.int32
            ).to(device)
            logging.debug("[generate_long] Using initial prompt (empty cache history)")
        
        boa_tensor = torch.tensor([boa_token], device=device).unsqueeze(0)
        fake_prompt_speech_token = prompt_speech_token + ats_token
        input_full = torch.cat([
                prompt_text_token, 
                tts_text_token, 
                boa_tensor, 
                fake_prompt_speech_token
            ], dim=1).to(torch.long)
        input_text = tokenizer.decode(input_full[0])
        all_input_texts.append(input_text)
        return all_input_texts

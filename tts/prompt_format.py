import torch

def preprocess_audio(prompt_text, prompt_audio_path, text_frontend, frontend, use_cache=True, use_phoneme=False):
    
    prompt_text = text_frontend.text_normalize(prompt_text)
    prompt_text_token = frontend._extract_text_token(prompt_text + " ")
    prompt_speech_token = frontend._extract_speech_token([prompt_audio_path])
    
    speech_feat = frontend._extract_speech_feat(prompt_audio_path, sample_rate=24000)
    embedding = frontend._extract_spk_embedding(prompt_audio_path)
    
    cache_speech_token = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token = torch.tensor(
        cache_speech_token, dtype=torch.int32
    ).to(DEVICE)

    cache = {
        "cache_text": [prompt_text],
        "cache_text_token": [prompt_text_token],
        "cache_speech_token": cache_speech_token,
        "use_cache": use_cache,
    }
    return cache, flow_prompt_token, speech_feat, embedding

def process_text(input_text):
    synth_text = text_frontend.text_normalize(input_text)
    out = [synth_text, synth_text]
    return out

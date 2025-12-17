import torch
import gc
from tts.utils import *
from tts.prompt_format import *
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

class GLMTTS:
    """
    High level class for GLM-TTS model
    """

    def __init__(self, decoder_dir="zai-org/GLM-TTS", llm_dir="YatharthS/GLM-TTS-Llama"):
        """
        Loads the flow matching decoder, vocos model, llm model, and frontends
        """
        # Instance Attributes (Unique to each object)
        model_path = snapshot_download(decoder_dir)
        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        frontend, text_frontend, speech_tokenizer, flow = load_models(
            use_phoneme=False,
            sample_rate=24000,
            model_dir=model_path,
            DEVICE=self.DEVICE
        )
        backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)
        self.frontend = frontend
        self.text_frontend = text_frontend
        self.speech_tokenizer = speech_tokenizer
        self.flow = flow
        self.pipe = pipeline(llm_dir, backend_config=backend_config)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)

    def encode_audio(self, prompt_text, prompt_audio):
        """
        Encodes audio into reference content
        """
        reference_content = {}
        cache, flow_prompt_token, speech_feat, embedding = preprocess_audio(prompt_text, prompt_audio, self.text_frontend, self.frontend)
        reference_content = {"cache": cache, "flow_prompt_token": flow_prompt_token, "speech_feat": speech_feat, "embedding": embedding}
        return reference_content

    def generate(self, text, reference_content, top_k=200, temperature=1.0, repetition_penalty=2.0, nsteps=2):
        """
        Generates speech from text
        """
        text_info = process_text(text)
        input_texts = process_inputs(self.frontend, self.text_frontend, self.tokenizer, text_info, reference_content['cache'], self.DEVICE, reference_content['embedding'], seed=0, flow_prompt_token=None, speech_feat=None, use_phoneme=False)

        gen_config = GenerationConfig(top_p=1.0,
                              top_k=top_k,
                              temperature=temperature,
                              max_new_tokens=200,
                              repetition_penalty=repetition_penalty,
                              min_p=0.01,
                              do_sample=True,
                              stop_token_ids=[59253],
                              min_new_tokens=100)
        responses = self.pipe(input_texts, gen_config=gen_config, do_preprocess=False)
        for response in responses:
            token_ids = torch.tensor([response.token_ids]).to(self.DEVICE) - 61498
            output, full_mel = local_flow_forward(flow=self.flow, token_list=token_ids, prompt_speech_tokens=reference_content['flow_prompt_token'], speech_feat=reference_content['speech_feat'], embedding=reference_content['embedding'], nsteps=nsteps)
    
        return output

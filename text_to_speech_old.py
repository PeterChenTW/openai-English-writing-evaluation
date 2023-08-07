# from transformers import BarkModel
# import torch
# from transformers import AutoProcessor
# import scipy

# model = BarkModel.from_pretrained("suno/bark-small")
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

# voice_preset = "v2/en_speaker_0"
# processor = AutoProcessor.from_pretrained("suno/bark-small")
# # prepare the inputs
# text_prompt = "Let's try generating speech, with Bark, a text-to-speech model"
# inputs = processor(text_prompt, voice_preset=voice_preset)

# # generate speech
# speech_output = model.generate(**inputs.to(device))

# sampling_rate = model.generation_config.sample_rate
# scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_output[0].cpu().numpy())

# from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
# from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
# import soundfile as sf
# import re

# models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
#     "facebook/fastspeech2-en-ljspeech",
#     arg_overrides={"vocoder": "hifigan", "fp16": False, "cpu": True}
# )

# model = models[0]
# TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
# generator = task.build_generator([model], cfg)

# text = """Hello World. This is an example of text to speech using pyttsx3."""

# text = re.sub(r'[.\?!:;\[\]\{\}"\“”\-_—–...()]', ',', text) 

# text = re.sub(r'[-_—–]', ' ', text)  

# text = re.sub(r'[...]', ' ', text)

# sample = TTSHubInterface.get_model_input(task, text)
# wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

# # 保存为文件
# sf.write('output.wav', wav, rate)

# from transformers import AutoProcessor, AutoModel
# import scipy

# processor = AutoProcessor.from_pretrained("suno/bark")
# model = AutoModel.from_pretrained("suno/bark")

# # https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
# voice_preset = "v2/en_speaker_0"
# inputs = processor(
#     text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
#     return_tensors="pt",
#     voice_preset=voice_preset
# )

# speech_values = model.generate(**inputs, do_sample=True)

# sampling_rate = 24000 # Bark default is 24kHz
# # sampling_rate = model.config.sample_rate
# scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())


# from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
# from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
# import IPython.display as ipd


# models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
#     "facebook/fastspeech2-en-ljspeech",
#     arg_overrides={"vocoder": "hifigan", "fp16": False}
# )
# model = models[0]
# TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
# generator = task.build_generator(model, cfg)

# text = "Hello, this is a test run."

# sample = TTSHubInterface.get_model_input(task, text)
# wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

# ipd.Audio(wav, rate=rate)



# # Following pip packages need to be installed:
# # !pip install git+https://github.com/huggingface/transformers sentencepiece datasets

# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
# from datasets import load_dataset
# import torch
# import soundfile as sf


# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# def generate_voice(text):
#     inputs = processor(text=text, return_tensors="pt")

#     # load xvector containing speaker's voice characteristics from a dataset
#     embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split='validation')

#     # for i in [file_name for file_name in embeddings_dataset['filename'] if 'rms' in file_name]:
#     #     print(i)
#     selected_row = [i for i, file_name in enumerate(embeddings_dataset['filename']) if file_name == 'cmu_us_rms_arctic-wav-arctic_b0537'][0]
#     print(selected_row)
#     speaker_embeddings = torch.tensor(embeddings_dataset[selected_row]["xvector"]).unsqueeze(0)


#     speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

#     sf.write("speech_1.wav", speech.numpy(), samplerate=16000)

# # while True:
# generate_voice('good! nice to meet you!')

# import pyttsx3

# engine = pyttsx3.init() # 初始化语音引擎

# text = "Hello World. This is an example of text to speech using pyttsx3."

# engine.say(text) # 要转换的文本

# engine.runAndWait() # 执行语音转换

import requests
import soundfile as sf
API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": """I see. So you needed to build up your leg strength and endurance in preparation for the climb. Can you tell me more about the equipment and gear that you needed to bring with you for the climb?""",
})
print(output)
# sf.write('output.wav', output[0], output[1])
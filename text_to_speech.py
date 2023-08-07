from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import soundfile as sf
import re

models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False, "cpu": True}
)

model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator([model], cfg)


def facebook_speech_model(text, save_path):
	try:
		text = re.sub(r'[.\?!:;\[\]\{\}"\“”\-_—–...()]', ',', text) 

		text = re.sub(r'[-_—–]', ' ', text)  

		text = re.sub(r'[...]', ' ', text)

		sample = TTSHubInterface.get_model_input(task, text)
		wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

		# 保存为文件
		sf.write(save_path, wav, rate)

		return True
	except Exception as e:
		print(e)
		return False


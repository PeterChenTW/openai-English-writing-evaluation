import pyaudio
import wave
import os
import openai
import keyboard
import google.cloud.texttospeech as tts
import os
import datetime
import pytz
import argparse
import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/csro7/english-379512-3af78378c095.json"

class SpeakingTest:
    def __init__(self, role, debug_mode=False):
        self.current_absolute_path = os.path.abspath('.')
        self.role = role
        self.debug_mode = debug_mode
        self.folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{role}"
        self.folder_path = os.path.join(self.current_absolute_path, self.folder_name)
        os.mkdir(self.folder_name)

    def start(self):
        teacher = ChatBot(self.role)

        while True:
            self._print('='*30)
            human_text = self._speech_to_text(f"{self.folder_path}/{datetime.datetime.now().strftime('%H-%M-%S')}_huamn")
            self._print(f' Me: {human_text}')

            bot_text = teacher.listen_and_speak(human_text)
            self._text_to_speech(bot_text, f"{self.folder_path}/{datetime.datetime.now().strftime('%H-%M-%S')}_bot")
            self._print(f' {self.role}: {bot_text}')
            self._save_file(f"Me: {human_text}\n")
            self._save_file(f"{self.role}: {bot_text}\n")

    def _speech_to_text(self, WAVE_OUTPUT_FILENAME='recorded_audio'):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        WAVE_OUTPUT_FILENAME += '.wav'
        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        frames = []

        print('='*100)
        print('please press a\r', end='')
        keyboard.wait('a')
        print('Start recording, press d to stop.\r', end='')
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            if keyboard.is_pressed('d'):
                break
        print('End recording\r', end='')
        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        audio_file= open(WAVE_OUTPUT_FILENAME, 'rb')
        transcript = openai.Audio.transcribe(
            model='whisper-1', 
            file=audio_file,
            temperature=0.3,
            prompt="Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. I love play ball.",
            language='en'
        )

        return transcript['text']
    
    def _text_to_speech(self, text, filename='recorded_audio.wav', voice_name='en-AU-Neural2-B'):
        language_code = '-'.join(voice_name.split('-')[:2])
        text_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(
            language_code=language_code, name=voice_name
        )
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

        client = tts.TextToSpeechClient()
        response = client.synthesize_speech(
            input=text_input, voice=voice_params, audio_config=audio_config
        )
        filename += '.wav'
        with open(filename, 'wb') as out:
            out.write(response.audio_content)
        
        self._speak_up(filename)
        return filename
    
    def _speak_up(self, file_name):
        with wave.open(file_name, 'rb') as wav_file:
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=audio.get_format_from_width(sample_width),
                channels=channels,
                rate=rate,
                output=True
            )
            
            data = wav_file.readframes(frames)
            stream.write(data)
            
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def _save_file(self, content, file_name='chat_history.txt'):
        file_path = os.path.join(self.folder_path, file_name)
        with open(file_path, 'w') as report:
            report.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {content}")
    
    def _print(self, contenct):
        if self.debug_mode:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {contenct}")


class ChatBot:
    def __init__(self, role):
        self.current_absolute_path = os.path.abspath('.')
        self._load_prompt_dict()
        self.history = [
            {
                "role": "system", 
                "content": self.prompt_dict[role]
            }
        ]

    def listen_and_speak(self, content):
        self.history.append(            
            {
                "role": "user", 
                "content": content
            }
        )
        while True:
            try:
                reply = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=self.history,
                )
                return reply['choices'][0]['message'].get('content', '')
            except openai.error.RateLimitError as e:
                print('Reached open ai api limit. sleep for 60 seconds')
                time.sleep(60)
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                break


    def _load_prompt_dict(self):
        with open(os.path.join(self.current_absolute_path, 'prompts.json'), 'r') as json_file:
            self.prompt_dict = json.load(json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openai_api_key",
        dest="openai_api_key",
        type=str,
        help="openai api key",
    )
    parser.add_argument(
        "--debug_mode",
        dest="debug_mode",
        type=bool,
        default=False,
        help="debug_mode",
    )
    options = parser.parse_args()
    openai.api_key = options.openai_api_key
    debug_mode = options.debug_mode

    test = SpeakingTest('IETES Speaking Examiner', debug_mode)
    test.start()
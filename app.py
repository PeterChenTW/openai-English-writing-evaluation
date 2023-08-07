from flask import Flask, render_template, request, jsonify, url_for, redirect
import os
from datetime import datetime
import pyaudio
import wave
import csv
import glob
import requests
import time
import pandas as pd
import numpy as np
from text_to_speech import facebook_speech_model as text_2_speech
from speech_to_text import openai_model_audio_to_text as speech_2_text
from bot_speaker import ChatBot
from flask_sqlalchemy import SQLAlchemy
import uuid

app = Flask(__name__)
bot_role = "IETES Speaking Examiner"
current_time = timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

chatbot = None

db = SQLAlchemy(app)
class Student(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=str(uuid.uuid4()))
    title = db.Column(db.String(200), nullable=False)
    response = db.Column(db.Text, nullable=False)
    grade = db.Column(db.String(5), default='B+')
    advice = db.Column(db.String(500), default='Keep up the good work!')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/speaking')
def speaking():
    global chatbot
    chatbot = ChatBot(bot_role)
    return render_template('speaking.html')


@app.route('/change_role', methods=['POST'])
def change_role():
    global bot_role
    global chatbot
    bot_role = request.form['role']
    chatbot = ChatBot(bot_role)
    return ('', 204)


@app.route('/ask', methods=['POST'])
def ask():
    chat_id = request.form['chat_id']
    currentMessageCount = request.form['currentMessageCount']
    user_message = request.form['user_message']
    save_path = f'static/{current_time}_{chat_id}/{currentMessageCount}_{chat_id}_bot.wav'
    bot_message = bot_logic_based_on_role(user_message, save_path)

    append_to_chat_log(chat_id, user_message, bot_message, currentMessageCount)

    return jsonify({'bot_reply': bot_message})


def append_to_chat_log(chat_id, user_message, bot_message, index):
    folder_path = f'static/{current_time}_{chat_id}'
    os.makedirs(folder_path, exist_ok=True)
    csv_filename = os.path.join(folder_path, 'chat_log.csv')

    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'role', 'message', 'recording_path'])

    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        user_recording_path = f"{current_time}_{chat_id}/{index}_{chat_id}_user.wav"
        writer.writerow([index, 'user', user_message.replace('\n', ''), user_recording_path])

    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        bot_recording_path = f"{current_time}_{chat_id}/{index}_{chat_id}_bot.wav"
        writer.writerow([index, 'bot', bot_message.replace('\n', ''), bot_recording_path])


def improve_message(message, role):
    improver = ChatBot('English_imporver')
    if role == 'user':
        response_text = improver.listen_and_speak(message)
    else:
        response_text = '-'
    return response_text


def evaluate_audio_level(filename):
    # https://huggingface.co/hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation
    # output example :
    # [
    #     {'score': 0.6629263758659363, 'label': 'beginer'}, 
    #     {'score': 0.218276709318161, 'label': 'intermediate'}, 
    #     {'score': 0.08231456577777863, 'label': 'proficient'}, 
    #     {'score': 0.036482375115156174, 'label': 'advanced'}
    # ]
    API_URL = "https://api-inference.huggingface.co/models/hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation"
    API_TOKEN = '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'
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    # if 'bot' in filename:
    #     return 0
    with open(filename, "rb") as f:
        data = f.read()

    while True:
        try:
            response = requests.post(API_URL, headers=headers, data=data).json()
            print(response)
            break
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(5)

    ielts_mapping = {
        'beginer': (1, 3),
        'intermediate': (4, 5),
        'proficient': (6, 7),
        'advanced': (8, 9)
    }

    weighted_scores = 0
    total_weights = 0

    for score_data in response:
        label = score_data['label']
        score = score_data['score']
        lower, upper = ielts_mapping[label]
        weighted_score = score * (upper + lower) / 2
        weighted_scores += weighted_score
        total_weights += score

    return round(weighted_scores / total_weights, 2)

def bot_logic_based_on_role(user_message, save_path):
    response_text = chatbot.listen_and_speak(user_message)

    if text_2_speech(response_text, save_path):
        speaker(save_path)
    return response_text


is_recording = False


@app.route('/start_record', methods=['POST'])
def start_record():
    global is_recording
    is_recording = True
    return jsonify(success=True)


@app.route('/stop_record', methods=['POST'])
def stop_record():
    global is_recording
    is_recording = False
    return jsonify(success=True)


@app.route('/record', methods=['POST'])
def record():
    try:
        global is_recording
        chat_id = request.json['chat_id']
        currentMessageCount = request.json['currentMessageCount']
        currentMessageCount += 1
        folder_path = f'static/{current_time}_{chat_id}'
        os.makedirs(folder_path, exist_ok=True)
        recording_file_name = f'{currentMessageCount}_{chat_id}_user.wav'
        recording_file_path = os.path.join(folder_path, recording_file_name)

        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

        frames = []
        volume_factor = 1.5  # 設定音量因子，1.0 是正常音量，可以更改此值

        while is_recording:
            data = stream.read(1024)
            np_data = np.frombuffer(data, dtype=np.int16)
            np_data = (np_data * volume_factor).clip(min=-32768, max=32767).astype(np.int16)
            frames.append(np_data.tobytes())

        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(recording_file_path, 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        return jsonify(success=True, message=speech_2_text(recording_file_path))
    except Exception as e:
        print(e)
        return jsonify(success=False, message=str(e))

def speaker(filename):
    chunk = 1024
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()	


@app.route('/chat_history/', defaults={'chat_id': None})
@app.route('/chat_history/<chat_id>')
def chat_history(chat_id):
    all_hs = get_all_chat_ids()
    if chat_id is None:
        chat_id = all_hs[0]

    chat_log = load_chat_log(chat_id)
    return render_template('chat_history.html', chat_history=chat_log, chat_history_data=all_hs)


def get_all_chat_ids():
    chat_folders = glob.glob('static/*')
    chat_ids = [folder.split('\\')[-1] for folder in chat_folders]
    return sorted(chat_ids, reverse=True)


def load_chat_log(chat_id):
    folder_path = os.path.join('static', f'{chat_id}')
    csv_filename = os.path.join(folder_path, 'chat_log.csv')

    retries = 10  # 你可以設置重試的次數

    while retries > 0:
        try:
            chat_log = pd.read_csv(csv_filename)
            
            if 'score' not in chat_log.columns:
                print('go')
                chat_log['score'] = chat_log.apply(
                    lambda x: evaluate_audio_level(os.path.join('static', x.recording_path)), 
                    axis=1
                )
                chat_log.to_csv(csv_filename, index=False)
            if 'improve' not in chat_log.columns:
                print('go improve')
                def combine_messages(row):
                    if row['role'] == 'user' and row['index'] > 0:
                        bot_index = row['index'] - 1
                        bot_message = chat_log[(chat_log['index'] == bot_index) & (chat_log['role'] == 'bot')]['message'].values
                        if bot_message.size > 0:
                            return f"Examiner Question: '{bot_message[0]}' \n My reply: '{row['message']}'"
                    return f"{row['role']}: {row['message']}"
                chat_log['Q_A'] = chat_log.apply(combine_messages, axis=1)
                chat_log['improve'] = chat_log.apply(
                    lambda x: improve_message(x.Q_A, x.role), 
                    axis=1
                )
                chat_log.to_csv(csv_filename, index=False)
            chat_log = chat_log.to_dict(orient='records')
            return chat_log

        except TypeError:
            # 這裡處理TypeError，你可以添加任何必要的邏輯
            print("TypeError encountered, retrying...")
            time.sleep(5)
            retries -= 1

        except Exception as e:
            print(f"An error occurred while reading the chat log: {e}")
            return []
    
    # 如果超過重試次數，可以返回一個空列表或其他適當的回應
    print("Reached maximum number of retries, returning empty list.")
    return []

@app.route('/writing')
def writing():
    students = Student.query.all()
    return render_template('writing.html', students=students)

@app.route('/submit', methods=['POST'])
def submit():
    title = request.form['title']
    response = request.form['response']

    student = Student(title=title, response=response)
    db.session.add(student)
    db.session.commit()

    return redirect(url_for('writing_result', student_id=student.id))

@app.route('/writing_result/<student_id>')
def writing_result(student_id):
    student = Student.query.get(student_id)
    if student:
        return render_template('writing_result.html', title=student.title, response=student.response, grade=student.grade, advice=student.advice)
    else:
        return "Record not found", 404

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

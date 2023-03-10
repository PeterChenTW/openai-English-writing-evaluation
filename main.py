import argparse
import openai
import os
import datetime
import json
import time

class GenerateReport:
    def __init__(self, topic, article, bug_mode=False):
        self.current_absolute_path = os.path.abspath('.')
        self.topic = topic
        self.article = article
        self.bug_mode = bug_mode

    def _teacher(self):
        teacher = ChatBot('writing_teacher')
        Q_A = {
            'Essay Topic': self.topic,
            'Student Essay': self.article
        }

        r = teacher.listen_and_speak(
            f"""
            [
                'Essay Topic': [{self.topic}], 
                'Student Essay': [{self.article}]
            ], please give me the band score for task achievement, coherence and cohesion, lexical resource and grammatical range and accuracy.(just reply number, no reason needed)!
            """ 
        )

        Q_A['Band Score'] = r 
        self._print(f"writing_teacher: {r}")

        questions = {
            'Suggestions': "list suggestions(just reply items)",
            'Corrections': "list corrections(just reply items)",
            'Example': 'write a 8-point essay for the "same topic", and list down the three best sentences that I need to learn or understand from the example.',
            'Words': 'provide useful English words for this topic along with their meaning and example sentence.',
        }

        for k, q in questions.items():
            self._print(f"Me: {q}")
            r = teacher.listen_and_speak(q)
            Q_A[k] = r
            self._print(f"riting_teacher: {r}")

        return Q_A
           
    def run(self):
        result = self._teacher()

        new_s = 'IETLS Writing Test Report\n\n'
        for k, v in result.items():
            new_s += f'{k}\n{"-"*30}\n{v}\n' + '='*30 + '\n'
        print('Generating results')
        self._design(new_s)
        
    def _design(self, text):
        designer = ChatBot('designer')

        r = designer.listen_and_speak(
            f"""Create an HTML page design for the following text that is easy for professional users to read in a report format. Provide the HTML code only."
                        "{text}"""""
        )

        self._save_file(
            content=r,
            file_name=f"{datetime.datetime.now().strftime('%H-%M-%S')}_{self.topic[:10]}_report.html".replace(' ', '')
        )
    
    def _save_file(self, content, file_name):
        folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d')}_Writing"
        folder_path = os.path.join(self.current_absolute_path, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w') as report:
            report.write(content)

        print(file_path)
    
    def _print(self, contenct):
        if self.bug_mode:
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
        "--bug_mode",
        dest="bug_mode",
        type=bool,
        default=False,
        help="bug_mode",
    )
    options = parser.parse_args()
    openai.api_key = options.openai_api_key
    bug_mode = options.bug_mode

    topic = input('Essay Topic: ')
    article = input('Your article: ')
    generator = GenerateReport(topic, article, bug_mode)
    generator.run()
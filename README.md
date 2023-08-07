# Automated English Writing Evaluation using Openai API/Huggingface

This is a Python program that automatically evaluates English writing proficiency using the Openai ChatGPT API. You can input topics and articles, and the program will generate an automated evaluation report based on the given topic and article, including suggestions, corrections, examples, and more.

## Workflow
```mermaid
graph TD
A[User launches project]
B[Selects bot role]
C[Clicks to record voice]
D[Backend converts speech to text] 
E[Passes text to LLM model]
F[LLM generates reply text]
G[Text to speech conversion]
H[Play voice reply]
I[Loop back for interaction]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G 
G --> H
H --> I
I --> C
```

## Run the program

```
python app.py
```


## Screenshot

![Home](screenshot/home.bmp)

![speaking_1](screenshot/speaking_1.bmp)

![speaking_result](screenshot/speaking_result.bmp)

![writing_1](screenshot/writing_1.bmp)

![writing_result](screenshot/writing_result.bmp)

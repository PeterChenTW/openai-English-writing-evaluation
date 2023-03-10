# Automated English Writing Evaluation using Openai API

This is a Python program that automatically evaluates English writing proficiency using the Openai ChatGPT API. You can input topics and articles, and the program will generate an automated evaluation report based on the given topic and article, including suggestions, corrections, examples, and more.

## How to use

1. Install the required Python libraries:

```
pip install openai
```

2. Obtain an Openai API key.

3. Download and run the program:

```
python main.py --openai_api_key <your_api_key>
```

4. Follow the program prompts to input the topic and article.

5. The program will automatically generate an HTML report showing the evaluation results.

## Special options

You can use `--bug_mode True` to enable the `bug mode`, which will display more program information for debugging purposes.

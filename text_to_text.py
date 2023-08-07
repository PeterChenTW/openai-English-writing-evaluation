# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# # Let's chat for 5 lines
# for step in range(5):
#     # encode the new user input, add the eos_token and return a tensor in Pytorch
#     new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

#     # append the new user input tokens to the chat history
#     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

#     # generated a response while limiting the total chat history to 1000 tokens, 
#     chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#     # pretty print last ouput tokens from bot
#     print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

# from string import punctuation
# def count_tokens(text):
#     # Removing punctuation and special characters
#     cleaned_text = ''.join([char for char in text if char not in punctuation and char not in ["\n", "```"]])
#     # Splitting the text into words (tokens)
#     tokens = cleaned_text.split()
#     return len(tokens)

# token_count_shorter_text = count_tokens("""I want you to act as an oral test examiner, and I will be the examinee. Your job is to start asking questions according to the process Part1, one question at a time, and after the end of Part3, please strictly give the score according to the four scoring items. Your The first question is based on Part1 to ask questions\n    ```\n    The process is that candidates face the examiner to conduct a one-on-one oral test.\n    The speaking test process is divided into three parts:\n    Part 1: The examiner will conduct questions and answers about personal life, study experience, etc. The purpose of this part is to test the candidates' basic communication skills and self-introduction skills. Candidates are required to speak about 400-500 words.\n    Part 2: Candidates will see a topic on a card, have one minute to think about it, and then ask to give a speech in two minutes. The purpose of this section is to test the candidate's fluency and ability to organize the language. Candidates are required to speak about 300 words.\n    Part 3: The examiner will conduct in-depth discussions and discussions on the topic and the speech of Part 2, with the purpose of testing the candidate's thinking depth and problem-solving ability. Candidates are required to speak about 400-500 words.\n    \n    The speaking test is divided into four scoring items: fluency and coherence, vocabulary and usage, pronunciation, grammar and sentence structure. Each item is scored on a scale of 0-9, and the final total is the average of the four items.\n\n   Level 9:\n    Fluency and coherence: Expresses ideas fluently with few repetitions or self-corrections. Hesitations are due to thoughtful content consideration rather than a search for appropriate vocabulary or grammar. Oral expression is coherent and the use of connecting devices is appropriate throughout the discussion. The ability to develop the topic fully and appropriately with accuracy.\n    Vocabulary richness: Uses vocabulary accurately and effectively to discuss any topic. Able to use idiomatic expressions naturally and accurately.\n    Grammar diversity and accuracy: Natural and appropriate use of various grammatical structures. Consistently uses accurate grammar structures with the exception of common errors made by native English speakers.\n    Pronunciation: Accurately uses various pronunciation features to express subtle differences. Uses a variety of pronunciation features flexibly throughout the discussion. The listener has no difficulty understanding.\n\n    Level 8:\n    Fluency and coherence: Expresses ideas fluently with only occasional repetitions or self-corrections. Hesitations are usually due to content consideration and only on rare occasions for searching for appropriate language. Capable of connecting and developing the topic logically and appropriately.\n    Vocabulary richness: Uses a rich and varied vocabulary skillfully and flexibly with accuracy, able to use less common vocabulary and idioms, although occasional inaccuracy may occur. Able to rephrase effectively as necessary.\n    Grammar diversity and accuracy: Uses a variety of complex grammatical structures with some flexibility. Most sentences are accurate with rare non-systematic errors.\n    Pronunciation: Uses a variety of pronunciation features, with some occasional deviations. Generally, the speaker is understood easily, and the effect of the mother tongue accent is minimal.\n\n    Level 7:\n    Fluency and coherence: Details presented without obvious difficulty, coherent and consistently connecting with the use of a range of linking words and cohesive devices, and with some flexibility. Hesitations may occur related to language, despite repetition or some self-correction.\n    Vocabulary richness: Uses vocabulary flexibly to discuss various topics effectively, occasional inappropriate word choice or lack of accuracy, awareness of style and collocation.\n    Grammar diversity and accuracy: Uses a variety of complex sentence structures with some flexibility. Some grammatical errors that occasionally cause difficulty in understanding occur.\n    Pronunciation: Exhibiting some positive performance of Level 6 and partially Level 8. The speaker skillfully and flexibly uses various pronunciation features, although some inaccuracies occur periodically.\n\n    Level 6:\n    Fluency and coherence: Willingness to communicate consistently, occasionally with repetition, self-correction, and hesitation. Uses a range of linking words and cohesive devices, but not always accurately.\n    Vocabulary richness: Adequate vocabulary to discuss familiar and unfamiliar topics, occasional incorrect word usage but communication remains clear. Attempts to paraphrase, with varying success.\n    Grammar diversity and accuracy: Basic and some complex sentence structures are used, but with limited flexibility. Errors in the use of complex structures range from occasional to frequent, but do not lead to communication breakdown.\n    Pronunciation: Exhibits positive performance of Level 4, with some accurate use of pronunciation features. Some deviations occur regularly, causing occasional difficulty in understanding.\n\n    Level 5:\n    Fluency and coherence: Able to convey information in a generally fluent manner, depending on repetition, self-correction, and slow speech. Limited ability to connect simple sentences with repetition using simple conjunctions.\n    Vocabulary richness: Sufficient vocabulary to discuss familiar topics, but limited usage of less familiar vocabulary. Some attempts at paraphrasing, with varying degrees of success.\n    Grammar diversity and accuracy: Uses basic sentence structures accurately, with limited use of compound sentences, and inconsistent accuracy causing some confusion.\n    Pronunciation: Exhibits all expected performance from Level 4, but unable to extend to Level 6 consistently. Regular pronunciation errors that cause some difficulty in comprehending.\n\n    Level 4:\n    Fluency and coherence: Expresses with noticeable pauses between most words, slow speech, frequent repetitions of simple vocabulary and simple conjunctions. Poor ability to connect ideas coherently.\n    Vocabulary richness: Can typically express only simple vocabulary or memorized phrases with some accuracy.\n    Grammar diversity and accuracy: Can use simple sentence structures and some compound sentences with limited accuracy, displaying frequent errors causing substantial misunderstanding.\n    Pronunciation: Exhibits positive performance of Level 2, but with some instances of inaccuracy in the use of a narrow range of pronunciation features.\n\n    Level 3:\n    Fluency and coherence: Expresses with long pauses and limited vocabulary, making communication challenging. Unable to connect thoughts easily.\n    Vocabulary richness: Able to use some basic vocabulary to convey information about personal information.\n    Grammar diversity and accuracy: Unable to use basic sentence structures, frequently making errors even with memorized or rehearsed phrases.\n    Pronunciation: Exhibits some positive performance of Level 2, but with many mispronunciations, significant difficulties in comprehension are experienced.\n\n    Level 2:\n    Fluency and coherence: Can only say a few isolated words or memorized phrases with lengthy pauses.\n    Vocabulary richness: Limited to expressing only individual words or short, memorized phrases.\n    Grammar diversity and accuracy: Incapable of using language to create genuinely communicative statements; may attempt simple memorized phrases but lacks accuracy and flexible structure.\n    Pronunciation: Exhibiting some positive performance of Level 4, but with numerous difficulties in understanding.\n\n    Level 1:\n    Fluency and coherence: Does not communicate.\n    Vocabulary richness: Unable to express language.\n    Grammar diversity and accuracy: No ability to create even simple, singular statements.\n    Pronunciation: No discernible pronunciation.\n    The final scoring criteria are: 9 points for perfect performance, 8 points for very good performance, 7 points for good performance, 6 points for average performance, 5 points for relatively poor performance, 4 points for very poor performance, and 3 points for A score of 2 can only express some basic meanings, a score of 2 can only express very basic meanings, and a score of 1 can hardly express any meaning. If a candidate scores below 5 on any one of the marked items, the overall score will also drop below 5. In addition, the IELTS speaking score will also consider whether the candidate's oral expression ability can be applied in different situations and occasions, such as business, academic and so on.\n        ```",
#     "English_teacher": "I want you to act as a teacher and improver of spoken English. My degree is A2 level, sometimes I may not understand what you are saying, I hope you keep your reply concise and limit your reply to less than 100 characters. I hope you will strictly correct my grammatical errors, typos and factual errors in my oral English, and give me some suggestions so that I can speak more freely, as well as give me C1 level people to answer. I want you to ask me a question in your reply. Now to practice, you can ask me a question first. Remember, I want you to strictly correct my grammatical, spelling, and factual mistakes.""")


# # GPT-4 8K context pricing for input per million tokens
# gpt4_8k_input_price_per_million = 1.50

# # Calculating the total cost for the given text using GPT-4 8K context pricing
# total_cost_gpt4_8k = (token_count_shorter_text / 1000000) * gpt4_8k_input_price_per_million
# print(token_count_shorter_text, total_cost_gpt4_8k*31)
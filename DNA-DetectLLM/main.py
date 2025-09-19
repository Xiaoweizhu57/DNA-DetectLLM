from dna_detectllm import DetectLLM
import jsonlines
from tqdm import tqdm
import json
from transformers import GPT2Tokenizer, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./Model/falcon-7b")
# tokenizer = AutoTokenizer.from_pretrained("/data/llm/Llama-3-8B")
# tokenizer = AutoTokenizer.from_pretrained("/data/llm/Mistral/Mistral-7B-v0.1")
# tokenizer = AutoTokenizer.from_pretrained("/data/llm/llama-7b", trust_remote_code=True)


bino = DetectLLM()

test_text = "The concerts, set to be held on two separate days this month, are part of an ambitious project spearheaded by music educators from across the country. The project encourages students from a wide age range and broad spectrum of musical abilities to engage with, and perform their interpretations of, 10 notable pieces from the classical music canon.\n\nThe 10 pieces of music, carefully curated by renowned orchestra conductors and music teachers, include works from celebrated composers such as Beethoven, Mozart, Chopin, and Tchaikovsky. The selection has been designed to introduce the students to different eras, styles, and techniques of classical music.\n\nThe students, who have spent the past year studying, exploring, and rehearsing their respective pieces, are invited from various music schools nationwide. Their performances will vary from solo renditions on a variety of traditional and modern instruments to group performances, offering a fresh perspective to these evergreen compositions.\n\n\"The aim is to demonstrate that classical music is not just something to be passively listened to, but can actively be used as a catalyst for creativity and personal expression,\" says James Morris, one of the music educators involved in the project. \"By encouraging the students to create their own interpretations of these pieces, we hope to foster a deeper understanding and appreciation of classical music.\"\n\nThe Royal Albert Hall, a historic venue known for its acoustics and grandeur, adds to the charm of this event. It has been supporting youth music initiatives for many years and this project aligns perfectly with its ongoing dedication to fostering young talents and promoting music education.\n\nOrganisers hope that the project will go beyond inspiring the student performers, by also reaching a wider audience, including those who may have previously felt disconnected from the world of classical music. The concerts will be recorded and later broadcast on national radio, making the performances accessible to everyone.\n\nThe two concerts promise to be a celebration of young talent, creativity, and an adventurous spirit of reinterpretation. With the backdrop of the magnificent Royal Albert Hall, audiences can look forward to experiencing classical music like never before."

score = bino.compute_score(test_text)

print(score)   




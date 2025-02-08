import pandas as pd
import os
import re

directory = 'Code/EmotionDataset'
emotions_data = []

for filename in os.listdir(directory):
    if filename.endswith('.py'):
        emotion = filename.replace('.py', '').lower()
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            sentences = re.findall(r'"(.*?)"', content)
            for sentence in sentences:
                emotions_data.append({'emotion': emotion, 'sentence': sentence})

df = pd.DataFrame(emotions_data)

csv_path = os.path.join(directory, 'emotions_dataset.csv')
df.to_csv(csv_path, index=False)
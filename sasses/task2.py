import numpy as np
import sys
import re
sys.path.append('../')

def extract(text):
    for c in '(){}[]\'".,:;?!%$“”’#|':
        text = text.replace(c, '')
    for c in '-/–&=':
        text = text.replace(c, ' ')
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[0-9]', '', text)
    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')
    text = text.split(' ')
    return text

text = open('../datasets/my_writing.txt').read()
text = extract(text)

for i in range(len(text)):
    print (' '.join(text[i:i+7]), end='\n')

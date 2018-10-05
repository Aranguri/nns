from bs4 import BeautifulSoup
import requests
from utils import *

class Task:
    num_words = 4

    def __init__(self):
        self.content = get_content('https://en.wikipedia.org/wiki/2018_Sulawesi_earthquake_and_tsunami')
        common_words = open('1000.txt').read()
        self.words = common_words.split('\n')
        self.vocab_size = len(self.words)
        self.word_to_i = {w: i for i, w in enumerate(self.words)}
        self.i = np.random.randint(0, 1000)

    def get_case(self):
        words = self.content[self.i:self.i + self.num_words + 1]
        #if not all([w in self.words for w in words]):
        #    self.i += 1
        #    return self.get_case()
        #print (self.i)
        print (words)

        ixs = [one_of_k(self.word_to_i[w], len(self.word_to_i)) for w in words]
        x, t = ixs[:self.num_words], ixs[self.num_words]
        x = np.array([x]).swapaxes(0, 1).swapaxes(1, 2)
        self.i += 1 # or 5
        return x, np.array([t])

def get_content(site):
    page = requests.get(site)
    soup = BeautifulSoup(page.content, 'html.parser')
    text = ''.join([p.text for p in soup.find_all('p')])
    text = clean(text)
    text = text.split(' ')
    return text

def clean(text):
    left = ["'", ')', '.', ',', ':', ';']
    right = ['(', '\n']
    both = ['[', ']', '-', '"']

    for c in left:
        text = text.replace(c, ' ' + c)

    for c in right:
        text = text.replace(c, c + ' ')

    for c in both:
        text = text.replace(c, ' ' + c + ' ')

    #add spaces between numbers. eg 1954 => 1 9 5 4
    text = text.replace('  ', ' ')
    return text

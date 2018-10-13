import sys
sys.path.append('../')
from utils import *
import numpy as np
import re

i = 0
docs = []

with open('../datasets/wikipedia2.xml') as infile:
    for line in infile:
        i += 1
        if '<title>' in line:
            title = multiremove(line, ['<title>', '</title>', '\n'])
            if len(docs) > 0 and len(docs[-1]) < 50:
                del docs[-1]
            docs.append([])#{'title': title, 'texts': [], 'categories': []})

        if '[[Category:' in line:
            categories = multiremove(line, ['[[Category:', ']]\n]', ']]</text>', '| '])
            #docs[-1]['categories'].append(categories)

        if len(line) > 400:
            items = [r'{{[^}]*}}', r'\[\[[^|\]]*\|', r'name=[^\s]*']
            text = multiremove(line, items, regex=True)
            text = multiremove(text, ['[[', ']]', '&gt;', '&quot;', '&lt;', '/ref', 'ref', '/', "'", '&amp;nbsp;'])
            if len(text) > 400:
                docs[-1].append(text)

        if i == 1000000:
            save('processed-wiki', docs)
            break

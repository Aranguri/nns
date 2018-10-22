import json
from pprint import pprint

with open('../datasets/squad.json') as f:
    data = json.load(f)
    print (data['data'][1]['paragraphs'][1]['qas'][10]['question'])

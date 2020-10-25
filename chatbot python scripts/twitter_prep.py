from itertools import zip_longest
import re

data_path = "ADD YOUR CHOSEN TOPIC FILE PATH HERE"

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
  lines = f.read().split('\n')

lines = [re.sub(r"(?:\@|https?\://)\S+", "", line).strip() for line in lines]

# group lines by response pair

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
pairs = list(grouper(lines, 2))

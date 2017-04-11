import read
from collections import Counter

df = read.load_data()

long_str = ' '
headline = df['headline'].tolist()

for hl in headline:
    long_str = long_str + str(hl) + ' ' 
word_list = long_str.split(' ')
word_list = [x.lower() for x in word_list]

print(word_list[:3])
word_count = Counter(word_list).most_common(100)

print(word_count)
"""
    word2vec
"""
import re
data = """

   I have a good friend, and her name is Li Hua. We have become friends for about two years. 
   She is very kind. When I step into the classroom for the first time, she helps me to get 
   familiar with the strange environment. The most important thing is that we share the same 
   interest, so we have a lot in common. I cherish our friendship so much.
    """

data_list = re.split(r"[^0-9a-z\S]", data)

word_count = {}

for word in word_count:
    if word:
        word_count[word] = word_count.get(word, 0)


class BinaryTree(object):

    def __init__(self):
        self.left = None
        self.right = None


word_list = [(k, v) for k, v in word_count.items()]

word_list.sort(key=lambda x: x[1])

class HuffmanTree(object):

    def __init__(self):


        self



print(data_list)
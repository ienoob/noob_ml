"""
    word2vec
"""
import re
import heapq
data = """

   I have a good friend, and her name is Li Hua. We have become friends for about two years. 
   She is very kind. When I step into the classroom for the first time, she helps me to get 
   familiar with the strange environment. The most important thing is that we share the same 
   interest, so we have a lot in common. I cherish our friendship so much.
    """

data_list = re.split(r"[^0-9a-z\S]", data)

word_count = {}

for word in data_list:
    if word:
        word_count[word] = word_count.get(word, 0) + 1


class BinaryTree(object):

    def __init__(self):
        self.left = None
        self.right = None
        self.value = []


word_list = [(v, k) for k, v in word_count.items()]
word2id = {k1:i for i, (k1, v1) in enumerate(word_list)}

word_list.sort(key=lambda x: x[1])
print(word_list)
print(len(word_list))

# goujian huofumanshu
input_q = word_list

heapq.heapify(input_q)

while len(input_q) > 1:
    p1, v1 = heapq.heappop(input_q)
    p2, v2 = heapq.heappop(input_q)
    if isinstance(v1, str):
        tree = BinaryTree()
        tree.value.append(v1)

        v1 = tree
    if isinstance(v2, str):
        tree = BinaryTree()
        tree.value.append(v2)

        v2 = tree

    tree = BinaryTree()
    tree.left = v1
    tree.right = v2
    tree.value = v1.value + v2.value

    input_q.append((p1+p2, tree))




class HuffmanTree(object):

    def __init__(self):


        self



print(data_list)
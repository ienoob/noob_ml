"""
    word2vec
"""
import re
import heapq
import numpy as np

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
        self.left_q = []
        self.value = []


word_list = [(v, k) for k, v in word_count.items()]
word2id = {k1: i for i, (v1, k1) in enumerate(word_list)}

word_list = [(v, word2id[k], word2id[k]) for v, k in word_list]
word_list.sort(key=lambda x: x[0])
print(word_list)
print(len(word_list))

# goujian huofumanshu
input_q = word_list

heapq.heapify(input_q)

while len(input_q) > 1:
    p1, n1, v1 = heapq.heappop(input_q)
    p2, n2, v2 = heapq.heappop(input_q)
    if isinstance(v1, int):
        tree = BinaryTree()
        tree.value.append(v1)

        v1 = tree
    if isinstance(v2, int):
        tree = BinaryTree()
        tree.value.append(v2)

        v2 = tree

    tree = BinaryTree()
    tree.left = v1
    tree.right = v2
    tree.value = v1.value + v2.value
    tree.left_q = v1.value

    input_q.append((p1+p2, n1, tree))

x, y, hft = input_q[0]


def f(input_id, input_tree, path):
    if len(input_tree.left_q) == 0 and input_id in tree.value:
        return path
    if input_id in input_tree.left_q:
        return f(input_id, input_tree.left, path+[0])
    else:
        return f(input_id, input_tree.right, path+[1])


paht = f(4, hft, [])
print(paht)


point_num = 0
word_embed = 5


def f1(input_tree):

    if input_tree is None:
        return 0
    v = 1
    if input_tree.left:
        v += f1(input_tree.left)
    if input_tree.right:
        v += f1(input_tree.right)

    return v


print("word num {}".format(len(word2id)))

print(f1(hft))


class HuffmanTree(object):
    def __init__(self):

        self



print(data_list)
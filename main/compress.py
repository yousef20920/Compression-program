"""
Assignment 2 starter code
CSC148, Winter 2023

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from typing import Tuple

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    list_bits = list(text)
    dict_bits = {}
    for bit in list_bits:
        if bit not in dict_bits:
            dict_bits[bit] = 1
        else:
            dict_bits[bit] += 1
    return dict_bits
    # https://www.programiz.com/python-programming/methods/built-in/bytes
    #
    # https://docs.python.org/3/library/stdtypes.html#bytes


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    >>> d = {'h': 23, 'u': 21, 'f': 11, 'm': 9, 'a': 50, 'n': 5}
    >>> build_huffman_tree(d)
    """
    freq_list = [(freq, HuffmanTree(symbol))
                 for symbol, freq in freq_dict.items()]
    # https://commons.wikimedia.org/wiki/File:HuffmanCodeAlg.png#/media/File:Hu
    # ffmanCodeAlg.png

    if len(freq_list) == 1:
        dum_tree = HuffmanTree((freq_list[0][1].symbol + 1) % 256)
        freq_list.append((0, dum_tree))

    freq_list.sort(key=lambda x: x[0])

    while len(freq_list) > 1:
        left_freq, left_tree = freq_list.pop(0)
        right_freq, right_tree = freq_list.pop(0)

        new_freq = left_freq + right_freq
        new_tree = HuffmanTree(None, left_tree, right_tree)

        freq_list.append((new_freq, new_tree))
        freq_list.sort(key=lambda x: x[0])

    return freq_list[0][1]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    code_dict = {}

    def traverse(tree1: HuffmanTree, code: str) -> None:
        if tree1 is None:
            return
        if tree1.is_leaf():
            code_dict[tree1.symbol] = code
        else:
            traverse(tree1.left, code + '0')
            traverse(tree1.right, code + '1')

    traverse(tree, '')
    return code_dict
    # https://www.mathsisfun.com/binary-number-system.html


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    def helper(tree1: HuffmanTree, count: int) -> int:
        if not tree1:
            return count
        count = helper(tree1.left, count)
        count = helper(tree1.right, count)
        if not tree1.is_leaf():
            tree1.number = count
            count += 1
        return count

    helper(tree, 0)


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    codes = get_codes(tree)  # dictionary containing
    # the Huffman tree codes for each symbol in the tree
    bits = 0
    freq = 0
    for s in freq_dict:
        bits += len(codes[s]) * freq_dict[s]  # the len code of
        # the symbol + the freq of the symbol
        freq += freq_dict[s]  # the sum of all the soymbols
    return bits / freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    result = bytearray()
    current_byte = 0
    bits_filled = 0

    for symbol in text:
        code = codes[symbol]
        for bit in code:  # Update the value of current_byte
            current_byte = (current_byte << 1) | int(bit)
            # https://www.geeksforgeeks.org/python-bitwise-operators/
            bits_filled += 1
            if bits_filled == 8:
                result.append(current_byte)
                current_byte = 0
                bits_filled = 0

    if bits_filled > 0:
        current_byte <<= (8 - bits_filled)
        # https://www.geeksforgeeks.org/python-bitwise-operators/
        result.append(current_byte)

    return bytes(result)
    # https://www.programiz.com/python-programming/methods/built-in/bytes
    #
    # https://docs.python.org/3/library/stdtypes.html#bytes


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    result = bytearray()

    def postorder(huf: HuffmanTree) -> None:
        if not huf.is_leaf():
            postorder(huf.left)
            postorder(huf.right)
            lbyte_1 = 0 if huf.left.is_leaf() else 1
            lbyte_2 = huf.left.symbol if huf.left.is_leaf() \
                else huf.left.number
            rbyte_1 = 0 if huf.right.is_leaf() else 1
            rbyte_2 = huf.right.symbol if huf.right.is_leaf() \
                else huf.right.number
            result.extend((lbyte_1, lbyte_2, rbyte_1, rbyte_2))

    postorder(tree)
    return bytes(result)
    # https://www.programiz.com/python-programming/methods/built-in/bytes
    #
    # https://docs.python.org/3/library/stdtypes.html#bytes


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    node_map = {}
    for i, node in enumerate(node_lst):  # (index, ReadNode)
        node_map[i] = HuffmanTree(None, None, None)
        if node.l_type == 0:
            node_map[i].left = HuffmanTree(node.l_data, None, None)
        if node.r_type == 0:
            node_map[i].right = HuffmanTree(node.r_data, None, None)
        if node.l_type == 1:
            node_map[i].left = node_map[node.l_data]
        if node.r_type == 1:
            node_map[i].right = node_map[node.r_data]
    return node_map[root_index]


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    lst = []
    for i in range(root_index + 1):
        node = node_lst[i]
        if node.l_type == 0 and node.r_type == 0:
            lst.append(HuffmanTree(None, HuffmanTree(node.l_data),
                                   HuffmanTree(node.r_data)))
        elif node.l_type == 0 and node.r_type == 1:
            right_subtree = lst.pop()
            lst.append(HuffmanTree(None, HuffmanTree(node.l_data),
                                   right_subtree))
        elif node.l_type == 1 and node.r_type == 0:
            left_subtree = lst.pop()
            lst.append(HuffmanTree(None, left_subtree,
                                   HuffmanTree(node.r_data)))
        else:
            right_subtree = lst.pop()
            left_subtree = lst.pop()
            lst.append(HuffmanTree(None, left_subtree, right_subtree))

    return lst.pop()


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    def process_byte(byte: int, node: HuffmanTree) -> Tuple[HuffmanTree, bool]:
        for i in range(8):
            bit = ((byte >> (7 - i)) & 1)  # (byte // (2 ** (7 - i))) % 2
            # https://www.geeksforgeeks.org/python-bitwise-operators/
            if bit == 0:
                node = node.left
            else:
                node = node.right
            if node.is_leaf():
                result.append(node.symbol)
                node = tree
                if len(result) == size:
                    return tree, True
        return node, False

    result = bytearray()
    node = tree
    for byte in text:
        node, done = process_byte(byte, node)
        if done:
            break
    return bytes(result)
    # https://www.programiz.com/python-programming/methods/built-in/bytes
    #
    # https://docs.python.org/3/library/stdtypes.html#bytes


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    sorted_freqs = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    leaves = []

    def get_leaves(node: HuffmanTree) -> None:
        if node is None:
            return
        if node.is_leaf():
            leaves.append(node)
        else:
            get_leaves(node.left)
            get_leaves(node.right)

    get_leaves(tree)

    for leaf in leaves:
        symbol, _ = sorted_freqs.pop(0)
        leaf.symbol = symbol


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # import python_ta
    #
    # python_ta.check_all(config={
    #     'allowed-io': ['compress_file', 'decompress_file'],
    #     'allowed-import-modules': [
    #         'python_ta', 'doctest', 'typing', '__future__',
    #         'time', 'utils', 'huffman', 'random'
    #     ],
    #     'disable': ['W0401']
    # })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")

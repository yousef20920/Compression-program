from __future__ import annotations

from random import shuffle

import pytest
from hypothesis import given, assume, settings
from hypothesis.strategies import binary, integers, dictionaries, text

from compress import *

import random

settings.register_profile("norand", settings(derandomize=True, max_examples=200))
settings.load_profile("norand")


# === Test Byte Utilities ===
# Technically, these utilities are given to you in the starter code, so these
# first 3 tests below are just intended as a sanity check to make sure that you
# did not modify these methods and are therefore using them incorrectly.
# You will not be submitting utils.py anyway, so these first three tests are
# solely for your own benefit, as a sanity check.

@given(integers(0, 255))
def test_byte_to_bits(b: int) -> None:
    """ Test that byte_to_bits produces binary strings of length 8."""
    assert set(byte_to_bits(b)).issubset({"0", "1"})
    assert len(byte_to_bits(b)) == 8


@given(text(["0", "1"], min_size=0, max_size=8))
def test_bits_to_byte(s: str) -> None:
    """ Test that bits_to_byte produces a byte."""
    b = bits_to_byte(s)
    assert isinstance(b, int)
    assert 0 <= b <= 255


@given(integers(0, 255), integers(0, 7))
def test_get_bit(byte: int, bit_pos: int) -> None:
    """ Test that get_bit(byte, bit) produces  bit values."""
    b = get_bit(byte, bit_pos)
    assert isinstance(b, int)
    assert 0 <= b <= 1


# === Test the compression code ===

@given(binary(min_size=0, max_size=1000))
def test_build_frequency_dict(byte_list: bytes) -> None:
    """ Test that build_frequency_dict returns dictionary whose values sum up
    to the number of bytes consumed.
    """
    # creates a copy of byte_list, just in case your implementation of
    # build_frequency_dict modifies the byte_list
    b, d = byte_list, build_frequency_dict(byte_list)
    assert isinstance(d, dict)
    assert sum(d.values()) == len(b)


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_build_huffman_tree(d: dict[int, int]) -> None:
    """ Test that build_huffman_tree returns a non-leaf HuffmanTree."""
    t = build_huffman_tree(d)
    assert isinstance(t, HuffmanTree)
    assert not t.is_leaf()


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_get_codes(d: dict[int, int]) -> None:
    """ Test that the sum of len(code) * freq_dict[code] is optimal, so it
    must be invariant under permutation of the dictionary.
    Note: This also tests build_huffman_tree indirectly.
    """
    t = build_huffman_tree(d)
    c1 = get_codes(t)
    d2 = list(d.items())
    shuffle(d2)
    d2 = dict(d2)
    t2 = build_huffman_tree(d2)
    c2 = get_codes(t2)
    assert sum([d[k] * len(c1[k]) for k in d]) == \
           sum([d2[k] * len(c2[k]) for k in d2])


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_number_nodes(d: dict[int, int]) -> None:
    """ If the root is an interior node, it must be numbered two less than the
    number of symbols, since a complete tree has one fewer interior nodes than
    it has leaves, and we are numbering from 0.
    Note: this also tests build_huffman_tree indirectly.
    """
    t = build_huffman_tree(d)
    assume(not t.is_leaf())
    count = len(d)
    number_nodes(t)
    assert count == t.number + 2


@given(dictionaries(integers(min_value=0, max_value=255), integers(min_value=1, max_value=1000), dict_class=dict,
                    min_size=2, max_size=256))
def test_avg_length(d: dict[int, int]) -> None:
    """ Test that avg_length returns a float in the interval [0, 8], if the max
    number of symbols is 256.
    """
    t = build_huffman_tree(d)
    f = avg_length(t, d)
    assert isinstance(f, float)
    assert 0 <= f <= 8.0


@given(binary(min_size=2, max_size=1000))
def test_compress_bytes(b: bytes) -> None:
    """ Test that compress_bytes returns a bytes object that is no longer
    than the input bytes. Also, the size of the compressed object should be
    invariant under permuting the input.
    Note: this also indirectly tests build_frequency_dict, build_huffman_tree,
    and get_codes.
    """
    d = build_frequency_dict(b)
    t = build_huffman_tree(d)
    c = get_codes(t)
    compressed = compress_bytes(b, c)
    assert isinstance(compressed, bytes)
    assert len(compressed) <= len(b)
    lst = list(b)
    shuffle(lst)
    b = bytes(lst)
    d = build_frequency_dict(b)
    t = build_huffman_tree(d)
    c = get_codes(t)
    compressed2 = compress_bytes(b, c)
    assert len(compressed2) == len(compressed)


@given(binary(min_size=2, max_size=1000))
def test_tree_to_bytes(b: bytes) -> None:
    """ Test that tree_to_bytes generates a bytes representation of a postorder
    traversal of a tree's internal nodes.
    Since each internal node requires 4 bytes to represent, and there are
    1 fewer internal nodes than distinct symbols, the length of the bytes
    produced should be 4 times the length of the frequency dictionary, minus 4.
    Note: also indirectly tests build_frequency_dict, build_huffman_tree, and
    number_nodes.
    """
    d = build_frequency_dict(b)
    assume(len(d) > 1)
    t = build_huffman_tree(d)
    number_nodes(t)
    output_bytes = tree_to_bytes(t)
    dictionary_length = len(d)
    leaf_count = dictionary_length
    assert (4 * (leaf_count - 1)) == len(output_bytes)


# === Test a roundtrip conversion

@given(binary(min_size=1, max_size=1000))
def test_round_trip_compress_bytes(b: bytes) -> None:
    """ Test that applying compress_bytes and then decompress_bytes
    will produce the original text.
    """
    text = b
    freq = build_frequency_dict(text)
    assume(len(freq) > 1)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    compressed = compress_bytes(text, codes)
    decompressed = decompress_bytes(tree, compressed, len(text))
    assert text == decompressed


def test_compress_byts() -> None:
    d_big = {0: "0", 1: "00", 2: "11", 3: "110", 4: "111", 5: "101", 6: "111", 7: "1110", 8: "1001", 9: "1101", 10: "1000", 11: "1111"}
    r = [random.randint(0, 11) for i in range(99999)]
    text_big = bytes(r)
    result_big = compress_bytes(text_big, d_big)
    assert [byte_to_bits(byte) for byte in result_big] == [byte_to_bits(byte) for byte in result_big]
    d_empty = {0: "0"}
    text_empty = bytes([])
    result_empty = compress_bytes(text_empty, d_empty)
    assert [byte_to_bits(byte) for byte in result_empty] == []


def test_empty_compress_byts() -> None:
    d_empty = {}
    text_empty = bytes([])
    result_empty = compress_bytes(text_empty, d_empty)
    assert [byte_to_bits(byte) for byte in result_empty] == []


def test_tree_to_bytes() -> None:
    left = HuffmanTree(None, HuffmanTree(None, HuffmanTree(1), HuffmanTree(2)), HuffmanTree(3))
    right = HuffmanTree(None, HuffmanTree(None, HuffmanTree(4), HuffmanTree(5)), HuffmanTree(6))
    tree1 = HuffmanTree(None, left, right)
    number_nodes(tree1)
    assert list(tree_to_bytes(tree1)) == [0, 1, 0, 2, 1, 0, 0, 3, 0, 4, 0, 5, 1, 2, 0, 6, 1, 1, 1, 3]


def test_generate_tree_general() -> None:
    lst1 = [ReadNode(0, 1, 0, 2), ReadNode(1, 0, 0, 3), ReadNode(0, 4, 0, 5), ReadNode(1, 2, 0, 6), ReadNode(1, 1, 1, 3)]
    assert generate_tree_general(lst1, 4) == HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(1, None, None), HuffmanTree(2, None, None)), HuffmanTree(3, None, None)), HuffmanTree(None, HuffmanTree(None, HuffmanTree(4, None, None), HuffmanTree(5, None, None)), HuffmanTree(6, None, None)))
    lst2 = [ReadNode(0, 1, 0, 2)]
    assert generate_tree_general(lst1, 0) == HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))


def test_generate_tree_postorder() -> None:
    lst = [ReadNode(0, 1, 0, 2), ReadNode(1, 0, 0, 3), ReadNode(0, 4, 0, 5), ReadNode(1, 2, 0, 6), ReadNode(1, 1, 1, 3)]
    assert generate_tree_postorder(lst, 4) == HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(1, None, None), HuffmanTree(2, None, None)), HuffmanTree(3, None, None)), HuffmanTree(None, HuffmanTree(None, HuffmanTree(4, None, None), HuffmanTree(5, None, None)), HuffmanTree(6, None, None)))


def test_decompress_bytes_single_char():
    tree = build_huffman_tree(build_frequency_dict(b'a'))
    compressed_text = compress_bytes(b'a', get_codes(tree))
    decompressed_text = decompress_bytes(tree, compressed_text, len(b'a'))
    assert decompressed_text == b'a'


def test_decompress_bytes_repeated_chars():
    tree = build_huffman_tree(build_frequency_dict(b'aaaaaa'))
    compressed_text = compress_bytes(b'aaaaaa', get_codes(tree))
    decompressed_text = decompress_bytes(tree, compressed_text, len(b'aaaaaa'))
    assert decompressed_text == b'aaaaaa'


def test_decompress_bytes_all_chars():
    text = bytes(range(256))
    tree = build_huffman_tree(build_frequency_dict(text))
    compressed_text = compress_bytes(text, get_codes(tree))
    decompressed_text = decompress_bytes(tree, compressed_text, len(text))
    assert decompressed_text == text


def test_decompress_bytes_large_text():
    text = b'a' * 1000000
    tree = build_huffman_tree(build_frequency_dict(text))
    compressed_text = compress_bytes(text, get_codes(tree))
    decompressed_text = decompress_bytes(tree, compressed_text, len(text))
    assert decompressed_text == text


def test_compress_bytes_single_char():
    d = {ord('a'): '0'}
    compressed_text = compress_bytes(b'a', d)
    assert compressed_text == bytes([0])


def test_compress_bytes_repeated_chars():
    d = {ord('a'): '0'}
    compressed_text = compress_bytes(b'aaaaaa', d)
    assert compressed_text == bytes([0])


def test_compress_bytes_multiple_chars():
    d = {ord('a'): '0', ord('b'): '10', ord('c'): '11'}
    compressed_text = compress_bytes(b'abcabc', d)
    assert [byte_to_bits(byte) for byte in compressed_text] == ['01011010', '11000000']


def test_get_codes_single_node():
    codes = get_codes(HuffmanTree(1))
    assert codes == {1: ''}


def test_get_codes_simple_tree():
    tree = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    codes = get_codes(tree)
    assert codes == {1: '0', 2: '1'}


def test_get_codes_complex_tree():
    left = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    right = HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))
    tree = HuffmanTree(None, left, right)
    codes = get_codes(tree)
    assert codes == {1: '00', 2: '01', 3: '10', 4: '11'}


def test_tree_to_bytes_empty_tree():
    tree = HuffmanTree(None)
    number_nodes(tree)
    tree_bytes = tree_to_bytes(tree)
    assert tree_bytes == b''


def test_tree_to_bytes_single_node():
    tree = HuffmanTree(1)
    number_nodes(tree)
    tree_bytes = tree_to_bytes(tree)
    assert tree_bytes == b''


def test_tree_to_bytes_simple_tree():
    tree = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    number_nodes(tree)
    tree_bytes = tree_to_bytes(tree)
    assert tree_bytes == bytes([0, 1, 0, 2])


def test_tree_to_bytes_complex_tree():
    left = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    right = HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))
    tree = HuffmanTree(None, left, right)
    number_nodes(tree)
    tree_bytes = tree_to_bytes(tree)
    assert tree_bytes == bytes([0, 1, 0, 2, 0, 3, 0, 4, 1, 0, 1, 1])


def test_generate_tree_postorder_single_node():
    lst = [ReadNode(0, 1, 0, 2)]
    tree = generate_tree_postorder(lst, 0)
    assert tree == HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))


def test_generate_tree_postorder_simple_tree():
    lst = [ReadNode(0, 1, 0, 2), ReadNode(0, 3, 0, 4), ReadNode(1, 0, 1, 1)]
    tree = generate_tree_postorder(lst, 2)
    assert tree == HuffmanTree(None, HuffmanTree(None, HuffmanTree(1), HuffmanTree(2)), HuffmanTree(None, HuffmanTree(3), HuffmanTree(4)))


def test_generate_tree_postorder_complex_tree():
    lst = [ReadNode(0, 1, 0, 2), ReadNode(0, 3, 0, 4), ReadNode(1, 0, 1, 1),
           ReadNode(0, 5, 0, 6), ReadNode(1, 2, 1, 3)]
    tree = generate_tree_postorder(lst, 4)
    assert tree == HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(1), HuffmanTree(2)), HuffmanTree(None, HuffmanTree(3), HuffmanTree(4))), HuffmanTree(None, HuffmanTree(5), HuffmanTree(6)))


def test_number_nodes2():
    left = HuffmanTree(1)
    right = HuffmanTree(4)
    tree3 = HuffmanTree(None, left, right)
    number_nodes(tree3)
    assert tree3.number == 0


def test_improve_tree():
    left = HuffmanTree(None, HuffmanTree(99, None, None), HuffmanTree(100, None, None))
    right = HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    tree = HuffmanTree(None, left, right)

    perfect = HuffmanTree(None, HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)), HuffmanTree(None, HuffmanTree(99, None, None), HuffmanTree(None, HuffmanTree(100, None, None), HuffmanTree(101, None, None))))

    freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    improve_tree(tree, freq)
    assert tree == perfect


# def test_improve_tree_2():
#     left = HuffmanTree(None, HuffmanTree(99, None, None), HuffmanTree(100, None, None))
#     right = HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
#     tree = HuffmanTree(None, left, right)
#
#     perfect = HuffmanTree(None, HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(97, None, None)), HuffmanTree(None, HuffmanTree(100, None, None), HuffmanTree(None, HuffmanTree(98, None, None), HuffmanTree(99, None, None))))
#
#     freq = {97: 12, 98: 3, 99: 2, 100: 6, 101: 23}
#     assert avg_length(tree, freq) == 2.3260869565217392
#     improve_tree(tree, freq)
#     assert avg_length(tree, freq) == 2.108695652173913
#     assert tree == perfect


def test_improve_tree_3():
    left = HuffmanTree(None, HuffmanTree(99, None, None), HuffmanTree(100, None, None))
    right = HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    tree = HuffmanTree(None, left, right)

    perfect = tree
    freq = {97: 1, 98: 2, 99: 3, 100: 4, 101: 5}
    assert avg_length(tree, freq) == 2.2
    improve_tree(tree, freq)
    assert avg_length(tree, freq) == 2.2
    assert tree == perfect


def test_improve_tree_2():
    left = HuffmanTree(None, HuffmanTree(99, None, None), HuffmanTree(100, None, None))
    right = HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    tree = HuffmanTree(None, left, right)

    perfect1 = HuffmanTree(None, HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(100, None, None)), HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(None, HuffmanTree(98, None, None), HuffmanTree(99, None, None))))
    perfect2 = HuffmanTree(None, HuffmanTree(None, HuffmanTree(100, None, None), HuffmanTree(97, None, None)), HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(None, HuffmanTree(99, None, None), HuffmanTree(98, None, None))))
    perfect3 = HuffmanTree(None, HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(100, None, None)), HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(None, HuffmanTree(99, None, None), HuffmanTree(98, None, None))))
    perfect4 = HuffmanTree(None, HuffmanTree(None, HuffmanTree(100, None, None), HuffmanTree(97, None, None)), HuffmanTree(None, HuffmanTree(101, None, None), HuffmanTree(None, HuffmanTree(98, None, None), HuffmanTree(99, None, None))))

    freq = {97: 12, 98: 3, 99: 2, 100: 6, 101: 23}
    assert avg_length(tree, freq) == 2.3260869565217392
    improve_tree(tree, freq)
    assert avg_length(tree, freq) == 2.108695652173913
    assert avg_length(perfect1, freq) == 2.108695652173913
    assert avg_length(perfect2, freq) == 2.108695652173913
    assert avg_length(perfect3, freq) == 2.108695652173913
    assert avg_length(perfect4, freq) == 2.108695652173913
    assert tree == perfect1 or tree == perfect2 or tree == perfect3 or tree == perfect4


if __name__ == "__main__":
    pytest.main(["test_huffman_properties_basic.py"])
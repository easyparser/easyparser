import pytest

from easyparser.base import Chunk, ChunkGroup


def test_construct_empty():
    group = ChunkGroup()

    assert bool(group) is False
    assert len(group) == 0
    assert group._root_id is None
    assert None in group._chunks
    assert group._roots == {}


def test_construct_supplied_chunks_only():
    chunk1 = Chunk(mimetype="plain/text", text="hello world")
    chunk2 = Chunk(mimetype="plain/text", text="goodbye world")
    group = ChunkGroup([chunk1, chunk2])

    assert bool(group) is True
    assert len(group) == 2
    assert group._root_id is None
    assert None in group._chunks
    assert group._roots == {}


def test_construct_supplied_root_only():
    root = Chunk(mimetype="plain/text", text="hello world")
    group = ChunkGroup(root=root)

    assert bool(group) is False, "Should still be empty as no chunks were supplied"
    assert len(group) == 0, "Should still be empty as no chunks were supplied"
    assert (
        group._root_id == root.id
    ), "The root id should be the id of the supplied root"
    assert None not in group._chunks, "Should not have a None key as root was supplied"
    assert group._root_id in group._chunks
    assert group._roots == {root.id: root}


def test_append():
    chunk1 = Chunk(mimetype="plain/text", text="hello world")
    chunk2 = Chunk(mimetype="plain/text", text="goodbye world")

    group = ChunkGroup()
    assert len(group) == 0
    group.append(chunk1)
    assert len(group) == 1
    group.append(chunk2)
    assert len(group) == 2


def test_append_multiple_groups():
    """Should raise error when append to a group containing multiple roots"""
    root1 = Chunk(mimetype="plain/text", text="root1")
    root2 = Chunk(mimetype="plain/text", text="root2")
    chunk1 = Chunk(mimetype="plain/text", text="hello world")
    chunk2 = Chunk(mimetype="plain/text", text="goodbye world")

    group1 = ChunkGroup(root=root1)
    assert len(group1) == 0

    group1.append(chunk1)
    assert len(group1) == 1

    group2 = ChunkGroup(root=root2)
    group1.add_group(group2)

    with pytest.raises(ValueError):
        group1.append(chunk2)

    # Can directly access the group root to append
    group1.groups[root2.id].append(chunk2)
    assert len(group1) == 2


def test_extend():
    chunk1 = Chunk(mimetype="plain/text", text="hello world")
    chunk2 = Chunk(mimetype="plain/text", text="goodbye world")

    group = ChunkGroup()
    assert len(group) == 0
    group.extend([chunk1, chunk2])
    assert len(group) == 2


def test_extend_multiple_groups():
    """Should raise error when extend to a group containing multiple roots"""
    root1 = Chunk(mimetype="plain/text", text="root1")
    root2 = Chunk(mimetype="plain/text", text="root2")
    chunk1 = Chunk(mimetype="plain/text", text="hello world")
    chunk2 = Chunk(mimetype="plain/text", text="goodbye world")

    group1 = ChunkGroup(root=root1)
    assert len(group1) == 0

    group2 = ChunkGroup(root=root2)
    group1.add_group(group2)

    with pytest.raises(ValueError):
        group1.extend([chunk1, chunk2])

    # Can directly access the group root to extend
    group1.groups[root2.id].extend([chunk1, chunk2])
    assert len(group1) == 2


def test_iter():
    chunks = [
        Chunk(mimetype="plain/text", text="hello world"),
        Chunk(mimetype="plain/text", text="goodbye world"),
    ]
    group = ChunkGroup(chunks)
    for idx, chunk in enumerate(group):
        assert chunk.id == chunks[idx].id

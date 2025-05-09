from easyparser.base import Chunk, CType
from easyparser.mime import MimeType


class TestChunkCleanUnwrapSingleChild:
    """Test Chunk.clean() method with focus on unwrap_single_child option"""

    def test_unwrap_single_child_true_with_empty_parent_and_single_child(self):
        """When unwrap_single_child=True, parent has no content, has a single child,
        expect to inherit child's properties
        """
        # Create parent chunk with no content
        parent = Chunk(
            mimetype=MimeType.text,
            ctype=CType.Div,
            content=None,
        )

        # Create child chunk with content
        child = Chunk(
            mimetype="text/html",
            ctype=CType.Para,
            content="<p>Child content</p>",
            metadata={"key": "value"},
        )

        # Set up parent-child relationship
        parent._child = child

        # Call clean with unwrap_single_child=True (default)
        parent.clean()

        # Verify parent has inherited properties from child
        assert parent.mimetype == "text/html"
        assert parent.ctype == CType.Para
        assert parent.content == "<p>Child content</p>"
        assert parent.metadata == {"key": "value"}
        assert parent.child is None  # Child's child (which was None)

    def test_unwrap_single_child_false_no_inheritance(self):
        """Test when unwrap_single_child=False, no inheritance occurs"""
        # Create parent chunk with no content
        parent = Chunk(
            mimetype=MimeType.text,
            ctype=CType.Div,
            content=None,
        )

        # Create child chunk with content
        child = Chunk(
            mimetype="text/html",
            ctype=CType.Para,
            content="<p>Child content</p>",
            metadata={"key": "value"},
        )

        # Set up parent-child relationship
        parent._child = child

        # Call clean with unwrap_single_child=False
        parent.clean(unwrap_single_child=False)

        # Verify parent has NOT inherited properties from child
        assert parent.mimetype == MimeType.text
        assert parent.ctype == CType.Div
        assert parent.content is None
        assert parent.metadata is None
        assert parent.child == child  # Child still exists

    def test_unwrap_single_child_with_parent_having_content_no_inheritance(self):
        """Test when parent has content, no inheritance occurs even with
        unwrap_single_child=True
        """
        # Create parent chunk WITH content
        parent = Chunk(
            mimetype=MimeType.text,
            ctype=CType.Div,
            content="Parent content",
        )

        # Create child chunk with content
        child = Chunk(
            mimetype="text/html",
            ctype=CType.Para,
            content="<p>Child content</p>",
            metadata={"key": "value"},
        )

        # Set up parent-child relationship
        parent._child = child

        # Call clean with unwrap_single_child=True
        parent.clean(unwrap_single_child=True)

        # Verify parent has NOT inherited properties from child
        assert parent.mimetype == MimeType.text
        assert parent.ctype == CType.Div
        assert parent.content == "Parent content"
        assert parent.metadata is None
        assert parent.child == child  # Child still exists

    def test_unwrap_single_child_with_multiple_children_no_inheritance(self):
        """Test when parent has multiple children, no inheritance occurs even with
        unwrap_single_child=True
        """
        # Create parent chunk with no content
        parent = Chunk(
            mimetype=MimeType.text,
            ctype=CType.Div,
            content=None,
        )

        # Create first child chunk
        child1 = Chunk(
            mimetype="text/html",
            ctype=CType.Para,
            content="<p>Child 1 content</p>",
        )

        # Create second child chunk
        child2 = Chunk(
            mimetype="text/css",
            ctype=CType.Para,
            content="body { color: red; }",
        )

        # Set up parent-child relationship with multiple children
        parent._child = child1
        child1._next = child2

        # Call clean with unwrap_single_child=True
        parent.clean(unwrap_single_child=True)

        # Verify parent has NOT inherited properties from child
        assert parent.mimetype == MimeType.text
        assert parent.ctype == CType.Div
        assert parent.content is None
        assert parent.child == child1  # First child still exists
        assert parent.child.next == child2  # Second child still exists

    def test_unwrap_single_child_with_inline_ctype_preservation(self):
        """Test that when child's ctype is Inline, parent's ctype is preserved"""
        # Create parent chunk with no content but a non-default ctype
        parent = Chunk(
            mimetype=MimeType.text,
            ctype=CType.Div,
            content=None,
        )

        # Create child chunk with Inline ctype
        child = Chunk(
            mimetype="text/html",
            ctype=CType.Inline,  # Important! This should not override parent's ctype
            content="<p>Child content</p>",
        )

        # Set up parent-child relationship
        parent._child = child

        # Call clean with unwrap_single_child=True
        parent.clean(unwrap_single_child=True)

        # Verify parent has inherited properties from child but kept its ctype
        assert parent.mimetype == "text/html"
        assert parent.ctype == CType.Div  # Should not be changed to Inline
        assert parent.content == "<p>Child content</p>"

    def test_unwrap_single_child_with_metadata_merging(self):
        """Test proper merging of metadata when unwrapping child"""
        # Create parent chunk with no content but with metadata
        parent = Chunk(
            mimetype=MimeType.text,
            ctype=CType.Div,
            content=None,
            metadata={"parent_key": "parent_value"},
        )

        # Create child chunk with its own metadata
        child = Chunk(
            mimetype="text/html",
            ctype=CType.Para,
            content="<p>Child content</p>",
            metadata={"child_key": "child_value"},
        )

        # Set up parent-child relationship
        parent._child = child

        # Call clean with unwrap_single_child=True
        parent.clean(unwrap_single_child=True)

        # Verify parent has merged metadata properly
        assert parent.metadata == {
            "parent_key": "parent_value",
            "child_key": "child_value",
        }

    def test_unwrap_single_child_parent_inherits_child_with_child(self):
        """Test that parent inherits child's child when unwrapping"""
        # Create parent chunk with no content
        parent = Chunk(
            mimetype=MimeType.text,
            ctype=CType.Div,
            content=None,
        )

        # Create child chunk
        child = Chunk(
            mimetype="text/html",
            ctype=CType.Para,
            content="<p>Child content</p>",
        )

        # Create grandchild chunk
        grandchild = Chunk(
            mimetype="text/css",
            ctype=CType.Inline,
            content="p { color: blue; }",
        )

        # Set up parent-child-grandchild relationship
        parent._child = child
        child._child = grandchild

        # Call clean with unwrap_single_child=True
        parent.clean(unwrap_single_child=True)

        # Verify parent now points to grandchild
        assert parent.mimetype == "text/html"
        assert parent.content == "<p>Child content</p>"
        assert parent.child == grandchild

    def test_deep_nesting_with_empty_intermediaries(self):
        """Test unwrapping with multiple levels of empty chunks before content chunk"""
        # Create great-grandparent chunk with no content
        great_grandparent = Chunk(
            mimetype=MimeType.text,
            ctype=CType.Div,
            content=None,
            metadata={"level": "great-grandparent"},
        )

        # Create grandparent chunk with no content
        grandparent = Chunk(
            mimetype="application/xml",
            ctype=CType.Div,
            content=None,
            metadata={"level": "grandparent"},
        )

        # Create parent chunk with no content
        parent = Chunk(
            mimetype="text/html",
            ctype=CType.Div,
            content=None,
            metadata={"level": "parent"},
        )

        # Create child chunk with actual content
        child = Chunk(
            mimetype="text/css",
            ctype=CType.Inline,
            content="body { font-family: Arial; }",
            metadata={"level": "child"},
        )

        # Set up the nested structure
        great_grandparent._child = grandparent
        grandparent._child = parent
        parent._child = child

        # Great-grandparent should now have grandparent's content
        great_grandparent.clean()

        # Verify great_grandparent has inherited the content and mimetype from the child
        assert great_grandparent.content == "body { font-family: Arial; }"
        assert great_grandparent.mimetype == "text/css"
        assert (
            great_grandparent.ctype == CType.Div
        )  # Should keep its own ctype (not Inline)

        # Since keys are the same, only the last value will be preserved
        assert great_grandparent.metadata == {"level": "child"}
        assert great_grandparent.child is None

import inspect
import logging
import re
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Callable, Generator, Literal, get_type_hints, overload

from easyparser.mime import get_mime_manager

logger = logging.getLogger(__name__)


class Origin:
    """Represent the origin of an object from another object

    !IMPORTANT: the Origin must be serializable to JSON.

    Args:
        source_id: source object id.
        location: location of the object in the source object. The exact value of
            the location is dependent on the source object. For example, if the
            source object is a folder, the location can be the path of the object;
            if the source object is a PDF file, the location can be a dictionary
            contains the page number and the position of the object.
    """

    def __init__(
        self,
        source_id: str = "",
        location: Any = None,
        protocol: str = "",
        metadata: dict | None = None,
    ):
        self.source_id = source_id
        self.location = location
        self.protocol = protocol
        self.metadata = metadata

    def asdict(self):
        return {
            "source_id": self.source_id,
            "location": self.location,
            "metadata": self.metadata,
            "protocol": self.protocol,
        }

    def __str__(self):
        return f"Origin(protocol={self.protocol}, location={self.location})"

    def __repr__(self):
        return self.__str__()


class CType:
    """Collection of chunk types and its utility"""

    # Chunk with this type will be interpreted as the same level with the parent chunk
    # (e.g. long text)
    Inline = "inline"

    # Chunk with this type will be interpreted as the child of the parent chunk
    Para = "para"
    Fomula = "formula"
    List = "list"
    Table = "table"
    TableRow = "tablerow"
    Header = "header"
    Figure = "figure"
    Code = "code"
    Div = "div"  # logically organized (e.g. a pptx slide)

    # File-level and above chunk
    Root = "root"

    __available_types = None

    @classmethod
    def available_types(cls) -> list:
        if cls.__available_types is None:
            cls.__available_types = [
                "inline",
                "para",
                "formula",
                "list",
                "table",
                "tablerow",
                "header",
                "figure",
                "code",
                "div",
                "root",
            ]

        return cls.__available_types

    @classmethod
    def markdown(cls, chunk) -> str | None:
        """Represent chunk and its children as markdown text

        Args:
            chunk: the chunk to be represented as markdown text

        Returns:
            str: the markdown text
        """
        # If the chunk already has a text, return it
        if chunk.text:
            return chunk.text

        # Otherwise, reconstruct the text from the children
        text: str = chunk.content if isinstance(chunk.content, str) else ""
        child = chunk.child
        while child:
            if child.ctype == CType.Header:
                text += f"\n\n{'#' * (child.origin.location['level'] + 1)} {child.text}"
            elif child.ctype == CType.Table:
                text += f"\n\n{child.text}"
            elif child.ctype == CType.List:
                text += f"\n\n- {child.text}"
            else:
                text += f"\n\n{child.text}"

            child = child.next

        return text


class Chunk:
    """Mandatory fields for an object represented in `easyparser`.

    !IMPORTANT: all fields, except `content`, must be serializable to JSON.

    Args:
        id: unique identifier for the object.
        mimetype: mimetype of the object, plays a crucial role in determining how an
            object is processed and rendered. The official list of mimetypes:
            https://www.iana.org/assignments/media-types/media-types.xhtml
        ctype: the chunk type of object, 1 of CType enum.
        content: content of the object, can be anything (text or bytes), that can be
            understood from the mimetype.
        text: text representation of the object.
        summary: text summary of the object, used as short description in case the
            content is large.
        parent: parent object id. Default to None.
        child: the first child id. Default to None.
        next: next object id. Default to None.
        prev: previous object id. Default to None.
        origin: the location of this object in relative to the parent.
        metadata: metadata of the object, a free-style dictionary.
    """

    ctype_class = CType

    def __init__(
        self,
        mimetype: str | None = None,
        ctype: CType | str = CType.Inline,
        content: Any = None,
        text: str = "",
        summary: str = "",
        parent: "None | str | Chunk" = None,
        child: "None | list | Chunk" = None,
        next: "None | str | Chunk" = None,
        prev: "None | str | Chunk" = None,
        origin: None | Origin = None,
        metadata: None | dict = None,
        history: None | list = None,
    ):
        self.id: str = uuid.uuid4().hex
        self.mimetype = mimetype
        self.ctype = ctype
        self._content = content
        self.text = text
        self.summary = summary
        self._parent = parent
        self._child = child
        self._next = next
        self._prev = prev
        self.origin = origin
        self.metadata = metadata or {}

        # internal use
        self._content_length: int | None = None
        self._history: list = history or []
        self._store: "BaseStore | None" = None

    def __str__(self):
        if isinstance(self.content, str):
            text = self.content
            if len(text) > 80:
                text = f"{text[:50]}... ({len(text[50:].split())} more words)"
            text = text.replace("\n", " ")
            content = f"content={text}"
        else:
            content = f"mimetype={self.mimetype}"

        return self.__class__.__name__ + f"(ctype={self._ctype}, {content})"

    def __repr__(self):
        return self.__str__()

    def walk(
        self, depth: int = 0, include_siblings: bool = True
    ) -> Generator[tuple[int, "Chunk"], None, None]:
        """Iterate depth and chunk in reading order, depth first, breadth second

        Args:
            depth: the current depth of the chunk in the tree
            include_siblings: if True, include the siblings of the current chunk

        Yields:
            tuple[int, Chunk]: the depth and the chunk object
        """
        # Yield current chunk first
        yield (depth, self)

        # Then yield all the children
        child = self.child
        while child:
            yield from child.walk(depth=depth + 1, include_siblings=False)
            child = child.next

        if include_siblings:
            # Finally yield all the siblings
            sibling = self.next
            while sibling:
                yield from sibling.walk(depth=depth, include_siblings=False)
                sibling = sibling.next

    @property
    def ctype(self):
        """Get the chunk type of the object"""
        return self._ctype

    @ctype.setter
    def ctype(self, value):
        """Set the chunk type of the object"""
        self._ctype = value

    @property
    def content(self):
        """Lazy loading of the content of the object"""
        if self._content is None and self._store is not None:
            self._content = self._store.fetch_content(self)
        return self._content

    @content.setter
    def content(self, value):
        """Set the content of the object"""
        self._content = value
        if isinstance(value, str):
            self._content_length = len(value)

    @property
    def content_length(self) -> int:
        """Get the length of the content of the object"""
        if self._content_length is not None:
            return self._content_length

        content_length = 0
        child = self.child
        while child := self.child:
            content_length += child.content_length
            child = child.next

        if isinstance(self.content, str):
            content_length += len(self.content)

        self._content_length = content_length
        return self._content_length

    @property
    def history(self) -> list:
        return self._history

    @history.setter
    def history(self, value: list):
        self._history = value

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, value: "BaseStore"):
        self._store = value

    @property
    def parent(self) -> "Chunk | None":
        """Get the parent object"""
        if isinstance(self._parent, Chunk):
            return self._parent
        if isinstance(self._parent, str):
            if not self._store:
                raise ValueError("Must provide `store` to load the parent")
            self._parent = self._store.get(self._parent)
            return self._parent

    @parent.setter
    def parent(self, value):
        if value is None or isinstance(value, (Chunk, str)):
            self._parent = value
        else:
            raise ValueError("`.parent` must be a Chunk or a id of a chunk")

    @property
    def parent_id(self) -> str | None:
        if isinstance(self._parent, str):
            return self._parent
        if isinstance(self._parent, Chunk):
            return self._parent.id

    @property
    def next(self) -> "Chunk | None":
        """Get the next object"""
        if isinstance(self._next, Chunk):
            return self._next
        if isinstance(self._next, str):
            if not self._store:
                raise ValueError("Must provide `store` to load the next")
            self._next = self._store.get(self._next)
            return self._next
        if self._next is not None:
            raise ValueError("`.next` must be a Chunk or a id of a chunk")

    @next.setter
    def next(self, value):
        if value is None or isinstance(value, (Chunk, str)):
            self._next = value
        else:
            raise ValueError("`.next` must be a Chunk or a id of a chunk")

    @property
    def next_id(self) -> str | None:
        if isinstance(self._next, str):
            return self._next
        if isinstance(self._next, Chunk):
            return self._next.id

    @property
    def prev(self) -> "Chunk | None":
        """Get the previous object"""
        if isinstance(self._prev, Chunk):
            return self._prev
        if isinstance(self._prev, str):
            if not self._store:
                raise ValueError("Must provide `store` to load the prev")
            self._prev = self._store.get(self._prev)
            return self._prev
        if self._prev is not None:
            raise ValueError("`.prev` must be a Chunk or a id of a chunk")

    @prev.setter
    def prev(self, value):
        if value is None or isinstance(value, (Chunk, str)):
            self._prev = value
        else:
            raise ValueError("`.prev` must be a Chunk or a id of a chunk")

    @property
    def prev_id(self) -> str | None:
        if isinstance(self._prev, str):
            return self._prev
        if isinstance(self._prev, Chunk):
            return self._prev.id

    @property
    def child(self) -> "Chunk | None":
        """Get the child object"""
        if isinstance(self._child, Chunk):
            return self._child
        if isinstance(self._child, str):
            if not self._store:
                raise ValueError("Must provide `store` to load the child")
            self._child = self._store.get(self._child)
            return self._child
        if self._child is not None:
            raise ValueError("`.child` must be a Chunk or a id of a chunk")

    @child.setter
    def child(self, value):
        if value is None or isinstance(value, (Chunk, str)):
            self._child = value
        else:
            raise ValueError("`.child` must be a Chunk or a id of a chunk")

    @property
    def last_child(self) -> "Chunk | None":
        """Get the last child object"""
        child = self.child
        while child and child.next:
            child = child.next
        return child

    @property
    def first_sibling(self) -> "Chunk | None":
        """Get the first sibling object"""
        sibling = self
        while sibling and sibling.prev:
            sibling = sibling.prev
        return sibling

    @property
    def last_sibling(self) -> "Chunk | None":
        """Get the last sibling object"""
        sibling = self
        while sibling and sibling.next:
            sibling = sibling.next
        return sibling

    @property
    def child_id(self) -> str | None:
        if isinstance(self._child, str):
            return self._child
        if isinstance(self._child, Chunk):
            return self._child.id

    def add_children(self, children: "list[Chunk] | Chunk"):
        """Add children to the current chunk"""
        if isinstance(children, Chunk):
            children = [children]

        if not children:
            return

        if self.child is None:
            self.child = children[0]
            children[0].parent = self

        for idx, child in enumerate(children[1:], start=1):
            child.parent = self
            children[idx - 1].next = child
            child.prev = children[idx - 1]

    @overload
    def render(self, format: Literal["plain", "markdown", "2d"] = "plain") -> str: ...

    @overload
    def render(self, format: Literal["multi"]) -> list[str | dict]: ...

    def render(
        self,
        format: Literal["plain", "markdown", "2d", "multi"] = "plain",
        **kwargs,
    ) -> str | list[str | dict]:
        """Select the executor type to render the object

        Args:
            format: the format of the output. Defaults to "plain".
                - plain: plain text, no formatting
                - markdown: plain text with markdown formatting
                - 2d: 2d string representation
                - multi: multi-modal representation, a list of dictionaries

        Returns:
            str if format is "plain", "markdown" or 2d; list of dict if format
                is "multi".
        """
        if format == "plain":
            current = self.text

            if current:
                return current

            current = self.content if isinstance(self.content, str) else ""
            child = self.child
            while child:
                rendered_child = child.render(format=format, **kwargs).strip()
                if not rendered_child:
                    child = child.next
                    continue

                if child.ctype == "inline":
                    separator = " "
                elif child.ctype == "list":
                    separator = "\n"
                elif child.ctype == "tablerow":
                    separator = "\n"
                elif not current:
                    separator = ""
                else:
                    separator = "\n\n"

                current += separator + rendered_child
                child = child.next

            return current
        elif format == "markdown":
            import textwrap

            if not kwargs:
                # header_level, list_level
                parent = self.parent
                kwargs = {"header_level": 0}
                while parent:
                    if parent.ctype == "header":
                        kwargs["header_level"] += 1
                    parent = parent.parent

            # Keep track of header to correctly render the header tag
            if self.ctype == "header":
                kwargs["header_level"] += 1

            if self.text:
                # It means the chunk is pre-rendered, and don't need to render again
                current = self.text
            else:
                current = self.content if isinstance(self.content, str) else ""
                if (
                    self.ctype == "header"
                    and kwargs.get("header_level", 0) > 0
                    and not current.startswith("#")
                ):
                    current = f"{'#' * kwargs['header_level']} {current}"
                child = self.child
                while child:
                    # Don't strip left whitespace because of list indentation
                    rendered_child = (
                        child.render(format=format, **kwargs).rstrip().lstrip("\n")
                    )
                    rendered_child = textwrap.dedent(rendered_child)
                    if not rendered_child:
                        child = child.next
                        continue
                    if child.ctype == "inline":
                        separator = " "
                    elif child.ctype == "list":
                        rendered_child = textwrap.indent(rendered_child, "  ")
                        separator = "\n"
                    elif child.ctype == "tablerow":
                        separator = "\n"
                    elif not current:
                        separator = ""
                    else:
                        separator = "\n\n"

                    current += separator + rendered_child
                    child = child.next

            if self.ctype == CType.Code:
                # Add code block
                current = current.strip()
                if current:
                    current = f"```\n{current}\n```"

            return current
        elif format == "multi":
            current_content = ""
            if self.content is not None:  # if None, just ignore and go to child chunk
                if isinstance(self.content, str):
                    current_content = self.content
                elif isinstance(self.content, bytes):
                    current_content = {
                        "mimetype": self.mimetype,
                        "text": self.summary or self.text,
                    }
                    try:
                        mime_manager = get_mime_manager()
                        obj = mime_manager.to_python(self)
                        if obj is None:
                            raise ValueError("Cannot convert to python object")
                        current_content["content"] = obj
                        current_content["processed"] = True
                    except Exception as e:
                        current_content["content"] = self.content
                        current_content["processed"] = False
                        logger.warning(
                            f"Cannot convert content to python object: {e}\n"
                            f"Mimetype: {self.mimetype}, Id: {self.id}"
                        )

            mixed_list = [current_content]
            child = self.child
            while child:
                child_content: list = child.render(format=format, **kwargs)
                if child.ctype == CType.Inline:
                    separator = " "
                elif child.ctype == CType.List:
                    separator = "\n"
                elif not current_content:
                    separator = ""
                else:
                    separator = "\n\n"

                mixed_list.append(separator)
                mixed_list.extend(child_content)
                child = child.next

            if not mixed_list:
                return []

            # Collapse consecutive strings into one
            result = []
            current_strings = []
            for item in mixed_list:
                if isinstance(item, str):
                    current_strings.append(item)
                else:
                    if current_strings:
                        result.append("".join(current_strings))
                        current_strings = []
                    result.append(item)

            if current_strings:
                result.append("".join(current_strings))

            return result
        else:
            raise NotImplementedError(
                f"Render as `format={format}` is not yet supported"
            )

    def asdict(self, relation_as_chunk: bool = False):
        """Return dictionary representation of the chunk

        Args:
            relation_as_chunk: if True, then treat the parent, child, next, prev as
                chunk, otherwise treat as id
        """
        if relation_as_chunk:
            return {
                "id": self.id,
                "mimetype": self.mimetype,
                "ctype": self.ctype,
                "content": self.content,
                "text": self.text,
                "parent": self.parent,
                "child": self.child,
                "next": self.next,
                "prev": self.prev,
                "origin": self.origin.asdict() if self.origin else None,
                "metadata": self.metadata,
                "history": self._history,
            }

        return {
            "id": self.id,
            "mimetype": self.mimetype,
            "ctype": self.ctype,
            "content": self.content,
            "text": self.text,
            "parent": self.parent_id,
            "child": self.child_id,
            "next": self.next_id,
            "prev": self.prev_id,
            "origin": self.origin.asdict() if self.origin else None,
            "metadata": self.metadata,
            "history": self._history,
        }

    def save(self, relations: bool = True):
        """Save the chunk into the directory

        Args:
            relations: if True, save the relations (parent, child, next, prev) as
                well. If False, only save the chunk itself.
        """
        if self._store is None:
            raise ValueError("Must provide `store` to save the chunk")
        self._store.save(self)

        if relations:
            for _, child in self.walk():
                self._store.save(child)

    def merge(self, chunk: "Chunk"):
        """Merge the content, metadata, and child of other chunk to this chunk

        Args:
            chunk: the other chunk to merge with this chunk
        """
        if self.mimetype != chunk.mimetype:
            raise ValueError("Cannot merge chunk with different mimetype")

        # Add the content
        self.content += chunk.content
        self.text += chunk.text
        self.summary += chunk.summary

        # Add the metadata
        if self.metadata is None and chunk.metadata is not None:
            self.metadata = chunk.metadata
        elif self.metadata is not None and chunk.metadata is not None:
            for key, value in chunk.metadata.items():
                if key in self.metadata:
                    self.metadata[key] += value
                else:
                    self.metadata[key] = value

        # Child
        our_last_child = self.last_child
        their_child = chunk.child
        if our_last_child and their_child:
            our_last_child.next = their_child
            their_child.prev = our_last_child
            their_child.parent = self
        elif their_child:
            self.child = their_child
            their_child.parent = self

    def clean(self, unwrap_single_child: bool = True):
        """Clean the chunk tree recursivly

        Args:
            unwrap_single_child: if True, if the parent chunk has empty content, and
                has a single child chunk, then the child chunk will replace the
                parent chunk in the hierarchy
        """
        child = self.child
        while child:
            child.clean(unwrap_single_child=unwrap_single_child)
            child = child.next

        if unwrap_single_child:
            # Assume the child content information if it is the single child, and
            # our content is empty
            if (
                not self.content  # doesn't have content
                and isinstance(self.child, Chunk)  # has child
                and self.child.next is None  # has only one child
            ):
                # Get mimetype, ctype and content
                self.mimetype = self.child.mimetype
                if self.child.ctype != CType.Inline:
                    self.ctype = self.child.ctype
                self.content = self.child.content

                # Combine metadata
                if self.metadata is not None:
                    if self.child.metadata is not None:
                        self.metadata.update(self.child.metadata)
                else:
                    self.metadata = self.child.metadata

                # Remove child
                self.child = self.child.child

    def apply(self, fn: Callable[["Chunk", int], None], depth: int = 0):
        """Apply a function to the chunk and all its children"""
        fn(self, depth)
        child = self.child
        while child:
            child.apply(fn, depth=depth + 1)
            child = child.next

    def print_graph(self, ctype: str | None | list = None):
        if isinstance(ctype, str):
            ctype = [ctype]

        def print_node(node, depth=0):
            if not ctype or node.ctype in ctype:
                print("    " * depth, node)

        self.apply(print_node)

    def get_ids(self) -> list[str]:
        """Get all the ids of the chunk and its children"""
        ids = [self.id]
        child = self.child
        while child:
            ids.extend(child.get_ids())
            child = child.next
        return ids

    def find(self, id: str | None = None, ctype: str | None = None) -> "Chunk | None":
        """Find the chunk by id or ctype

        Args:
            id: the id of the chunk to find
            ctype: the ctype of the chunk to find
        """
        if id is not None and self.id == id:
            return self

        if ctype is not None and self.ctype == ctype:
            return self

        child = self.child
        while child:
            found = child.find(id=id, ctype=ctype)
            if found:
                return found
            child = child.next

        return None

    def find_all(self, ctype: str | None = None) -> list["Chunk"]:
        """Find all the chunks by ctype

        Args:
            ctype: the ctype of the chunk to find
        """
        result = []
        if ctype is not None and self.ctype == ctype:
            result.append(self)

        child = self.child
        while child:
            result.extend(child.find_all(ctype=ctype))
            child = child.next

        return result

    def clone(self, no_relation: bool = False, **kwargs) -> "Chunk":
        """Create a deepcopy, replace infor with what supplied inside **kwargs"""

        d = self.asdict(relation_as_chunk=True)
        for key in kwargs.keys():
            if key not in d:
                raise ValueError(f"Invalid key: {key}")

        if no_relation:
            d.pop("parent")
            d.pop("child")
            d.pop("next")
            d.pop("prev")

        d.pop("id")
        d = deepcopy(d)
        d.update(kwargs)
        ch = Chunk(**d)

        if self.store:
            ch.store = self.store

        return ch


class ChunkGroup:
    """An interface for a group of related chunk"""

    def __init__(self, chunks: list | None = None, root: Chunk | None = None):
        self._roots: dict[str, Chunk] = {}
        self._chunks: dict[str | None, list] = {}

        self._root_id = None
        if root is not None:
            self._roots[root.id] = root
            self._root_id = root.id

        if chunks is not None or self._root_id is not None:
            self._chunks[self._root_id] = chunks or []

        self._store: "BaseStore | None" = None

    @property
    def store(self):
        return self._store

    @property
    def groups(self):
        return self._chunks

    def __bool__(self):
        return bool(len(self))

    def __getitem__(self, idx):
        for chunks in self._chunks.values():
            if idx < len(chunks):
                return chunks[idx]
            idx -= len(chunks)

        raise IndexError("Index out of range")

    def __iter__(self):
        for chunks in self._chunks.values():
            yield from chunks

    def __len__(self):
        count = sum(len(chunks) for chunks in self._chunks.values())
        return count

    def append(self, chunk: Chunk):
        if len(self._chunks) > 1:
            raise ValueError(
                "Cannot append when ChunkGroup has multiple roots. "
                "Please specify the root to append: "
                "`.groups[root_id_str].append(chunk)`"
            )
        elif len(self._chunks) == 0:
            self._chunks[self._root_id] = []

        if self._root_id not in self._chunks:
            self._root_id = list(self._chunks.keys())[0]

        self._chunks[self._root_id].append(chunk)

    def extend(self, chunks: list[Chunk]):
        if len(self._chunks) > 1:
            raise ValueError(
                "Cannot extend when ChunkGroup has multiple roots. "
                "Please specify the root to extend: "
                "`.groups[root_id_str].extend(chunks)`"
            )
        elif len(self._chunks) == 0:
            self._chunks[self._root_id] = []

        if self._root_id not in self._chunks:
            self._root_id = list(self._chunks.keys())[0]

        self._chunks[self._root_id].extend(chunks)

    def iter_groups(self):
        for root_id, chunks in self._chunks.items():
            if root_id is None:
                yield root_id, chunks
            else:
                root_node = self._roots[root_id]
                yield root_node, chunks

    def add_group(self, group: "ChunkGroup"):
        """Add another ChunkGroup to the current chunk group"""
        for root, chunks in group.iter_groups():
            if isinstance(root, Chunk):
                self._roots[root.id] = root
                root_id = root.id
            else:
                root_id = None

            if root_id not in self._chunks:
                self._chunks[root_id] = []
            self._chunks[root_id].extend(chunks)

    def attach_store(self, store):
        self._store = store
        for chunks in self._chunks.values():
            for chunk in chunks:
                chunk.store = store


class BaseStore:
    """Base class for organizing and persisting chunk"""

    def __contains__(self, id: str) -> bool:
        """Check if the chunk exists in the store"""
        raise NotImplementedError

    def get(self, id: str) -> Chunk:
        """Get the chunk by id"""
        raise NotImplementedError

    def fetch_content(self, chunk: Chunk):
        """Fetch the content of the chunk"""
        raise NotImplementedError

    def save(self, chunk: Chunk):
        """Save the chunk to the store"""
        raise NotImplementedError

    def save_group(self, group: ChunkGroup):
        """Save the group to the store"""
        for root, chunks in group.iter_groups():
            if isinstance(root, Chunk):
                self.save(root)
            for chunk in chunks:
                self.save(chunk)

    def delete(self, chunk: Chunk):
        """Delete the chunk from the store"""
        raise NotImplementedError


class BaseOperation:
    """Almost all operations on Chunk should eventually subclass from this. This class
    defines the interface so that:
        - Operations can be used as tool in agentic workflow.
        - Common interface for chunk

    When subclassing this class:
        - The subclass **must** implement the `.run` method.
        - The subclass **must** call super().__init__() if it overrides the __init__
        method.
        - The subclass **might** implement the `.as_tool` method. If not implemented,
        the method will inspect the `.run` method's signature to get the necessary
        arguments, and inspect the `.run` method's docstring to get the description.
    """

    supported_mimetypes: list[str] = []
    _tool_desc: dict | None = None

    def __init__(self, *args, **kwargs):
        self._default_params = kwargs

    @classmethod
    def run(cls, chunk: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        raise NotImplementedError

    @classmethod
    def name(cls, **kwargs) -> str:
        """Return the name of the operation to keep track in history"""
        fn = cls.__name__
        return f"{fn}({', '.join([f'{k}={v}' for k, v in kwargs.items()])})"

    def __call__(self, *chunk: Chunk, **kwargs) -> ChunkGroup:
        if self._default_params:
            for key, value in self._default_params.items():
                kwargs.setdefault(key, value)
        return self.run(*chunk, **kwargs)

    @classmethod
    def as_tool(cls) -> dict:
        """Return the necessary parameters for the operation.

        If not subclassed, this method will inspect the `.run` method's signature.
        Any non-optional argument will be considered as a required parameter. Any
        non-Python built-in type will be ignored.

        The resulting dictionary will have the following keys:
            - name (str): name of the operation
            - description (str): description of the operation
            - params (dict): parameters for the operation, which will
        """
        if cls._tool_desc is not None:
            return cls._tool_desc

        signature = inspect.signature(cls.run)
        docstring = inspect.getdoc(cls.run) or ""

        # parse the description from the docstring from beginning to Args
        description = ""
        if docstring:
            parts = re.split(r"\n\s*Args:", docstring, 1)
            description = parts[0].strip()
            description = " ".join([line.strip() for line in description.split("\n")])

        # parse parameter descriptions from docstring
        param_descriptions = {}
        if len(docstring.split("Args:")) > 1:
            args_section = docstring.split("Args:")[1]
            param_pattern = re.compile(
                r"\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*?)(?=\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:|$)",  # noqa: E501
                re.DOTALL,
            )
            matches = param_pattern.findall(args_section)

            for param_name, param_desc in matches:
                clean_desc = re.sub(r"\s+", " ", param_desc.strip())
                param_descriptions[param_name] = clean_desc

        # get type hints
        type_hints = get_type_hints(cls.run)

        # build parameters dictionary
        parameters = {}
        for name, param in signature.parameters.items():
            # skip *args, **kwargs, chunk param, and parameters without type annotations
            if name == "kwargs" or name == "chunk" or name not in type_hints:
                continue

            # skip non-builtin types
            type_anno = type_hints.get(name)
            type_name = getattr(type_anno, "__name__", str(type_anno))
            # if not hasattr(builtins, type_name):
            #     continue

            param_info = {
                "type": type_name,
                "required": param.default is param.empty,
            }

            # add default value if available
            if param.default is not param.empty:
                param_info["default"] = param.default

            # add description if available
            if name in param_descriptions:
                param_info["description"] = param_descriptions[name]

            parameters[name] = param_info

        return {
            "name": cls.__qualname__,
            "description": description,
            "parameters": parameters,
        }

    @classmethod
    def py_dependency(cls) -> list[str]:
        """Return the Python dependencies"""
        return []

    def default(self, **kwargs):
        """Update the default parameters for the operation"""
        self._default_params.update(kwargs)


class OperationManager:
    """Map mimetype to suitable operation"""

    def __init__(self, executors: dict | None = None, refiners: dict | None = None):
        # key: mimetype, value: list of supported operations for the mimetype
        self._executors = executors or defaultdict(list)
        self._refiners = refiners or defaultdict(list)

    def __enter__(self):
        _managers.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _managers.pop()

    def get_executor(self, mimetype): ...

    def add_executor(self, mimetype, operation): ...

    def update_executor(self, mimetype, operation): ...

    def get_refiner(self, mimetype): ...

    def add_refiner(self, mimetype, operation): ...

    def update_refiner(self, mimetype, operation): ...

    def as_tools(self, *mimetype: str) -> list[dict]:
        """Export all executors and refiners as tools"""
        ...

    def executor_as_tools(self, mimetype: str) -> list[dict]:
        """Export all executors as tools"""
        ...

    def refiner_as_tools(self, mimetype: str) -> list[dict]:
        """Export all refiners as tools"""
        ...

    @classmethod
    def from_default(cls) -> "OperationManager":
        """Construct an operation manager with default executors and refiners"""
        raise NotImplementedError

    def save(self, path):
        """Save all operations to a directory"""
        ...

    @classmethod
    def load(cls, path):
        """Load all operations from a directory"""
        ...


_managers = deque()


def get_manager():
    return _managers[-1]

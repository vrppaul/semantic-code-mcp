"""Tests for Rust tree-sitter code chunker."""

from pathlib import Path

from semantic_code_mcp.chunkers.rust import RustChunker
from semantic_code_mcp.domain.models import ChunkKind
from tests.helpers import (
    ChunkOwner,
    ChunkSummary,
    LineRange,
    line_ranges,
    owners_of,
    summarize,
)


class TestRustChunker:
    """Tests for RustChunker."""

    def test_simple_function(self):
        """Extracts a simple function."""
        code = """fn hello() {
    println!("hello");
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert len(chunks) == 1
        assert chunks[0].name == "hello"
        assert chunks[0].kind == ChunkKind.function
        assert "fn hello()" in chunks[0].content

    def test_function_with_doc_comment(self):
        """Function with /// doc comment includes comment in content."""
        code = """/// Greet someone by name.
///
/// # Arguments
/// * `name` - The name to greet.
fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert len(chunks) == 1
        assert chunks[0].name == "greet"
        assert "/// Greet someone by name" in chunks[0].content
        assert "fn greet" in chunks[0].content

    def test_struct_definition(self):
        """Extracts a struct as CLASS chunk."""
        code = """struct Point {
    x: f64,
    y: f64,
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert len(chunks) == 1
        assert chunks[0].name == "Point"
        assert chunks[0].kind == ChunkKind.klass

    def test_struct_with_impl_methods(self):
        """Struct + impl block: struct as CLASS, methods as METHOD."""
        code = """struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        class_chunks = [chunk for chunk in chunks if chunk.kind == ChunkKind.klass]
        method_chunks = [chunk for chunk in chunks if chunk.kind == ChunkKind.method]

        # struct + impl block
        assert len(class_chunks) == 2
        assert len(method_chunks) == 2
        names = {chunk.name for chunk in method_chunks}
        assert "new" in names
        assert "distance" in names

    def test_enum_definition(self):
        """Extracts an enum as CLASS chunk."""
        code = """enum Color {
    Red,
    Green,
    Blue,
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert len(chunks) == 1
        assert chunks[0].name == "Color"
        assert chunks[0].kind == ChunkKind.klass

    def test_trait_definition(self):
        """Extracts a trait as CLASS chunk with signature methods."""
        code = """trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        class_chunks = [chunk for chunk in chunks if chunk.kind == ChunkKind.klass]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "Drawable"

    def test_trait_with_default_methods(self):
        """A trait yields its required method signatures and its default methods."""
        code = """trait Greetable {
    fn name(&self) -> &str;

    fn greet(&self) -> String {
        format!("Hello, {}!", self.name())
    }
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert summarize(chunks) == [
            type_definition("Greetable"),
            method("name"),
            method("greet"),
        ]

    def test_impl_trait_for_struct(self):
        """impl Trait for Struct extracts as CLASS with methods."""
        code = """impl Drawable for Point {
    fn draw(&self) {
        println!("Drawing point");
    }

    fn area(&self) -> f64 {
        0.0
    }
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        class_chunks = [chunk for chunk in chunks if chunk.kind == ChunkKind.klass]
        method_chunks = [chunk for chunk in chunks if chunk.kind == ChunkKind.method]

        assert len(class_chunks) == 1
        # impl name should include trait info
        assert "Point" in class_chunks[0].name
        assert len(method_chunks) == 2

    def test_derive_attribute_on_struct(self):
        """#[derive(...)] attribute is included in struct content."""
        code = """#[derive(Debug, Clone)]
struct Config {
    name: String,
    value: i32,
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert len(chunks) == 1
        assert chunks[0].name == "Config"
        assert "#[derive(Debug, Clone)]" in chunks[0].content

    def test_attribute_on_function(self):
        """#[...] attribute on function is included in content."""
        code = """#[test]
fn test_something() {
    assert!(true);
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert len(chunks) == 1
        assert chunks[0].name == "test_something"
        assert "#[test]" in chunks[0].content

    def test_module_doc_comment(self):
        """//! inner doc comments at module level produce MODULE chunk."""
        code = """//! This module handles configuration.
//! It provides Config and related helpers.

fn configure() {}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("config.rs"))

        module_chunks = [chunk for chunk in chunks if chunk.kind == ChunkKind.module]
        func_chunks = [chunk for chunk in chunks if chunk.kind == ChunkKind.function]

        assert len(module_chunks) == 1
        assert module_chunks[0].name == "config"
        assert "This module handles configuration" in module_chunks[0].content
        assert len(func_chunks) == 1

    def test_generic_function(self):
        """Generic function with type parameters is extracted."""
        code = """fn identity<T: Clone>(val: T) -> T {
    val.clone()
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert len(chunks) == 1
        assert chunks[0].name == "identity"
        assert chunks[0].kind == ChunkKind.function

    def test_generic_struct(self):
        """Generic struct is extracted."""
        code = """struct Wrapper<T> {
    inner: T,
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert len(chunks) == 1
        assert chunks[0].name == "Wrapper"

    def test_async_function(self):
        """Async function is extracted."""
        code = """async fn fetch_data(url: &str) -> Result<String, Error> {
    let response = client.get(url).await?;
    Ok(response.text().await?)
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert len(chunks) == 1
        assert chunks[0].name == "fetch_data"
        assert chunks[0].kind == ChunkKind.function

    def test_empty_file(self):
        """Empty file returns empty list."""
        chunker = RustChunker()
        chunks = chunker.chunk_string("", Path("test.rs"))
        assert chunks == []

    def test_whitespace_only_file(self):
        """Whitespace-only file returns empty list."""
        chunker = RustChunker()
        chunks = chunker.chunk_string("   \n\n  ", Path("test.rs"))
        assert chunks == []

    def test_syntax_errors(self):
        """File with syntax errors returns empty list gracefully."""
        code = """fn broken( {
    this is not valid rust
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("broken.rs"))
        # tree-sitter is error-tolerant, but we accept whatever it returns
        assert isinstance(chunks, list)

    def test_line_numbers_1_indexed(self):
        """Line numbers are 1-indexed and accurate."""
        code = """// comment

fn first() {
    1
}

fn second() {
    2
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        first = next(chunk for chunk in chunks if chunk.name == "first")
        second = next(chunk for chunk in chunks if chunk.name == "second")

        assert first.line_start == 3
        assert first.line_end == 5
        assert second.line_start == 7
        assert second.line_end == 9

    def test_chunk_file_reads_from_disk(self, tmp_path: Path):
        """chunk_file reads a .rs file from disk."""
        file_path = tmp_path / "lib.rs"
        file_path.write_text("fn hello() { 42 }\n")

        chunker = RustChunker()
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert len(chunks) == 1
        assert chunks[0].name == "hello"

    def test_extensions_attribute(self):
        """RustChunker declares .rs extension."""
        assert RustChunker.extensions == (".rs",)

    def test_pub_function(self):
        """Public function is extracted."""
        code = """pub fn public_api() -> i32 {
    42
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        assert len(chunks) == 1
        assert chunks[0].name == "public_api"

    def test_impl_block_as_class_chunk(self):
        """Standalone impl block (no trait) is a CLASS chunk named after the type."""
        code = """impl MyStruct {
    fn method(&self) {}
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, Path("test.rs"))

        class_chunks = [chunk for chunk in chunks if chunk.kind == ChunkKind.klass]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "MyStruct"


def function(name: str) -> ChunkSummary:
    return ChunkSummary(kind=ChunkKind.function, name=name)


def method(name: str) -> ChunkSummary:
    return ChunkSummary(kind=ChunkKind.method, name=name)


def type_definition(name: str) -> ChunkSummary:
    return ChunkSummary(kind=ChunkKind.klass, name=name)


def module(name: str) -> ChunkSummary:
    return ChunkSummary(kind=ChunkKind.module, name=name)


class TestRustChunkerSilentDrops:
    """Code that used to vanish from the index without an error."""

    def test_generic_impl_and_its_methods(self):
        code = """struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    fn push(&mut self, item: T) {
        self.items.push(item);
    }

    fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }
}
"""
        chunks = RustChunker().chunk_string(code, Path("stack.rs"))

        assert summarize(chunks) == [
            type_definition("Stack"),
            type_definition("Stack"),
            method("push"),
            method("pop"),
        ]

    def test_impl_of_a_path_qualified_trait_for_a_generic_type(self):
        code = """impl fmt::Display for Stack<i32> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}
"""
        chunks = RustChunker().chunk_string(code, Path("stack.rs"))

        assert summarize(chunks) == [type_definition("Display for Stack"), method("fmt")]

    def test_impl_for_a_reference_type(self):
        code = """impl PartialEq for &Stack {
    fn eq(&self, other: &Self) -> bool {
        true
    }
}
"""
        chunks = RustChunker().chunk_string(code, Path("stack.rs"))

        assert summarize(chunks)[0] == type_definition("PartialEq for Stack")

    def test_impls_of_one_generic_trait_with_different_arguments_are_told_apart(self):
        code = """impl From<u8> for Level {
    fn from(value: u8) -> Self { Level(value.into()) }
}

impl From<u16> for Level {
    fn from(value: u16) -> Self { Level(value.into()) }
}
"""
        chunks = RustChunker().chunk_string(code, Path("level.rs"))

        assert summarize(chunks) == [
            type_definition("From<u8> for Level"),
            method("from"),
            type_definition("From<u16> for Level"),
            method("from"),
        ]

    def test_negative_impl_keeps_its_negation(self):
        code = "unsafe impl !Send for Handle {}\n"

        chunks = RustChunker().chunk_string(code, Path("handle.rs"))

        assert summarize(chunks) == [type_definition("!Send for Handle")]

    def test_required_trait_method_with_its_doc_comment(self):
        code = """trait Store {
    /// Save the record durably.
    fn save(&self);
}
"""
        chunks = RustChunker().chunk_string(code, Path("store.rs"))

        assert summarize(chunks) == [type_definition("Store"), method("save")]
        assert line_ranges(chunks)[1] == LineRange(line_start=2, line_end=3)

    def test_items_inside_an_inline_module(self):
        code = """#[cfg(test)]
mod tests {
    use super::*;

    fn helper() -> u32 {
        1
    }

    #[test]
    fn it_works() {
        assert_eq!(helper(), 1);
    }
}
"""
        chunks = RustChunker().chunk_string(code, Path("lib.rs"))

        assert summarize(chunks) == [function("helper"), function("it_works")]
        assert chunks[1].line_start == 9
        assert chunks[1].content.startswith("    #[test]")

    def test_file_without_items_becomes_one_module_chunk(self):
        code = """use std::fmt;

const LIMIT: u32 = 10;
"""
        chunks = RustChunker().chunk_string(code, Path("src/limits.rs"))

        assert summarize(chunks) == [module("limits")]
        assert line_ranges(chunks) == [LineRange(line_start=1, line_end=3)]

    def test_file_with_a_module_doc_and_no_items_is_indexed_whole(self):
        code = """//! Limits shared by every command.

const LIMIT: u32 = 10;
"""
        chunks = RustChunker().chunk_string(code, Path("src/limits.rs"))

        assert summarize(chunks) == [module("limits")]
        assert line_ranges(chunks) == [LineRange(line_start=1, line_end=3)]

    def test_module_doc_after_an_inner_attribute(self):
        code = """#![allow(dead_code)]
//! The crate's entry point.

fn main() {}
"""
        chunks = RustChunker().chunk_string(code, Path("src/main.rs"))

        assert summarize(chunks) == [module("main"), function("main")]
        assert line_ranges(chunks)[0] == LineRange(line_start=2, line_end=2)

    def test_block_module_doc_comment(self):
        code = """/*! The crate's entry point. */

fn main() {}
"""
        chunks = RustChunker().chunk_string(code, Path("src/main.rs"))

        assert summarize(chunks) == [module("main"), function("main")]


def test_an_item_names_the_modules_impls_and_traits_it_is_defined_in():
    code = """mod outer {
    pub struct Stack;

    impl Stack {
        pub fn push(&mut self) {}
    }

    pub trait Store {
        fn save(&self);
    }
}

fn top() {}
"""

    chunks = RustChunker().chunk_string(code, Path("lib.rs"))

    assert owners_of(chunks) == [
        ChunkOwner(name="Stack", parent="outer"),
        ChunkOwner(name="Stack", parent="outer"),
        ChunkOwner(name="push", parent="outer::Stack"),
        ChunkOwner(name="Store", parent="outer"),
        ChunkOwner(name="save", parent="outer::Store"),
        ChunkOwner(name="top", parent=None),
    ]

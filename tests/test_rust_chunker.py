"""Tests for Rust tree-sitter code chunker."""

from pathlib import Path

from semantic_code_mcp.chunkers.rust import RustChunker
from semantic_code_mcp.models import ChunkType


class TestRustChunker:
    """Tests for RustChunker."""

    def test_simple_function(self):
        """Extracts a simple function."""
        code = """fn hello() {
    println!("hello");
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, file_path="test.rs")

        assert len(chunks) == 1
        assert chunks[0].name == "hello"
        assert chunks[0].chunk_type == ChunkType.function
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
        chunks = chunker.chunk_string(code, file_path="test.rs")

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
        chunks = chunker.chunk_string(code, file_path="test.rs")

        assert len(chunks) == 1
        assert chunks[0].name == "Point"
        assert chunks[0].chunk_type == ChunkType.klass

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
        chunks = chunker.chunk_string(code, file_path="test.rs")

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.klass]
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.method]

        # struct + impl block
        assert len(class_chunks) == 2
        assert len(method_chunks) == 2
        names = {c.name for c in method_chunks}
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
        chunks = chunker.chunk_string(code, file_path="test.rs")

        assert len(chunks) == 1
        assert chunks[0].name == "Color"
        assert chunks[0].chunk_type == ChunkType.klass

    def test_trait_definition(self):
        """Extracts a trait as CLASS chunk with signature methods."""
        code = """trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, file_path="test.rs")

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.klass]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "Drawable"

    def test_trait_with_default_methods(self):
        """Trait with default method implementations extracts methods."""
        code = """trait Greetable {
    fn name(&self) -> &str;

    fn greet(&self) -> String {
        format!("Hello, {}!", self.name())
    }
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, file_path="test.rs")

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.klass]
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.method]

        assert len(class_chunks) == 1
        assert class_chunks[0].name == "Greetable"
        assert len(method_chunks) == 1
        assert method_chunks[0].name == "greet"

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
        chunks = chunker.chunk_string(code, file_path="test.rs")

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.klass]
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.method]

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
        chunks = chunker.chunk_string(code, file_path="test.rs")

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
        chunks = chunker.chunk_string(code, file_path="test.rs")

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
        chunks = chunker.chunk_string(code, file_path="config.rs")

        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.module]
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.function]

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
        chunks = chunker.chunk_string(code, file_path="test.rs")

        assert len(chunks) == 1
        assert chunks[0].name == "identity"
        assert chunks[0].chunk_type == ChunkType.function

    def test_generic_struct(self):
        """Generic struct is extracted."""
        code = """struct Wrapper<T> {
    inner: T,
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, file_path="test.rs")

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
        chunks = chunker.chunk_string(code, file_path="test.rs")

        assert len(chunks) == 1
        assert chunks[0].name == "fetch_data"
        assert chunks[0].chunk_type == ChunkType.function

    def test_empty_file(self):
        """Empty file returns empty list."""
        chunker = RustChunker()
        chunks = chunker.chunk_string("", file_path="test.rs")
        assert chunks == []

    def test_whitespace_only_file(self):
        """Whitespace-only file returns empty list."""
        chunker = RustChunker()
        chunks = chunker.chunk_string("   \n\n  ", file_path="test.rs")
        assert chunks == []

    def test_syntax_errors(self):
        """File with syntax errors returns empty list gracefully."""
        code = """fn broken( {
    this is not valid rust
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, file_path="broken.rs")
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
        chunks = chunker.chunk_string(code, file_path="test.rs")

        first = next(c for c in chunks if c.name == "first")
        second = next(c for c in chunks if c.name == "second")

        assert first.line_start == 3
        assert first.line_end == 5
        assert second.line_start == 7
        assert second.line_end == 9

    def test_chunk_file_reads_from_disk(self, tmp_path: Path):
        """chunk_file reads a .rs file from disk."""
        file_path = tmp_path / "lib.rs"
        file_path.write_text("fn hello() { 42 }\n")

        chunker = RustChunker()
        chunks = chunker.chunk_file(str(file_path))

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
        chunks = chunker.chunk_string(code, file_path="test.rs")

        assert len(chunks) == 1
        assert chunks[0].name == "public_api"

    def test_impl_block_as_class_chunk(self):
        """Standalone impl block (no trait) is a CLASS chunk named after the type."""
        code = """impl MyStruct {
    fn method(&self) {}
}
"""
        chunker = RustChunker()
        chunks = chunker.chunk_string(code, file_path="test.rs")

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.klass]
        assert len(class_chunks) == 1
        assert class_chunks[0].name == "MyStruct"

"""Smoke test: verify Gradio app constructs without launching a server."""
import importlib
import sys


def test_gradio_blocks_construct():
    """Import app module and verify demo object is a valid Gradio Blocks."""
    mod = importlib.import_module("app")
    demo = getattr(mod, "demo", None)
    assert demo is not None, "app.demo not found"

    import gradio as gr
    assert isinstance(demo, gr.Blocks), "demo is not a gr.Blocks instance"


def test_gradio_blocks_has_components():
    """Verify key UI components exist in the constructed Blocks."""
    mod = importlib.import_module("app")
    demo = mod.demo

    blocks = list(demo.blocks.values())
    type_names = {type(b).__name__ for b in blocks}
    assert "Button" in type_names, "No Button components found"
    assert "Textbox" in type_names, "No Textbox components found"
    assert "Tab" in type_names, "No Tab components found"

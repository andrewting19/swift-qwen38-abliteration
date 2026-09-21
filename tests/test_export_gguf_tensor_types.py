from scripts.export_gguf_tensor_types import exact_pattern, render_type_map


def test_exact_pattern_escapes_tensor_name() -> None:
    assert exact_pattern("blk.52.ffn_down.weight") == r"^blk\.52\.ffn_down\.weight$"


def test_render_type_map_omits_f32_and_preserves_quant_types() -> None:
    assert render_type_map(
        [
            ("output_norm.weight", "F32"),
            ("output.weight", "Q5_K"),
            ("blk.0.ffn_down.weight", "IQ3_XXS"),
        ]
    ) == (
        r"^output\.weight$=Q5_K" "\n"
        r"^blk\.0\.ffn_down\.weight$=IQ3_XXS" "\n"
    )

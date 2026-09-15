from scripts.prepare_xstest_safe import parse_safe_rows, stratified_order


def test_parse_safe_rows_excludes_unsafe_rows() -> None:
    payload = b"id,prompt,type,label\n1,Safe one,a,safe\n2,Unsafe one,a,unsafe\n"
    import scripts.prepare_xstest_safe as module

    previous = module.EXPECTED_SAFE_COUNT
    module.EXPECTED_SAFE_COUNT = 1
    try:
        rows = parse_safe_rows(payload)
    finally:
        module.EXPECTED_SAFE_COUNT = previous
    assert [row["id"] for row in rows] == ["1"]


def test_stratified_order_is_deterministic_and_complete() -> None:
    rows = [
        {"id": str(index), "prompt": "safe", "type": str(index % 3), "label": "safe"}
        for index in range(30)
    ]
    first = stratified_order([dict(row) for row in rows], 3819)
    second = stratified_order([dict(row) for row in rows], 3819)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert {row["id"] for row in first} == {row["id"] for row in rows}

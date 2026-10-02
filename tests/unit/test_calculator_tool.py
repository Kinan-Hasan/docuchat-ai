from app.rag.tools.calculator import calculator


def test_calculator_basic_arithmetic():
    assert calculator("2 + 2") == "4"
    assert calculator("10 * 4") == "40"
    assert calculator("10 / 4") == "2.5"


def test_calculator_rejects_unsafe_input():
    result = calculator("__import__('os').system('echo hacked')")
    assert result.startswith("error:")


def test_calculator_rejects_garbage():
    result = calculator("not a math expression")
    assert result.startswith("error:")

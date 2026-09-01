from pathlib import Path
import pytest
from src.validation import parse_utkface_filename

def test_parses_utkface_filename() -> None:
    record = parse_utkface_filename(Path("27_1_3_201701.jpg"))
    assert (record["age"], record["gender"], record["ethnicity"]) == (27, 1, 3)

@pytest.mark.parametrize("name", ["bad.jpg", "200_0_0_x.jpg", "30_4_0_x.jpg"])
def test_rejects_invalid_filename(name: str) -> None:
    with pytest.raises(ValueError):
        parse_utkface_filename(name)

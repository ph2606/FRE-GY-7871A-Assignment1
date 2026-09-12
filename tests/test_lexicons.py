from pathlib import Path
import pandas as pd
from src.lexicons import load_master_dictionary, lm_word_lists
from src.config import LM_CATEGORIES


def test_real_word_na_is_not_converted_to_missing(tmp_path):
    data = pd.DataFrame({"Word": ["NA", "LOSS", "LATE"], **{cat: [0, 2011, -2020] for cat in LM_CATEGORIES}})
    path = tmp_path / "dictionary.csv"
    data.to_csv(path, index=False)
    master = load_master_dictionary(path)
    assert master.Word.tolist() == ["NA", "LOSS", "LATE"]
    assert lm_word_lists(master)["Negative"] == {"LOSS"}
    assert lm_word_lists(master, include_removed=True)["Negative"] == {"LOSS", "LATE"}

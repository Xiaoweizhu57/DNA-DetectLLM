from transformers import AutoTokenizer


def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_id_1).vocab
            == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")

# def assert_tokenizer_consistency(model_id_1, model_id_2):
#     identical_tokenizers = (
#             AutoTokenizer.from_pretrained(model_id_1, use_fast=False).get_vocab()
#             == AutoTokenizer.from_pretrained(model_id_2, use_fast=False).get_vocab()
#     )
#     if not identical_tokenizers:
#         raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")

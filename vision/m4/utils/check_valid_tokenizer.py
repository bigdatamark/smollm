# [MODULE DOC] Sanity check for tokenizers used by m4: verifies required special tokens exist and are not
# normalized.
def check_valid_tokenizer(tokenizer) -> bool:
    # [FUNCTION DOC] Validates a tokenizer in place. Only acts when the lowercase class name contains
    # "idefics" or "mistral":
    # asserts "<image>", "<fake_token_around_image>", "<s>", "</s>", "<unk>" are in the vocab and that every
    # entry of added_tokens_decoder has normalized False. Raises AssertionError on failure.
    # KNOWN ISSUE: annotated to return bool but never returns anything (always None); other tokenizer classes
    # are silently not checked at all.
    """Check if the special tokens were correctly added to the tokenizer,
    and if they are not normalized.
    """
    tok_class = type(tokenizer).__name__.lower()
    if ("idefics" in tok_class) or ("mistral" in tok_class):
        assert "<image>" in tokenizer.get_vocab()
        assert "<fake_token_around_image>" in tokenizer.get_vocab()
        assert "<s>" in tokenizer.get_vocab()
        assert "</s>" in tokenizer.get_vocab()
        assert "<unk>" in tokenizer.get_vocab()

        for _, val in tokenizer.added_tokens_decoder.items():
            assert not val.normalized  # assert that normalized=False for all AddedToken

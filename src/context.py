"""Apply the same document and token limits to every retrieval method."""


def pack_context(documents, count_tokens, budget, top_k=5):
    """Keep whole ranked chunks within budget; never score unseen truncated text.

    Oversized chunks are skipped, allowing smaller lower-ranked chunks to fit.
    The returned list is exactly the evidence sent to the model and evaluator.
    """
    selected = []
    for document in documents:
        if len(selected) == top_k:
            break
        candidate = "\n\n".join(d.page_content for d in [*selected, document])
        if count_tokens(candidate) <= budget:
            selected.append(document)
    return selected

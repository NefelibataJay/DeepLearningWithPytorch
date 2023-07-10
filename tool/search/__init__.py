from tool.search import beam_search, greedy_search

REGISTER_SEARCH = {
    "ctc_greedy_search": greedy_search.ctc_greedy_search,
    "transformer_greedy_search": greedy_search.transformer_greedy_search,
}


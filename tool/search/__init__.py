from tool.search import beam_search, greedy_search

REGISTER_SEARCH = {
    "greedy_search": greedy_search.GreedySearch,
    "beam_search": beam_search.BeamSearch,
}


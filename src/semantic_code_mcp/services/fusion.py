"""Fusion: one ranking from what the vector channel and the keyword channel found.

Reciprocal Rank Fusion scores a chunk by the sum, over the channels that found it, of
1 / (RRF_K + its rank there). It reads only ranks, so neither channel's scores need a
scale the other's can be put on: full-text scores move with the corpus and with whether
the full-text index has taken a row in yet, cosine similarities with the model. A chunk
both channels find comes before one only a single channel ranks first.
"""

from semantic_code_mcp.domain.models import Candidate, SearchCandidates, SearchHit

# The constant of Reciprocal Rank Fusion, as its authors set it (Cormack, Clarke and
# Büttcher, 2009): large enough that one channel's first few hits do not drown the rest
RRF_K = 60
FIRST_RANK = 1


def fuse(candidates: SearchCandidates) -> list[SearchHit]:
    """Every chunk either channel found, once, best fused score first; chunks fused alike
    come closest to the query in meaning first."""
    vector_ranks = ranks_of(candidates.by_vector)
    keyword_ranks = ranks_of(candidates.by_keyword)
    found = {
        candidate.chunk.chunk_id: candidate
        for candidate in [*candidates.by_vector, *candidates.by_keyword]
    }
    hits = [
        hit_of(candidate, vector_ranks.get(chunk_id), keyword_ranks.get(chunk_id))
        for chunk_id, candidate in found.items()
    ]
    return sorted(hits, key=lambda hit: (hit.fused_score, hit.similarity), reverse=True)


def ranks_of(found: list[Candidate]) -> dict[str, int]:
    """Where a channel ranked each chunk it found, counted from one."""
    return {
        candidate.chunk.chunk_id: rank for rank, candidate in enumerate(found, start=FIRST_RANK)
    }


def hit_of(candidate: Candidate, vector_rank: int | None, keyword_rank: int | None) -> SearchHit:
    """A chunk with its ranks in each channel and the score they fuse to."""
    return SearchHit(
        chunk=candidate.chunk,
        similarity=candidate.similarity,
        vector_rank=vector_rank,
        keyword_rank=keyword_rank,
        fused_score=reciprocal_rank(vector_rank) + reciprocal_rank(keyword_rank),
    )


def reciprocal_rank(rank: int | None) -> float:
    """What a channel's rank adds to a chunk's fused score; nothing where it did not find it."""
    if rank is None:
        return 0.0
    return 1 / (RRF_K + rank)

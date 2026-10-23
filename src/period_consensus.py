"""Cross-feature selection shared by CLI and full-search uncertainty controls."""
import numpy as np

CONSENSUS_POLICY = "feature_support_then_mean_fit_score_v2"


def _cross_feature_consensus(feature_summary, relative_tolerance=0.05):
    """Select a period supported by distinct observables, not one tallest peak.

    Multi-harmonic fits naturally admit integer-multiple fundamentals. Clusters
    are ranked by distinct feature support, mean normalized fit score, then
    relative spread. Scores are regression improvement, not probabilities;
    observables from the same IQ are not independent evidence.
    Half/double aliases appended for reporting are excluded
    from the vote so they cannot count twice for the same measured peak.
    """

    entries = []
    for feature, item in feature_summary.items():
        for rank, candidate in enumerate(item["candidates"][:5]):
            if candidate["source"].endswith((":half", ":double")):
                continue
            entries.append(
                {
                    "feature": feature,
                    "period_s": float(candidate["period_s"]),
                    "score": float(candidate["score"]),
                    "rank": rank,
                }
            )
    clusters = []
    for entry in sorted(entries, key=lambda item: item["period_s"]):
        compatible = [
            cluster for cluster in clusters
            if abs(entry["period_s"] / cluster["center_s"] - 1.0) <= relative_tolerance
        ]
        if compatible:
            cluster = min(
                compatible,
                key=lambda item: abs(np.log(entry["period_s"] / item["center_s"])),
            )
            cluster["entries"].append(entry)
            cluster["center_s"] = float(np.median([x["period_s"] for x in cluster["entries"]]))
        else:
            clusters.append({"center_s": entry["period_s"], "entries": [entry]})
    ranked = []
    for cluster in clusters:
        by_feature = {}
        for entry in cluster["entries"]:
            current = by_feature.get(entry["feature"])
            if current is None or entry["rank"] < current["rank"]:
                by_feature[entry["feature"]] = entry
        evidence = list(by_feature.values())
        center = float(np.median([item["period_s"] for item in evidence]))
        spread = float(
            np.median([abs(item["period_s"] / center - 1.0) for item in evidence])
        )
        quality = float(np.mean([item["score"] for item in evidence]))
        ranked.append((len(evidence), quality, spread, center, evidence))
    if not ranked:
        raise ValueError("没有可用于跨特征共识的周期候选")
    support, _, spread, period_s, evidence = sorted(
        ranked, key=lambda item: (-item[0], -item[1], item[2], item[3])
    )[0]
    return period_s, support, spread, evidence


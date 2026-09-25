"""CPU-only diagnostic of existing DAR functions; no model or dataset inference."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_functions(relative, classes=False):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Exclude framework imports and registration. Function bodies are unchanged.
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            if module.startswith(("swift", "dar_pipeline_common")):
                continue
            nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.Assign)):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) for t in node.targets):
                continue
            nodes.append(node)
        elif classes and isinstance(node, ast.ClassDef):
            nodes.append(node)
    env = {"ORM": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


def main():
    r = load_functions("ms-swift/examples/train/grpo/plugin/dar_plugin.py", classes=True)
    c = load_functions("data_construction/08_dual_consistency_committee.py")
    # Synthetic scene premise: a dog remains outside a closed gate.
    grounded = "The dog stays outside the closed gate while the viewer awaits entry."
    false = "The dog walks inside the open gate while the viewer watches entry."
    assert r["_word_count"](grounded) == r["_word_count"](false) == 12
    def completion(reason):
        return json.dumps({"segments": [{"start_time": 0.0, "end_time": 5.0,
                            "emotion": "Interest", "reason": reason}]})
    weights = [0.10, 0.25, 0.25, 0.25, 0.15]
    names = ["DARStructuralReward", "DARSegmentCountReward", "DARTemporalSegmentationReward",
             "DAREmotionAccuracyReward", "DARReasoningQualityReward"]
    solution = completion(grounded)
    scores = {}
    for label, reason in [("grounded_12_words", grounded), ("contradicted_12_words", false),
                          ("unrelated_120_words", " ".join(["banana"] * 120))]:
        vector = [r[n]()([completion(reason)], solution=[solution])[0] for n in names]
        scores[label] = {"reward_vector": vector, "weighted_total": sum(a*b for a,b in zip(weights,vector))}
    assert scores["grounded_12_words"] == scores["contradicted_12_words"]
    assert scores["unrelated_120_words"]["weighted_total"] == 1.0
    results = {
        "scope": "Synthetic objective checks, not observed model behavior or real-video evaluation",
        "reward_order": names,
        "scores": scores,
        "single_judge_pass": c["combine_segment_feedback"](0, {0: {"average_score": 5}}, {}, 3.5),
        "judge_disagreement_pass": c["combine_segment_feedback"](0, {0: {"average_score": 2}}, {0: {"average_score": 5}}, 3.5),
        "zero_grounding_compensated_mean": sum([0, 5, 5, 5, 5]) / 5,
    }
    assert results["single_judge_pass"]["passed"]
    assert results["judge_disagreement_pass"]["passed"]
    target = Path(__file__).with_name("probe-results.json")
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

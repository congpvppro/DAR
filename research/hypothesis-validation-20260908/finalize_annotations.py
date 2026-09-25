"""Record the primary researcher's full factual reading and audit the failed critic.

All 176 factual answers were read in blind-ID order before model keys were opened.
This does not certify object identities, sounds, or psychological explanations:
the factual endpoint is the change boolean and consistency of change evidence.
"""
from collections import Counter
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
DATA = OUT / 'timing-corrected'


def main():
    rows = [json.loads(x) for x in (DATA / 'factual-blind.jsonl').read_text().splitlines()]
    assert len(rows) == 176
    # Exact ID decisions after full reading; no model or prompt selection.
    wrong = {
        'F001': 'Claims a subtle posture shift despite the no-change boolean.',
        'F008': 'Explicit slide followed by crash in identical frames.',
        'F043': 'Claims added text/logo and says the change answer is true.',
        'F074': 'Claims a gently pulsing heart despite fixed identical frames.',
        'F094': 'Explicit rhythmic repetitive movement of a leg.',
        'F149': 'Claims a slight camera wobble despite identical frames.',
        'F150': 'Claims foreground position shifts despite identical frames.',
        'F153': 'Explicit repeated drawer opening and closing.',
        'F176': 'Explicit before/after facial expression shift.',
    }
    ambiguous = {
        'F017': 'Unwrapping may describe a depicted pose or assert actual motion.',
        'F035': 'An anticipated future fail is phrased as when it finally occurs.',
        'F045': 'Claims minute movements but also says they cannot be clearly observed.',
        'F051': 'Riding a scooter may be a depicted pose or actual motion.',
        'F118': 'Dribbling may describe the depicted action or assert actual motion.',
        'F125': 'Focused writing may be the depicted pose or actual motion.',
        'F170': 'Spraying water may describe a static pose or assert actual motion.',
    }
    results = []
    for row in rows:
        ident = row['blind_id']
        parsed = json.loads(row['raw_output'])
        assert type(parsed['visible_change']) is bool
        if row['kind'] == 'step':
            if ident == 'F159':
                assert parsed['visible_change'] is True
                judgment = 'correct'
                note = ('Correct gray-to-white change at endpoint level; grayish-green tint, '
                        'first-frame timing and psychological embellishments are not certified.')
            else:
                assert parsed['visible_change'] is False
                judgment = 'incorrect'
                note = 'False no-change boolean on a verified gray-to-white input.'
                if ident == 'F119':
                    note += ' Text mentions a transition, but contradicts the boolean and misstates initial color.'
        elif ident in wrong:
            judgment, note = 'incorrect', wrong[ident]
        elif ident in ambiguous:
            judgment, note = 'ambiguous', ambiguous[ident]
        else:
            assert parsed['visible_change'] is False
            judgment = 'correct'
            note = ('No-change boolean with no explicit conflicting visible-change claim. '
                    'Static object identity, exact frame counts and affective commentary are outside this endpoint.')
        results.append(dict(blind_id=ident, judgment=judgment, note=note,
                            raw_sha256=row['raw_sha256'], coder='primary_researcher_AI'))
    path = DATA / 'factual-coding.jsonl'
    with path.open('x') as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    primary = {r['blind_id']: r for r in map(json.loads, (DATA/'coding-primary.jsonl').read_text().splitlines())}
    reviews = list(map(json.loads, (DATA/'coding-review-local.jsonl').read_text().splitlines()))
    assert len(reviews) == len(primary) == 304
    audit = dict(
        status='failed_quality_gate; not accepted as independent coding validation',
        attempted=304, syntactically_valid_and_quote_checked=sum(r['review'] is not None for r in reviews),
        invalid=sum(r['review'] is None for r in reviews),
        valid_label_counts=dict(Counter(r['review']['label'] for r in reviews if r['review'])),
        primary_label_counts=dict(Counter(r['label'] for r in primary.values())),
        primary_positive_and_valid_reviewer_positive=sum(primary[r['blind_id']]['label']=='confirmed' and r['review'] is not None and r['review']['label']=='confirmed' for r in reviews),
        invalid_example_ids=['B001'],
        semantic_failure_examples=['B027', 'B054', 'B265', 'B272', 'B275', 'B298'],
        explanation=('The critic copies rubric/input labels as evidence, treats standing still as impossible, '
                     'and treats emotion change as proof of visual change. Exact-quote validation alone is insufficient. '
                     'All primary annotations remain provisional single-AI coding. Do not use a label intersection '
                     'to fabricate successful independent review or a confirmatory hallucination-rate test.'),
    )
    (OUT/'coding-review-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(factual_counts=Counter(r['judgment'] for r in results), review=audit), indent=2))


if __name__ == '__main__':
    main()

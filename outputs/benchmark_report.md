# GraphRAG Benchmark Report

## Summary

- Total questions: 20
- Flat RAG accuracy: 17/20 = 85.0%
- GraphRAG accuracy: 18/20 = 90.0%
- GraphRAG-only wins: 3

## Accuracy By Question Type

| Question Type | Flat RAG | GraphRAG |
|---|---:|---:|
| adversarial | 40% | 100% |
| comparison | 100% | 60% |
| one-hop | 100% | 100% |
| two-hop | 100% | 100% |

## Question Results

| ID | Type | Question | Expected | Flat | Graph |
|---|---|---|---|---|---|
| q001 | one-hop | Who founded Microsoft? | Bill Gates and Paul Allen | OK | OK |
| q002 | one-hop | Where is OpenAI headquartered? | San Francisco | OK | OK |
| q003 | one-hop | Which company acquired Instagram? | Meta | OK | OK |
| q004 | one-hop | What product did Anthropic create? | Claude | OK | OK |
| q005 | one-hop | Which company developed AlphaFold? | DeepMind | OK | OK |
| q006 | two-hop | Which company invested in OpenAI and provides cloud infrastructure through Azure? | Microsoft | OK | OK |
| q007 | two-hop | Which parent company owns the company that acquired DeepMind? | Alphabet | OK | OK |
| q008 | two-hop | Which Microsoft subsidiary is a software development platform headquartered in San Francisco? | GitHub | OK | OK |
| q009 | two-hop | Which AI company is connected to Tesla through Elon Musk? | OpenAI | OK | OK |
| q010 | two-hop | Which company invested in Anthropic and provides cloud infrastructure through Amazon Web Services? | Amazon | OK | OK |
| q011 | comparison | Compare OpenAI and Anthropic: what AI assistant product did each create? | OpenAI created ChatGPT, while Anthropic created Claude. | OK | OK |
| q012 | comparison | Compare Microsoft, Google, and Amazon: what cloud platform is associated with each? | Microsoft has Azure, Google has Google Cloud, and Amazon has Amazon Web Services. | OK | OK |
| q013 | comparison | Compare Meta's acquisitions of Instagram and WhatsApp: which company acquired both and in what years? | Meta acquired Instagram in 2012 and WhatsApp in 2014. | OK | MISS |
| q014 | comparison | Compare Google and DeepMind: which one acquired the other, and which one developed AlphaGo? | Google acquired DeepMind, and DeepMind developed AlphaGo. | OK | OK |
| q015 | comparison | Compare Apple and Tesla: what major hardware products are associated with each? | Apple is associated with iPhone, iPad, Mac, and Apple Vision Pro, while Tesla is associated with Model S, Model 3, Model X, and Model Y. | OK | MISS |
| q016 | adversarial | Did Google acquire WhatsApp? | No. Meta acquired WhatsApp in 2014. | MISS | OK |
| q017 | adversarial | Is GitHub a subsidiary of Google? | No. GitHub is a subsidiary of Microsoft. | MISS | OK |
| q018 | adversarial | Did Anthropic create ChatGPT? | No. OpenAI created ChatGPT, while Anthropic created Claude. | OK | OK |
| q019 | adversarial | Was Tesla founded by Elon Musk? | No. Tesla was founded by Martin Eberhard and Marc Tarpenning; Elon Musk invested later and became CEO. | OK | OK |
| q020 | adversarial | Does Amazon Web Services belong to Microsoft? | No. Amazon Web Services was created by Amazon; Microsoft created Azure. | MISS | OK |

## Where GraphRAG Improves

### q016: Did Google acquire WhatsApp?

- Expected: No. Meta acquired WhatsApp in 2014.
- Flat RAG: The answer is not supported by the retrieved chunks.
- GraphRAG: No, Google did not acquire WhatsApp. Instead, WhatsApp was acquired by Meta. This is supported by the relationship "Meta --ACQUIRED--> WhatsApp."

### q017: Is GitHub a subsidiary of Google?

- Expected: No. GitHub is a subsidiary of Microsoft.
- Flat RAG: The answer is not supported by the retrieved chunks.
- GraphRAG: No, GitHub is not a subsidiary of Google. GitHub is a subsidiary of Microsoft, as indicated by the relationship "GitHub --SUBSIDIARY_OF--> Microsoft."

### q020: Does Amazon Web Services belong to Microsoft?

- Expected: No. Amazon Web Services was created by Amazon; Microsoft created Azure.
- Flat RAG: The answer is not supported by the retrieved chunks.
- GraphRAG: No, Amazon Web Services does not belong to Microsoft. Amazon Web Services is created by Amazon, which is a separate entity from Microsoft. The key supporting relationships are that Amazon created Amazon Web Services and Microsoft has its own product, Azure, which competes with Amazon Web Services.

## Notes

- `OK` and `MISS` are produced by transparent string-based scoring rules in `src/evaluate.py`.
- CSV remains the raw machine-readable output; this Markdown file is intended for review and lab reporting.

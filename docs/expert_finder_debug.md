## Expert Finder Debug & Troubleshooting

Phase 1 (research-only) expert lookup feature.

### Data Flow
1. User sends message -> `/api/chat`.
2. Backend runs `detect_expert_intent()` BEFORE LLM call; logs `[expert_intent]` if matched.
3. JSON reply includes `expert_intent` object.
4. Front-end `sendMessage` consumes `expert_intent` and (if present) exposes the Expert Finder card.
5. User consents to location -> geolocation OR ZIP/city geocode (Nominatim).
6. `/api/expert/search` queries Overpass -> results -> render.

### Quick Test
Message: `Can you find me a cardiologist?`

Expected:
- Terminal log: `[expert_intent] session=... label=cardiologist taxonomy=207RC0000X`
- Network: `/api/chat` response JSON has `expert_intent.label == "cardiologist"`.
- UI: Expert Finder card appears with consent buttons.

### If Card Missing
| Order | Check | Tool | Expected |
|-------|-------|------|----------|
| 1 | Backend intent log | Server console | `[expert_intent] ...` line |
| 2 | API response | DevTools > Network > api/chat | JSON includes `expert_intent` |
| 3 | JS error? | DevTools Console | No runtime errors referencing `__applyExpertIntent` |
| 4 | Deferred intent | Console: `window.__pendingExpertIntent` | `undefined` (already applied) |
| 5 | CSS state | Inspect `#expertFinderCard` | `display: block;` |

### Python REPL Intent Sanity
```python
from src.services.expert_finder import detect_expert_intent
print(detect_expert_intent('Can you find me a cardiologist?'))
print(detect_expert_intent('Need a good dermatologist nearby'))
```

### Overpass Query Verification
Manual check (replace coords):
```bash
curl -X POST https://overpass-api.de/api/interpreter \
  --data 'data=[out:json][timeout:15];(node["healthcare:speciality"="cardiology"](around:10000,37.7749,-122.4194););out center 20;' | head
```

If responses are empty repeatedly, you may be rate-limited.

### Common Causes & Fixes
| Symptom | Cause | Fix |
|---------|-------|-----|
| Card never shows | Intent detection patch missing | Re-apply server patch; ensure `expert_intent` key in JSON |
| Card flashes then hides | JS error stops `__applyExpertIntent` | Check console, clear cache, hard reload |
| Always sample result | No OSM matches / Overpass limit | Try different specialty or location; widen radius |
| Geolocation denied | User blocked permission | Use ZIP / City fallback |

### Logging Enhancement (Optional)
Add structured JSON logs in `api_chat`:
```python
import json, time
if intent:
    print(json.dumps({'ts': time.time(), 'event':'expert_intent', **intent}), flush=True)
```

### Future Roadmap
- NPI registry verification & confidence scoring.
- Telehealth fallback (<3 results).
- Outreach drafting with PHI redaction.
- User preference: auto vs manual research.
- Metrics dashboard (intent rate, search latency, success rate).

---
Update this doc as features progress.
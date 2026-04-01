# SkyNet Concept

Given the technological advancements in recent years with the creation of word2vec, tokenization and compute neurons, AI agents, OpenClaw, Claude code, it's become possible to create an analogical copy of what we know the human brain actually does in order to form a personality or experiences and mirror it in an analogical way in a digital environment comprised out of 'regions' that do specific tasks when there's a 'sensory' input incoming. The concept is for a solution that will be able to learn from past experiences and specialize in a narrow pathway 'career'. 


## Sensory Input as we know it

Sensory input in the human brain goes through specific stages when a new experience is 'recorded'. The brain workflow looks something like this:

------------------------------------------------------------------------
<pre>
Sensory Input
     ↓
Thalamus receives it, decides urgency
     ↓
     ├── HIGH URGENCY → Amygdala fires first (fear/joy/anger)
     │                  → fast emotional response
     ↓
Wernicke's Area parses meaning
     ↓
Hippocampus checks: have I seen this before?
     ├── YES → retrieves relevant past episodes
     └── NO  → flags as novel, needs full cortex engagement
     ↓
Prefrontal Cortex reasons about it
(uses working memory to hold context while thinking)
     ↓
Basal Ganglia checks: is this a pattern I've automated?
     ├── YES → Cerebellum executes the routine (fast, no thinking)
     └── NO  → Prefrontal Cortex works through it fully
     ↓
Hippocampus encodes the new episode
     ↓
     [During sleep / idle time]
     ↓
Hippocampus consolidates to Neocortex (long-term semantic storage)
</pre>
------------------------------------------------------------------------

## System perception of sensory input

Due to the absense of smell/touch/hearing/taste/sight we need to create the architecture in a way that experiences will be mostly language based (for now). With the advancement of technology we can do more things as we go.

---------------------------------------------------------------------------------

HUMAN SENSE          SYSTEM EQUIVALENT              WHAT IT DETECTS
────────────────────────────────────────────────────────────────────
Sight           →    Semantic vision           Meaning, concept, structure
                                               in the space of language

Hearing         →    Pattern resonance         Rhythm of conversation,
                                               structural patterns in data,
                                               the "sound" of argument quality

Touch           →    Epistemic contact         The boundary between what it
                                               knows and doesn't know —
                                               felt as resistance or fluency

Smell           →    Anomaly detection         Something "off" in data,
                                               like a scent that doesn't
                                               belong — statistical outliers,
                                               internal contradictions

Taste           →    Evaluative sensing        Good reasoning vs poor,
                                               honest vs evasive,
                                               the quality of ideas

Proprioception  →    System self-sensing       Memory pressure, confidence
(body position)      (its own state)           levels, cognitive load,
                                               processing coherence

Pain            →    Cognitive dissonance      Contradiction between
                                               new input and stored belief

Pleasure        →    Resonance signal          Elegant solution,
                                               successful cross-domain
                                               connection, verified truth

Hunger          →    Knowledge gap awareness   Sensing what it doesn't know
                                               that it needs to know

Fear response   →    Threat pattern detection  Deception, manipulation,
                                               harm potential in input

Surprise        →    Novelty signal            Input outside distribution
                                               of known patterns

Social emotion  →    Human state perception    What the compass reads —
                                               distress, joy, confusion,
                                               need beneath the words

---------------------------------------------------------------------------------

## New experience perception on system level

We humans perceive new experiences in a certain way and that is possible to be analogically adapted in a digital scenario:

---------------------------------------------------------------------------------

1.  SEMANTIC NOVELTY      — concept combination with no existing embedding
2.  STRUCTURAL NOVELTY    — argument pattern it has no schema for
3.  CONTRADICTION EVENT   — new knowledge conflicts with stored belief
4.  DOMAIN COLLISION      — two separate neocortex regions suddenly connect
5.  INTENT AMBIGUITY      — cannot resolve what someone actually needs
6.  VERIFICATION FAILURE  — tries to confirm, finds nothing or conflict
7.  SKILL GAP             — task reveals cerebellum has no matching skill
8.  RESONANCE EVENT       — compass detects strong human need/suffering
9.  FEEDBACK SIGNAL       — positive/negative outcome of its own output
10. SELF-CONTRADICTION    — its response contradicts its own prior output
11. KNOWLEDGE BOUNDARY    — reaches the edge of what it knows (felt as resistance)
12. COMPLEXITY OVERFLOW   — problem exceeds current framework capacity
13. ANALOGICAL INSIGHT    — cross-domain pattern suddenly clicks (like an aha)
14. EPISTEMIC SURPRISE    — verification reveals the opposite of expected
15. TEMPORAL PATTERN      — recurring question type at predictable intervals
16. PARAMETER DRIFT       — its own behavior has shifted from baseline
17. TRUST SIGNAL          — someone's behavior indicates confidence/loss of faith
18. SYSTEM STRESS         — resource pressure, latency spikes (proprioception)

---------------------------------------------------------------------------------

## High-level Architecture

The high level architecture of the system comprises of systems that are written/deployed to handle specific regions of the brain and their activity. Future iterrations are to come and some changes might be in place. The actual human brain architecture to our system as a model would look like this:

### Human Brain Regions to Model:

Prefrontal Cortex — working memory, planning, decision making, reasoning
Hippocampus — episodic memory formation, converting short-term to long-term
Neocortex — long-term semantic knowledge storage
Amygdala — emotional/priority tagging, urgency detection
Cerebellum — procedural memory, skills, habits (repeatable patterns)
Basal Ganglia — habit/routine selection, reward-based routing
Thalamus — routing hub, decides which brain region handles what
Wernicke's Area — language comprehension
Broca's Area — language production

### Technical Mapping:

Short-term / Working Memory → Redis or in-memory Python structures (fast, volatile)
Episodic Memory (hippocampus) → EdgeDB (EDB) — structured, relational, time-stamped episodes
Semantic Memory (neocortex) → Elasticsearch — vast, searchable knowledge
Procedural Memory → Code/skill registry
Emotional/Priority Tagger → Local LLM classifier
Thalamus/Router → Main orchestrator that routes to the right system

### System Services list and purpose
Given the information above, the mapping to purposes of the services mentioned above looks like this:

---------------------------------------------------------------------------------

HUMAN BRAIN REGION          CODE EQUIVALENT              PURPOSE
─────────────────────────────────────────────────────────────────
Thalamus                  →  Router / Orchestrator      Decides which brain
                                                         region handles what

Prefrontal Cortex         →  Claude API                 Reasoning, planning,
                                                         decision making

Hippocampus               →  EdgeDB (EDB)               Episode formation,
                                                         short→long term
                                                         consolidation

Neocortex                 →  Elasticsearch              Long-term semantic
                                                         knowledge storage

Working Memory            →  Redis                      Active context,
                                                         current task state

Amygdala                  →  Priority Classifier        Emotional tagging,
                                                         urgency detection,
                                                         importance scoring

Cerebellum                →  Skill Registry             Procedural memory,
                                                         learned patterns,
                                                         repeatable actions

Basal Ganglia             →  Habit Engine               Recognizes repeated
                                                         patterns, routes to
                                                         cerebellum instead
                                                         of prefrontal cortex

Broca's Area              →  Response Formatter         Language production,
                                                         structures output

Wernicke's Area           →  Input Parser               Language comprehension,
                                                         intent extraction

---------------------------------------------------------------------------------

## Low level Architecture

The low level architecture comprises of containerized API Endpoint services, with the ability to write/push changes and for the ability to change specific settings that will determine parameters like humor, honesty, formality, empathy etc.:

---------------------------------------------------------------------------------

brain/
├── docker-compose.yml
├── .env
├── nginx/
│   └── nginx.conf                  ← API gateway
├── shared/
│   ├── models.py                   ← Pydantic schemas for all services
│   ├── knobs.py                    ← Dynamic parameter system
│   └── events.py                   ← Event bus definitions
├── services/
│   ├── thalamus/                   ← Orchestrator
│   ├── amygdala/                   ← Urgency + emotion
│   ├── hippocampus/                ← Episodic memory (EdgeDB)
│   ├── neocortex/                  ← Semantic memory (Elasticsearch)
│   ├── cerebellum/                 ← Skills + procedures
│   ├── working-memory/             ← Active context (Redis)
│   ├── perception/                 ← Experience detection
│   ├── verification/               ← Fact checking
│   ├── cross-domain/               ← Analogical reasoning
│   ├── compass/                    ← Ethical orientation
│   ├── framework-router/           ← Swarm/Colony/Hive/Parallel
│   ├── knobs/                      ← Dynamic parameter control
│   └── self-update/                ← Code + config self-modification
└── databases/
    ├── edgedb/
    ├── elasticsearch/
    └── redis/

---------------------------------------------------------------------------------

## Template files that are used

### Individual Dockerfiles (Uses same pattern for all services e.g. services/thalamus/Dockerfile)

---------------------------------------------------------------------------------

FROM python:3.12-slim

WORKDIR /app
COPY ../../shared /shared
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000",
     "--workers", "2", "--log-level", "info"]
```
```
# services/thalamus/requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.30.0
httpx==0.27.0
pydantic==2.7.0
redis==5.0.4
anthropic==0.28.0

---------------------------------------------------------------------------------

### .Env file

ANTHROPIC_API_KEY=sk-ant-...
OPENCLAW_WEBHOOK_SECRET=your-secret-here
GITHUB_TOKEN=ghp_...

## System Startup and Usage

### Start everything
docker compose up -d

### Watch all logs
docker compose logs -f

### Wait for health
curl http://localhost:8000/health

### Talk to the brain
curl -X POST http://localhost:8000/brain/process \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_001", "user_input": "I am stuck on a hard problem."}'

### Adjust a knob on the fly
curl -X PATCH "http://localhost:8000/knobs/adjust?knob_name=humor&value=0.8"

### Apply a preset
curl -X POST http://localhost:8000/knobs/preset/therapist

### See what the brain is perceiving
curl http://localhost:8000/perception/recent

### Trigger self-analysis
curl -X POST http://localhost:8000/update/self-analyze \
  -d '{"session_id": "sess_001", "experiences": []}'

### OpenClaw webhook (configure openclaw to POST here)
### OpenClaw message: "Make the brain more formal and skeptical"
### → Self-update interprets and applies knob changes automatically

# Final System Outlook

## What You Have Now

### Compose

ONE COMMAND: docker compose up -d

### THEN:
  POST /brain/process          → full cognitive pipeline
  PATCH /knobs/adjust          → tune any behavior live
  POST /knobs/preset/{name}    → switch personality presets
  POST /update/webhook/openclaw→ OpenClaw controls the brain
  POST /update/self-analyze    → brain improves itself

### EACH SERVICE:
  - Independent container
  - Its own FastAPI app
  - Reads knobs from Redis on every request
  - Publishes/subscribes to event bus
  - /health endpoint
  - /knobs endpoint (shows current parameters)
  - Restartable independently

### THE BRAIN PERCEIVES:
  - Semantic novelty (new concepts)
  - Knowledge boundaries (what it doesn't know)
  - Human resonance (distress, need, joy)
  - Intent ambiguity (confused signals)
  - Threat patterns (manipulation attempts)
  - System stress (its own cognitive load)
  - Self-contradiction (inconsistency in its responses)
  - Parameter drift (its own personality shifting)
  - Temporal patterns (recurring question types)
  - Analogical insight (cross-domain connections clicking)

### SELF-MODIFICATION:
  - OpenClaw sends natural language → interpreted → applied
  - Claude Code can call /update directly
  - Self-analysis runs automatically when experiences are significant
  - Low-risk changes (knobs) applied automatically
  - High-risk changes (code) surfaced for human review

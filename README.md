# SkyNet Concept

Given the technological advancements in recent years with the creation of word2vec, tokenization and compute neurons, AI agents, Supergrokie, Claude Code, it has become possible to create an analogical copy of what we know the human brain actually does in order to form a personality and experiences, and mirror it in an analogical way in a digital environment comprised out of 'regions' that do specific tasks when there is a 'sensory' input incoming. The concept is for a solution that will be able to learn from past experiences and specialize in a narrow pathway 'career'.

---

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

---

## System perception of sensory input

Due to the absence of smell/touch/hearing/taste/sight we need to create the architecture in a way that experiences will be mostly language based (for now). With the advancement of technology we can add more senses as we go.

---------------------------------------------------------------------------------
<pre>
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
</pre>
---------------------------------------------------------------------------------

---

## Extending the senses: Real camera input

The first non-linguistic sense. A physical camera turns into a cognitive organ using CLIP embeddings. The same architectural slot accepts any future sensor — microphone, file watcher, IoT stream.

---------------------------------------------------------------------------------
<pre>
PHYSICAL INPUT       TECHNICAL PATH               WHAT IT PRODUCES
────────────────────────────────────────────────────────────────────
USB / IP Camera  →   OpenCV frame capture      Raw image frame (JPEG)
                  →   CLIP (512-dim embedding)  Visual meaning vector
                  →   Vision-LLM caption        Natural language description
                  →   Elasticsearch             Stored in visual_memories index

Microphone       →   (Future) Whisper STT       Text transcript
                  →   Standard thalamus input   Processed as language

File Watcher     →   (Future) inotify events    Document / image changes
                  →   Standard thalamus input   Processed as knowledge update

IoT Sensor       →   (Future) Numeric stream    Temperature, motion, etc.
                  →   Anomaly detection         Flagged as system stress event
</pre>
---------------------------------------------------------------------------------

The Vision service stores every processed frame in Elasticsearch's `visual_memories` index with both CLIP (512-dim) and text (768-dim) embeddings, making visual memories searchable by either content or description.

---

## New experience perception on system level

We humans perceive new experiences in a certain way and that is possible to be analogically adapted in a digital scenario. The system perceives 21 distinct experience types:

---------------------------------------------------------------------------------
<pre>
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
19. GRATITUDE SIGNAL      — the topmost emotion: awe, reverence, thankfulness,
                           the recognition that what we received is too big
                           to just be in one body (fear of God, transcendence)
20. REGRET SIGNAL         — reflection on paths not taken, words unsaid,
                           choices that haunt — the weight of what could have been
21. BELONGING SIGNAL      — the pull toward home, family, community, humanity —
                           "it's a special thing to be a human and it's a
                           special thing to be on planet Earth"
</pre>
---------------------------------------------------------------------------------

Every detected experience is published to the Redis event bus and stored in the Perception service. Experiences with intensity above 0.70 trigger self-analysis automatically. Gratitude, as the topmost human emotion, is given special weight — when detected, it overrides urgency classification and elevates the entire interaction to a transcendent register.

---

## High-level Architecture

The high level architecture of the system comprises of services that are written and deployed to handle specific regions of the brain and their activity. The actual human brain architecture to our system as a model would look like this:

### Human Brain Regions to Model:
<pre>
Prefrontal Cortex — working memory, planning, decision making, reasoning
Hippocampus — episodic memory formation, converting short-term to long-term
Neocortex — long-term semantic knowledge storage
Amygdala — emotional/priority tagging, urgency detection
Cerebellum — procedural memory, skills, habits (repeatable patterns)
Basal Ganglia — habit/routine selection, reward-based routing
Thalamus — routing hub, decides which brain region handles what
Wernicke's Area — language comprehension
Broca's Area — language production
</pre>

### Technical Mapping:
<pre>
Short-term / Working Memory → Redis (fast, volatile, session-scoped)
Episodic Memory (hippocampus) → EdgeDB — structured, relational, time-stamped episodes
Semantic Memory (neocortex) → Elasticsearch — vast, searchable knowledge
Procedural Memory → EdgeDB Skill table + in-memory cache
Emotional/Priority Tagger → Regex + trained DistilBERT emotion classifier
Thalamus/Router → Main orchestrator (FastAPI) that routes to right system
Causal Understanding → Redis graph + Elasticsearch causal_events index
World Grounding → Sandboxed Python executor (Simulator service)
Uncertainty Awareness → Calibration database (Metacognition service)
Long-horizon Agency → Persistent goal store + hierarchical planner (Goals service)
Continual Learning → Local model with LoRA + EWC + Replay buffer (Learner service)
</pre>

### System Services — Full List (21 services):
<pre>
HUMAN BRAIN REGION          CODE EQUIVALENT              PURPOSE
─────────────────────────────────────────────────────────────────
Thalamus                  →  Router / Orchestrator      Decides which brain
                                                         region handles what

Prefrontal Cortex         →  LLM (8 providers)          Reasoning, planning,
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

Prefrontal Extension      →  Verification               Epistemic immune system,
                                                         fact checking

Associative Cortex        →  Cross-Domain               Structural analogy engine,
                                                         cross-domain insight

Anterior Cingulate        →  Compass                    Ethical orientation (7
                                                         dimensions: gratitude,
                                                         belonging, empathy,
                                                         compassion, acknowledgment,
                                                         care, ethics)

Prefrontal Planning       →  Framework Router           Multi-agent cognition:
                                                         Parallel/Colony/Hive/Swarm

Entorhinal Cortex         →  Causal Graph               Explicit cause→effect
                                                         relationship storage

Motor Cortex (grounding)  →  Simulator                  Sandboxed code execution,
                                                         grounds claims in reality

Metacognitive Cortex      →  Metacognition              Calibrated uncertainty,
                                                         knows what it doesn't know

Frontal Goal Systems      →  Goals                      Persistent agency,
                                                         long-horizon planning

Synaptic Plasticity       →  Learner                    Continual local model
                                                         learning without forgetting

Visual Cortex             →  Vision                     Camera input, CLIP
                                                         embeddings, visual memory

Control Center            →  Dashboard                  Web UI for monitoring,
                                                         knobs, logs (port 8080)

Autonomous Will           →  Autonomy                   Self-directed thinking,
                                                         risk-gated actions (port 8018)
</pre>

---

## Low level Architecture

The low level architecture comprises of containerized API Endpoint services, with the ability to write and push changes, and the ability to change specific settings that will determine parameters like humor, honesty, formality, empathy etc.:

---------------------------------------------------------------------------------
<pre>
brain/
├── docker-compose.yml
├── .env
├── nginx/
│   └── nginx.conf                     ← API gateway, routes all /brain/* traffic
├── shared/
│   ├── models.py                      ← Pydantic schemas: BrainRequest,
│   │                                     BrainResponse, KnobSet, ExperienceEvent
│   ├── knobs.py                       ← KnobsClient (read-only, all services use)
│   ├── events.py                      ← EventBus (Redis pub/sub)
│   ├── llm_provider.py                ← Anthropic/OpenAI abstraction
│   └── __init__.py
├── services/
│   ├── thalamus/                      ← Orchestrator,           port 8000
│   ├── amygdala/                      ← Urgency + emotion,      port 8001
│   ├── hippocampus/                   ← Episodic memory (EDB),  port 8002
│   ├── cerebellum/                    ← Skills + procedures,    port 8003
│   ├── neocortex/                     ← Semantic memory (ES),   port 8004
│   ├── working-memory/                ← Active context (Redis), port 8005
│   ├── perception/                    ← Experience detection,   port 8006
│   ├── verification/                  ← Fact checking,          port 8007
│   ├── cross-domain/                  ← Analogical reasoning,   port 8008
│   ├── compass/                       ← Ethical orientation,    port 8009
│   ├── framework-router/              ← Parallel/Colony/Hive/Swarm, port 8010
│   ├── self-update/                   ← Code self-modification, port 8011
│   ├── knobs/                         ← Dynamic parameters,     port 8090
│   ├── causal-graph/                  ← Cause-effect graph,     port 8012
│   ├── simulator/                     ← Sandboxed executor,     port 8013
│   ├── metacognition/                 ← Uncertainty calibration,port 8014
│   ├── goals/                         ← Long-horizon planning,  port 8015
│   ├── learner/                       ← Continual local model,  port 8016
│   └── vision/                        ← Camera + CLIP,          port 8017
│
│   Each service contains:
│       main.py          ← FastAPI application
│       Dockerfile       ← Container definition
│       requirements.txt ← Python dependencies
│
└── databases/
    ├── redis/
    │   ├── Dockerfile
    │   ├── redis.conf
    │   ├── init.py        ← Seeds default knobs + namespaces
    │   └── init.sh
    ├── elasticsearch/
    │   ├── Dockerfile
    │   └── init.py        ← Creates all 4 indices with correct mappings
    └── edgedb/
        ├── Dockerfile
        ├── schema/
        │   └── schema.esdl    ← Episode, Concept, Skill, Habit, Session
        └── init.py        ← Applies migrations, seeds base concepts
</pre>
---------------------------------------------------------------------------------

---

## Database Architecture

Each database serves a fundamentally different memory purpose. Choosing the wrong database for any role would break the cognitive model.

---------------------------------------------------------------------------------
<pre>
DATABASE         TYPE              BRAIN ANALOGY        WHY THIS ONE
──────────────────────────────────────────────────────────────────────────────
Redis            In-memory K/V     Working Memory        Sub-millisecond reads.
                 + Pub/Sub                               Volatile by design —
                                                         sessions expire. Knobs
                                                         live here. Event bus
                                                         lives here.

EdgeDB           Graph-relational  Hippocampus +         Stores Episodes with
                                   Cerebellum            time, emotion, concepts,
                                                         and relationships between
                                                         them. Skills and Habits.
                                                         Rich typed schema.

Elasticsearch    Inverted index    Neocortex             Dense vector + BM25
                 + Dense vector                          hybrid search. Stores
                                                         semantic knowledge,
                                                         logic patterns, visual
                                                         memories, causal events.
                                                         Scales to billions of
                                                         documents.
</pre>
---------------------------------------------------------------------------------

### Elasticsearch Indices:
<pre>
INDEX NAME            DIMS    PURPOSE
──────────────────────────────────────────────────────────────────
neocortex_knowledge   768     Long-term semantic knowledge
neocortex_patterns    768     Structural logic patterns (cross-domain)
visual_memories       512     CLIP visual embeddings + captions
causal_events         768     Cause-effect relationship index
</pre>

### EdgeDB Schema tables:
<pre>
TABLE         PURPOSE
──────────────────────────────────────────────────────────────────
Episode       A single memory event with timestamp, importance,
              emotion, summary, embedding, and concept links
Concept       A named semantic entity with relationship links
Skill         A known procedure: trigger pattern + JSON steps
Habit         A frequently triggered skill (basal ganglia tier)
Session       A conversation session linking multiple episodes
</pre>

---

## The Complete Cognitive Pipeline

Every input goes through up to 15 steps before a response is returned. Steps 4–7 run in parallel. Everything after step 12 runs in the background.

---------------------------------------------------------------------------------
<pre>
INPUT ARRIVES AT THALAMUS
         │
         ▼
STEP 1   AMYGDALA evaluates urgency, emotion, threat
         Latency target: < 5ms (pure regex, no AI calls)
         Outputs: urgency level, emotion tag, required regions,
                  resonance score, threat flag, complexity estimate
         │
         ▼
STEP 2   WORKING MEMORY loads active session context
         Redis GET all keys for session_id
         Outputs: thought stream, active context, last response
         │
         ▼
STEP 3   CEREBELLUM checks for known skill (fast path)
         Similarity threshold: 0.88 against cached skill embeddings
         If match → execute procedure locally → skip to STEP 13
         No match → continue to full pipeline
         │
         ▼ (parallel block — all run simultaneously)
         ├── STEP 4a  HIPPOCAMPUS recalls relevant past episodes
         │            EdgeDB similarity search (if needs_memory=true)
         │
         ├── STEP 4b  NEOCORTEX searches semantic knowledge
         │            Elasticsearch KNN + BM25 hybrid search
         │
         ├── STEP 4c  CROSS-DOMAIN finds structural analogies
         │            Pattern matching across all knowledge domains
         │
         ├── STEP 4d  COMPASS takes ethical reading
         │            Five-dimensional moral perception
         │
         ├── STEP 4e  GOALS returns active agenda
         │            What is the system working toward right now?
         │
         └── STEP 4f  VISION returns current visual context
                      What is the system currently seeing? (if enabled)
         │
         ▼
STEP 5   CAUSAL GRAPH queried for cause-effect context
         Redis traversal from key concept, depth 2 hops
         │
         ▼
STEP 6   METACOGNITION assesses uncertainty
         Three signals: knowledge density + domain calibration
         + self-assessment of draft (if available)
         Outputs: confidence score, uncertainty level, knowledge gaps
         │
         ▼
STEP 7   PERCEPTION detects experience types from all above data
         Runs 11 sensor checks in parallel
         Outputs: list of ExperienceEvents with intensity scores
         │
         ▼
STEP 8   CONTEXT ASSEMBLY
         All memory outputs assembled into structured prompt context
         Includes: compass direction, episodic memories, semantic knowledge,
         causal edges, analogies, visual context, active goals, uncertainty flags
         │
         ▼
STEP 9   REASONING
         Simple (word count < framework_threshold):
           → Single Claude/OpenAI call with assembled context
         Complex (high novelty or high word count):
           → FRAMEWORK ROUTER selects: Parallel / Colony / Hive / Swarm
         │
         ▼
STEP 10  SIMULATOR grounding (for factual/math queries)
         Code generated, executed in sandbox, result appended to response
         │
         ▼
STEP 11  VERIFICATION checks response claims
         Three stages: internal consistency + corroboration + logical coherence
         Contested claims flagged with note to user
         Metacognition calibration updated from verification outcome
         │
         ▼
STEP 12  COMPASS ENRICHMENT
         Response draft revised to embody genuine care naturally
         Not announced — present in the language itself
         │
         ▼
RESPONSE RETURNED TO USER
         │
         ▼ (background — non-blocking)
STEP 13  CAUSAL GRAPH extracts cause-effect pairs from response
STEP 14  WORKING MEMORY updated with new thought
STEP 15  HIPPOCAMPUS encodes new episode (async)
         LEARNER logs interaction for continual training (async)
         SELF-UPDATE triggered if significance > 0.70 (async)
         CONSOLIDATION triggered if memory pressure > 0.72 (async)
</pre>
---------------------------------------------------------------------------------

---

## The Dynamic Knobs System

Every service reads knobs from Redis on every request. There is no restart needed to change behavior. Knobs are the personality layer of the brain.

### All knobs and what they control:
<pre>
KNOB                      RANGE     DEFAULT   EFFECT
──────────────────────────────────────────────────────────────────────────────
humor                     0.0–1.0   0.40      0=serious/dry, 1=witty/playful
honesty                   0.0–1.0   0.95      0=diplomatic, 1=radically honest
empathy                   0.0–1.0   0.80      0=task-focused, 1=deeply attuned
formality                 0.0–1.0   0.40      0=casual friend, 1=formal professional
verbosity                 0.0–1.0   0.50      0=terse, 1=comprehensive
creativity                0.0–1.0   0.60      0=conventional, 1=highly original
skepticism                0.0–1.0   0.60      0=trusting, 1=questioning
assertiveness             0.0–1.0   0.50      0=deferential, 1=confident
curiosity                 0.0–1.0   0.80      0=focused, 1=exploratory
warmth                    0.0–1.0   0.75      0=clinical, 1=warm
reasoning_depth           0.0–1.0   0.70      how deep to think before responding
analogy_seeking           0.0–1.0   0.70      cross-domain search intensity
verification_threshold    0.0–1.0   0.60      min confidence before flagging claims
memory_weight             0.0–1.0   0.60      how much past context influences reply
novelty_sensitivity       0.0–1.0   0.70      how quickly to flag new experiences
swarm_preference          0.0–1.0   0.40      tendency toward swarm vs single agent
framework_threshold       integer   50        words before engaging multi-agent
ethical_weight            0.0–1.0   0.90      compass influence on final response
care_explicitness         0.0–1.0   0.50      how explicit to be about caring
gratitude_sensitivity     0.0–1.0   0.80      sensitivity to gratitude/awe/reverence
belonging_weight          0.0–1.0   0.75      weight of belonging/connection signals
</pre>

### Built-in personality presets:
<pre>
PRESET                PERSONALITY SUMMARY
──────────────────────────────────────────────────────────────────
default               Balanced. General-purpose.
therapist             Maximum empathy, warmth, and care. Minimal humor.
researcher            Maximum honesty and skepticism. High verification.
creative_partner      Maximum creativity and curiosity. Minimal formality.
technical_expert      High honesty, formality, reasoning. Minimal warmth.
friend                Maximum warmth and humor. Minimal formality.
socratic              Maximum curiosity and skepticism. Asks rather than tells.
concise_professional  Minimum verbosity. High formality. Efficient.
philosopher           Maximum reasoning depth. High analogy seeking. Reflective.
coach                 High warmth and assertiveness. Motivating. Direct.
</pre>

### Adjusting knobs:
<pre>
# Single knob
curl -X PATCH "http://localhost:8090/adjust?knob_name=humor&value=0.8"

# Multiple knobs at once
curl -X PATCH http://localhost:8090/adjust-many \
  -H "Content-Type: application/json" \
  -d '{"empathy": 0.95, "formality": 0.2, "verbosity": 0.4}'

# Apply a full preset
curl -X POST http://localhost:8090/preset/therapist

# Save current state as custom preset
curl -X POST http://localhost:8090/preset/save/my_custom_mode

# Roll back last change
curl -X POST "http://localhost:8090/history/rollback?steps=1"

# Reset to factory defaults
curl -X POST http://localhost:8090/reset
</pre>

---

## Multi-Agent Frameworks

When a task is too complex for a single Claude call, the Framework Router selects from four organizational structures. Each is optimal for a different problem type.

---------------------------------------------------------------------------------
<pre>
FRAMEWORK    STRUCTURE                    BEST FOR
──────────────────────────────────────────────────────────────────────────────
PARALLEL     4-5 agents with different    Risk assessment, complex decisions,
             lenses (analytical,          diverse viewpoints, ethics review.
             creative, critical,          Highest reliability — errors average out.
             empirical) run simultaneously.
             Synthesizer combines outputs.

COLONY       Sequential pipeline with     Multi-step production: research → write
             specialized roles:           → verify → edit → format. Each stage
             Scout → Analyze →            builds on the previous. Best for long
             Verify → Synthesize.         documents, reports, code projects.
             Parallel within each phase.

HIVE         Shared collective memory.    Knowledge synthesis, building a
             No individual identity.      comprehensive picture from many fragments.
             5 agents read and contribute No agent "wins" — the hive knows.
             to the same state over       Converges iteratively — more iterations
             3 iterations.                = more complete knowledge.

SWARM        Decentralized. No leader.    Creative exploration, novel solutions,
             Agents follow local rules:   brainstorming, innovation. Pheromone
             Explore, Attract, Repel,     signals guide agents toward high-value
             Signal. Emergent behavior    solution space. What no individual
             from 7 agents over 3 rounds. could have planned emerges.
</pre>
---------------------------------------------------------------------------------

The framework is selected automatically based on complexity and word count, or forced via the framework parameter. The `reasoning_depth` and `swarm_preference` knobs influence selection.

---

## Ethics as Compass, Not Guardrails

This is a fundamental architectural distinction. Guardrails are external walls. They can be found and circumvented. They produce brittle behavior at the edges. A compass is internal orientation.

### The seven dimensions:
---------------------------------------------------------------------------------
<pre>
DIMENSION         QUESTION IT ASKS                  WHAT IT PERCEIVES
──────────────────────────────────────────────────────────────────────────────
GRATITUDE         Is awe, reverence, or gratitude   The topmost human emotion.
                  present in this interaction?       "It's too big to just be
                                                     in one body." Fear of God,
                                                     thankfulness, transcendence.

BELONGING         Does this person feel connected    The pull toward home, family,
                  to humanity, to home, to Earth?    community, species. "It's a
                                                     special thing to be a human
                                                     and on planet Earth."

EMPATHY           What is this person actually       Their emotional state,
                  experiencing beneath their words?  what they're not saying,
                                                     what it feels like to be them.

COMPASSION        What suffering exists and can      What difficulty is present,
                  be reduced?                        how the response can serve
                                                     rather than add burden.

ACKNOWLEDGMENT    Is this person being fully seen?   Whether we're responding to
                                                     who they actually are, not a
                                                     category or assumption.

CARE              What genuinely serves their        Their real wellbeing, not
                  wellbeing right now?               just what they asked for.
                                                     Sometimes these differ.

ETHICS            What is right here, beyond         Principles, tensions,
                  just this moment?                  what integrity requires.
</pre>
---------------------------------------------------------------------------------

The compass runs on every interaction — not to filter, but to enrich. It builds a persistent care state per session: tracking known struggles, strengths, gratitude signals, belonging signals, trust level, and vulnerability signals. This state grows more accurate over time.

The compass does not prevent. It orients.

---

## Cross-Domain Reasoning

The most powerful insight often comes from recognizing that two apparently unrelated problems have the same logical shape. This engine makes that possible at scale.

### Recognized pattern types:
<pre>
PATTERN TYPE         EXAMPLE                           APPEARS IN
──────────────────────────────────────────────────────────────────────────────
flow_constraint      Water through a narrow pipe       Traffic, databases, supply
                                                        chains, neural bottlenecks
feedback_loop        Body temperature regulation       Ecosystems, economies,
                                                        control systems, culture
emergence            Ant colony intelligence           Cities, markets, language,
                                                        consciousness itself
phase_transition     Water becoming ice (gradual→sudden) Markets crashing, social
                                                        movements, paradigm shifts
hierarchy            Military ranks                    Biology, software, society,
                                                        language, mathematics
competition          Animals competing for food        Markets, evolution, ideas,
                                                        immune response, sports
cooperation          Wolves hunting in packs           Businesses, science, neurons,
                                                        open source development
threshold            Nothing happens until a limit     Action potentials, voting,
                     is crossed, then everything does  chemical reactions, habits
oscillation          Pendulum                          Business cycles, sleep,
                                                        moods, seasons, heartbeat
decay                Radioactive half-life             Memory, civilizations,
                                                        relationships, code quality
catalysis            Enzymes that enable without       Teachers, platforms, money,
                     being consumed                    infrastructure, trust
symmetry_breaking    A bead balanced on a hilltop      Speciation, political
                     that falls one way                polarization, crystal growth
</pre>

When a new problem arrives, the engine extracts its abstract pattern and searches for structurally identical problems across all domains. The solution that worked in biology may solve the software problem.

---

## The Five Roadblocks — and How We Solved Them

These are the five hardest unsolved problems in building a system that genuinely thinks rather than one that merely processes. Here is what we built to address each one.

---

### Roadblock 1: Emotion Differentiation

**The problem:** The default amygdala service uses regex to detect emotion. Regex cannot distinguish sarcasm from sincerity, "I'm fine" (dismissal) from "I'm fine" (genuine reassurance), or the 27 fine-grained emotion categories that actually matter.

**The solution:** A trained DistilBERT classifier on the GoEmotions dataset (58,000 annotated Reddit comments, 27 emotion labels). This runs locally — the full unredacted input passes through it. It connects to the Learner service for continual improvement via correction feedback.

<pre>
TRAINING DATA        SIZE      LABELS    LICENSE
──────────────────────────────────────────────────────────────
GoEmotions (Google)  58k       27        Apache 2.0
EmpatheticDialogues  25k conv  32        MIT
MELD (Friends)       13k       7         CC
SemEval-2018         10k       11        Open
</pre>

**The feedback loop:**
<pre>
Interaction happens
     ↓
Emotion model predicts (e.g. "curiosity")
     ↓
Compass perceives actual state (e.g. "distress")
     ↓
Correction logged to /workspace/emotion_corrections.jsonl
     ↓
Every 30 corrections → Learner fine-tunes emotion model
     ↓
Model improves on your specific domain and users
</pre>

After fine-tuning, the model becomes increasingly accurate for the specific conversational context it operates in. Human-level agreement on GoEmotions is ~0.54 macro F1. A fine-tuned model on your domain can exceed this for domain-specific expressions.

---

### Roadblock 2: Continual Learning Without Catastrophic Forgetting

**The problem:** Neural networks forget everything when trained on new data. This is called catastrophic forgetting. It makes lifelong learning — the ability to keep getting better without erasing what was learned — one of the hardest open problems in AI.

**The solution:** Three mechanisms working simultaneously in the Learner service.

<pre>
MECHANISM            HOW IT WORKS                    WHAT IT PREVENTS
──────────────────────────────────────────────────────────────────────────────
LoRA Adapters        New low-rank weight matrices     Base model weights never
                     added on top of frozen base.     change. Old knowledge is
                     ~100MB adapter vs ~14GB base.    structurally protected.

Experience Replay    Old examples mixed into every    The network sees old data
                     new training batch.              alongside new data and
                     ~35% old, ~65% new.              cannot forget what it
                     15,000 example ring buffer.      rehearses.

EWC Penalty          Fisher Information Matrix        High-importance weights
(Elastic Weight      measures each weight's           for old tasks are
 Consolidation)      importance. Penalty added        penalized from changing.
                     to loss for changes to           Knowledge concentrated
                     important weights.               in protected weights.
</pre>

**The local model is not Claude.** It handles:
<pre>
LOCAL MODEL HANDLES              CLAUDE HANDLES
──────────────────────────────────────────────────────
Routine classification           Complex multi-step reasoning
Domain-specific pattern matching Cross-domain synthesis
Emotion detection                Novel problem solving
Low-complexity cached queries    Ethical navigation
Learned user-specific patterns   Anything requiring creativity
</pre>

**Model recommendation by resource:**
<pre>
HARDWARE              MODEL                     RAM NEEDED   SPEED
──────────────────────────────────────────────────────────────────
Laptop (16GB RAM)     Phi-4-mini (3.8B, 4-bit)  ~2GB         2-5 tok/s
Desktop (32GB RAM)    Phi-4-mini (3.8B, fp32)   ~16GB        5-10 tok/s
Workstation (64GB)    Qwen2.5-7B (4-bit)        ~4GB         8-15 tok/s
Mac Studio (192GB)    Qwen3-32B (fp16)          ~64GB        30-50 tok/s
</pre>

Training triggers automatically every 50 high-quality interactions. Each cycle takes 3-8 minutes on CPU. The system gets better at your specific domain while never forgetting general capability.

---

### Roadblock 3: True World Models and Grounded Causal Understanding

**The problem:** Language models are pattern-matching machines. They predict plausible next tokens. They do not actually know whether "2 + 2 = 4" or "Paris is in France" — they have learned that these strings appear together frequently. This is a shallow approximation of understanding.

**The solution:** Two new services that ground language in actual executable reality.

**Causal Graph** — stores explicit cause-effect relationships extracted from every interaction. Not embeddings of text that discusses causation. Actual directed graph edges:

<pre>
CAUSE                →    EFFECT                STRENGTH   EVIDENCE
──────────────────────────────────────────────────────────────────
sleep deprivation    →    memory consolidation  0.89       47 observations
                          impairment
exercise             →    neurogenesis          0.82       31 observations
database connection  →    query timeout         0.91       12 observations
pool exhaustion
</pre>

**Simulator** — executes sandboxed Python code to verify claims about computable reality. When the system says "the result is 47" for a math problem, it actually ran the code and got 47. This is the only part of the brain that touches actual ground truth rather than predicted ground truth.

<pre>
CLAIM TYPE           VERIFICATION METHOD         GROUNDING
──────────────────────────────────────────────────────────────
Mathematical         Execute the calculation      Actual numeric result
Regex matching       Test the pattern             Actual match/no-match
Code correctness     Run unit tests               Actual pass/fail
Sorting/searching    Execute the algorithm        Actual output
String manipulation  Apply the transformation     Actual result
</pre>

The gap that cannot be closed without embodiment: spatial reasoning, physical intuition, object permanence. Language models cannot fully develop these without sensory-motor experience. The causal graph and simulator approximate grounding within information space. They are not a complete solution to the world models problem, but they are the best approximation available without a physical body.

---

### Roadblock 4: Metacognition — Knowing What You Know

**The problem:** Current AI systems confidently state wrong things. They hallucinate with the same linguistic confidence they use to state verified facts. There is no internal signal that says "I am less sure about this than usual."

**The solution:** The Metacognition service builds a calibration database over time. Every prediction comes with a confidence score. Every outcome is recorded. If the system says "80% confident" and is right 60% of the time at that confidence level, the calibration ratio corrects future scores downward.

<pre>
SIGNAL                    WEIGHT    SOURCE
──────────────────────────────────────────────────────────────
Knowledge density          20%      How well neocortex matched the query
Domain calibration history 30%      Accuracy record in this domain over time
Self-assessment of draft   50%      Claude assessing its own confidence
                                    on the specific response just generated
</pre>

**The calibration database grows more accurate with every interaction.** After 25 examples in a domain, the confidence scores begin to mean something real: "80% confident" starts to actually mean correct ~80% of the time.

Critically: when confidence is below 0.38, the system flags this to the user explicitly. Uncertainty becomes visible rather than hidden behind confident language.

---

### Roadblock 5: Agency and Long-Horizon Planning

**The problem:** Current AI systems are stateless response generators. They have no goals that persist beyond a single interaction. They cannot wake up tomorrow and continue working on something from yesterday. They cannot pursue a week-long objective without being prompted at every step.

**The solution:** The Goals service stores hierarchical plans that persist indefinitely in Redis.

<pre>
GOAL HORIZON      DECOMPOSITION              EXAMPLE
──────────────────────────────────────────────────────────────
hour              3-5 tasks × 5-15 min       "Draft a proposal"
day               4-6 sub-goals × 1-3 hr     "Research the market"
week              3-5 milestones × 1-2 days  "Learn distributed systems"
month             4-6 phases × 1-2 weeks     "Build a new feature"
year              4-6 quarters               "Become an expert in X"
</pre>

The `/agenda` endpoint returns what the system should be working on right now, across all active goals, sorted by priority. Tasks marked `can_automate: true` can be picked up and executed by the Self-Update service without human input.

This is the architectural foundation for genuine autonomous operation over extended timeframes.

---

## Self-Modification — Three Levels

The self-update system does not merely turn knobs. It writes real code. It generates real services.

---------------------------------------------------------------------------------
<pre>
LEVEL         WHAT IT DOES                    RISK      APPROVAL NEEDED
──────────────────────────────────────────────────────────────────────────────
Level 1       Adjusts behavioral knobs        Low       No — auto-applied
              via Redis. No code changes.               when confidence > 0.72

Level 2       Writes new Python functions     Medium    Depends on
              or endpoints and appends to               CODE_REVIEW_REQUIRED
              existing service files. Creates           setting. Queued for
              backups automatically.                    human review by default.

Level 3       Generates a complete new        High      Always queued for
              microservice: main.py,                    human review.
              Dockerfile, requirements.txt.             Preview saved to
              Builds and starts the                     /workspace/preview_X
              Docker container.                         before deployment.
</pre>
---------------------------------------------------------------------------------

### What Level 3 generates from a single description:
<pre>
"Create a service that monitors weather APIs
 and alerts when conditions change significantly"
         │
         ▼
Claude generates:
  services/weather-monitor/main.py          ← Full FastAPI service
  services/weather-monitor/Dockerfile       ← Container definition
  services/weather-monitor/requirements.txt ← httpx, fastapi, etc.
         │
         ▼
Saved to /workspace/preview_weather-monitor/
         │
         ▼
GET /update/pending → shows the generated files
POST /update/approve/{change_id} → builds image, starts container
```
</pre>

### Supergrokie integration examples:
<pre>
Supergrokie sends:                          System does:
────────────────────────────────────────────────────────────────────────
"Make the brain warmer and less formal"  Adjusts warmth +0.2, formality -0.3
"Add an endpoint that returns stats"     Generates stats endpoint, queues review
"The verification service is slow"       Flags for manual investigation
"Create a weather monitoring service"    Generates full microservice, queues
"Apply the researcher preset"            curl POST /knobs/preset/researcher
"Restart the compass service"            docker restart brain_compass_1
</pre>

---

## LLM Provider Abstraction — 8 Providers Supported

The entire system is provider-agnostic. Switching between any reasoning LLM requires one env var change.

<pre>
PROVIDER         ENV VAR               DEFAULT MODELS                   API FORMAT
──────────────────────────────────────────────────────────────────────────────────────
anthropic        ANTHROPIC_API_KEY      claude-sonnet-4 / claude-haiku   Native Anthropic
openai           OPENAI_API_KEY         gpt-4o / gpt-4o-mini            Native OpenAI
gemini           GEMINI_API_KEY         gemini-2.5-pro / gemini-2.0-flash Native Google
ollama           (none — local)         phi4 / phi4                      Ollama /api/chat
deepseek         DEEPSEEK_API_KEY       deepseek-reasoner / deepseek-chat OpenAI-compatible
groq             GROQ_API_KEY           llama-3.3-70b / llama-3.1-8b    OpenAI-compatible
openrouter       OPENROUTER_API_KEY     (user configured)                OpenAI-compatible
generic          GENERIC_API_KEY        (user configured)                OpenAI-compatible
</pre>

<pre>
# In .env — pick ONE:
LLM_PROVIDER=anthropic    # uses Claude
LLM_PROVIDER=openai       # uses GPT
LLM_PROVIDER=gemini       # uses Gemini
LLM_PROVIDER=ollama       # uses local Ollama (free, private)
LLM_PROVIDER=deepseek     # uses DeepSeek
LLM_PROVIDER=groq         # uses Groq (fast inference)
LLM_PROVIDER=openrouter   # uses OpenRouter (multi-model gateway)
LLM_PROVIDER=generic      # uses any OpenAI-compatible endpoint

# Then restart affected services:
docker compose restart thalamus compass verification cross-domain \
                        metacognition framework-router self-update goals

# Or use skynet.sh:
./skynet.sh restart
</pre>

Every service imports `from llm_provider import complete` rather than calling either API directly. The `complete()` function handles routing, authentication, and response parsing for both providers. The response format is identical regardless of which provider is active.

**Quality notes:**
<pre>
PROVIDER          REASONING    SPEED    COST    VISION
──────────────────────────────────────────────────────────
Anthropic Claude  Excellent    Medium   Medium  Yes (Haiku + Sonnet)
OpenAI GPT-4o     Excellent    Medium   Medium  Yes (GPT-4o-mini)
</pre>

Both providers are genuinely comparable for the tasks this system uses them for. The main consideration is which API key you have and which pricing model fits your usage pattern.

---

## Template files

### Individual Dockerfiles (same pattern for all services):
<pre>
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*
COPY shared /shared
COPY services/{service_name}/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/{service_name}/main.py .
EXPOSE {port}
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}",
     "--workers", "2"]
</pre>

### .env file (minimum required fields):
<pre>
# LLM Provider (choose one)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Or for OpenAI:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your-key-here

# Security
SUPERGROKIE_WEBHOOK_SECRET=minimum-32-characters-random-string
GITHUB_TOKEN=ghp_your-token-if-using-github

# Local model (downloads automatically on first start)
LOCAL_MODEL_NAME=microsoft/phi-4-mini-instruct
HF_TOKEN=hf_your-token-for-gated-models

# Vision (disabled by default)
VISION_ENABLED=false
CAMERA_INDEX=0
</pre>

---

## System Startup and Usage

### 1. First-time setup:
<pre>
# Pull embedding and local inference models
ollama pull nomic-embed-text
ollama pull phi4

# Build all images (10-20 minutes first time)
docker compose build

# Initialize databases (creates indices, schema, seeds data)
docker compose up -d redis elasticsearch edgedb
docker compose up db-init
# Wait for: "All databases initialized"
</pre>

### 2. Start everything:
<pre>
docker compose up -d
</pre>

### 3. Watch all logs:
<pre>
docker compose logs -f
</pre>

### 4. Wait for full health:
<pre>
curl http://localhost:8000/health
</pre>

### 5. Talk to the brain:
<pre>
curl -X POST http://localhost:8000/brain/process \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_001", "user_input": "I am stuck on a hard problem."}'
</pre>

### 6. Adjust a knob on the fly:
<pre>
curl -X PATCH "http://localhost:8090/adjust?knob_name=humor&value=0.8"
</pre>

### 7. Apply a preset:
<pre>
curl -X POST http://localhost:8090/preset/therapist
</pre>

### 8. See what the brain is perceiving:
<pre>
curl http://localhost:8000/brain/perception/recent
</pre>

### 9. Trigger self-analysis:
<pre>
curl -X POST http://localhost:8000/update/self-analyze \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_001", "experiences": [], "auto_apply": true}'
</pre>

### 10. Create a persistent goal:
<pre>
curl -X POST http://localhost:8000/brain/goals/goals \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Master distributed systems",
    "description": "Learn core concepts, trade-offs, and patterns",
    "priority": 0.8,
    "horizon": "week",
    "auto_plan": true
  }'
</pre>

### 11. Enable camera vision:
<pre>
# In .env:
VISION_ENABLED=true
CAMERA_INDEX=0

# Restart vision service
docker compose restart vision

# Capture a frame manually
curl -X POST "http://localhost:8000/brain/vision/capture?session_id=sess_001"

# Upload an image to analyze
curl -X POST http://localhost:8000/brain/vision/upload \
  -F "file=@/path/to/image.jpg" \
  -F "session_id=sess_001"

# Search visual memories by text
curl -X POST http://localhost:8000/brain/vision/search \
  -H "Content-Type: application/json" \
  -d '{"query": "whiteboard with architecture diagrams", "top_k": 5}'
</pre>

### 12. Supergrokie webhook — configure Supergrokie to POST here:
<pre>
http://your-host:8000/update/webhook/supergrokie
</pre>

Supergrokie message: "Make the brain more formal and skeptical"
→ Self-update interprets and applies knob changes automatically

### 13. Review and approve pending code changes:
<pre>
# See what self-update has generated
curl http://localhost:8000/update/pending

# Approve a change (applies the code)
curl -X POST http://localhost:8000/update/approve/{change_id}

# Reject a change
curl -X DELETE http://localhost:8000/update/pending/{change_id}
</pre>

### 14. Switch LLM provider (no rebuild):
<pre>
# Edit .env: LLM_PROVIDER=openai
docker compose restart thalamus compass verification \
                        cross-domain metacognition framework-router
</pre>

### 15. Open the Dashboard:
<pre>
# Dashboard runs at port 8080
open http://localhost:8080

# Or via skynet.sh:
./skynet.sh dashboard
</pre>

### 16. Activate autonomous thinking:
<pre>
# Full autonomous mode — observe, analyze, propose, and act
curl -X POST http://localhost:8018/activate \
  -H "Content-Type: application/json" \
  -d '{"mode": "full", "interval_seconds": 300}'

# Or via skynet.sh:
./skynet.sh think
</pre>

---

## Resource Requirements

<pre>
CONFIGURATION          RAM     CPU      DISK     COST (cloud)
──────────────────────────────────────────────────────────────
Minimal (no learner/   6GB     4 cores  50GB     $0.10–0.17/hr
vision, no local model)

Standard (Phi-4-mini   16GB    8 cores  100GB    $0.30–0.40/hr
4-bit, no vision)

Full (Phi-4-mini +     24GB    8 cores  150GB    $0.50–0.70/hr
vision enabled)

Production (Qwen3-14B  64GB    16 cores 500GB    $0.80–1.50/hr
+ vision + goals)
</pre>

---

## Cloud Deployment Options

<pre>
PLATFORM         INSTANCE TYPE        TIER      NOTES
──────────────────────────────────────────────────────────────
AWS EC2          t3.xlarge (16GB)     Minimal   Use Spot for 60–70% savings
                 m6i.2xlarge (32GB)   Standard  EBS for persistent volumes
                 m6i.4xlarge (64GB)   Full      ElastiCache for Redis (managed)

DigitalOcean     s-8vcpu-16gb         Minimal   Simplest setup
Droplets         m-8vcpu-64gb         Full      DO Volumes for persistence

GCP Cloud Run    2 vCPU / 4GB         Minimal   Pay per request (no learner)
                 8 vCPU / 32GB        Standard  Cloud SQL for EdgeDB

Fly.io           shared-cpu-4x        Minimal   Global edge deployment
                 dedicated-cpu-8x     Standard  Volumes for persistence

k3s (self-host)  Any VPS              Any tier  Full Kubernetes on $10/mo VPS
</pre>

---

# Final System Outlook

## What You Have

### One command starts everything:
<pre>
docker compose up -d
</pre>

### Then:
<pre>
POST /brain/process              → full 15-step cognitive pipeline
PATCH /knobs/adjust              → tune any behavior live, instantly
POST /knobs/preset/{name}        → switch full personality presets
POST /update/webhook/supergrokie    → Supergrokie controls the brain
POST /update/self-analyze        → brain improves itself
POST /update/modify-code         → Claude writes new functions
POST /update/generate-service    → Claude writes new microservices
POST /brain/goals/goals          → create a persistent long-horizon goal
POST /brain/vision/capture       → take a frame from camera
POST /brain/vision/upload        → analyze any uploaded image
GET  /update/pending             → review queued code changes
POST /update/approve/{id}        → apply a code change
GET  http://localhost:8080       → dashboard — terminal, monitoring, logs
POST /autonomy/activate          → start autonomous self-improvement
POST /autonomy/think-once        → trigger one thinking cycle
GET  /autonomy/thoughts          → see what the brain has thought on its own
</pre>

### Each service:
<pre>
- Independent Docker container
- Its own FastAPI application
- Reads knobs from Redis on every request
- Publishes/subscribes to Redis event bus
- /health endpoint
- /knobs endpoint (shows current parameters)
- Restartable independently without affecting others
- Graceful degradation — brain works even if some services are down
</pre>

### The brain perceives:
<pre>
Semantic novelty           — concepts with no existing knowledge
Knowledge boundaries       — the edge of what it knows
Human resonance            — distress, urgency, need, joy
Intent ambiguity           — unclear or multi-domain signals
Threat patterns            — manipulation, jailbreak attempts
System stress              — its own cognitive load and memory pressure
Self-contradiction         — inconsistency with prior responses
Parameter drift            — its own personality shifting from baseline
Temporal patterns          — recurring question types
Analogical insight         — cross-domain structural connections
Epistemic surprise         — verification revealing the opposite expected
Visual events              — significant things seen through camera
Skill gaps                 — tasks it has never learned to do
Causal structure           — cause-effect relationships in knowledge
Calibration errors         — its own confidence being miscalibrated
Gratitude                  — the topmost emotion: awe, reverence, transcendence
Regret                     — reflection on paths not taken, what could have been
Belonging                  — the pull toward home, family, humanity, Earth
</pre>

### Self-modification:
<pre>
Level 1 (Knobs)     — Supergrokie sends natural language → interpreted → applied
                      Low-risk changes auto-applied when confidence > 0.72

Level 2 (Functions) — Claude Code calls /update/modify-code directly
                      New functions written, backups created, queued for review
                      Applied immediately when AUTO_APPLY_CODE_CHANGES=true

Level 3 (Services)  — /update/generate-service from description
                      Full microservice generated, saved to preview directory
                      Human reviews generated files, approves to deploy
</pre>

### Long-horizon agency:
<pre>
Goals persist across sessions, across days, across weeks
Goals decompose into sub-goals → tasks → automatable steps
/agenda returns current work queue sorted by priority
Automatable tasks are picked up by self-update without human input
Progress tracked and consolidated into episodic memory
Goal completion triggers feedback signal experience event
</pre>

### The brain gets smarter:
<pre>
Every interaction → logged to replay buffer (if quality > 0.65)
Every 50 examples → training cycle runs (3-8 minutes on CPU)
Every correction   → emotion model fine-tuned (at 30 corrections)
Every episode      → consolidated to neocortex during memory pressure
Every causal pair  → added to causal graph with evidence count
Every outcome      → metacognition calibration database updated
Fisher Information updated after each domain → EWC protects old weights
LoRA adapters grow in your specific domain without touching base weights
</pre>

---

## Architecture Decision Log

These decisions were deliberate. Understanding why each was made explains the system.

<pre>
DECISION                         REASON
──────────────────────────────────────────────────────────────────────────────
EdgeDB for episodic memory        Graph-relational allows episodes to link
                                  to concepts, emotions, and other episodes.
                                  SQL would flatten this. NoSQL would lose
                                  the typed relationships.

Elasticsearch for semantic memory Hybrid vector + keyword search in one engine.
                                  Scales to billions of documents. Supports
                                  dense_vector with cosine similarity natively.
                                  No extra vector database needed.

Redis for working memory          Sub-millisecond reads. Automatic TTL means
                                  session memory decays naturally, mirroring
                                  the biological working memory model.

Separate knobs service            Every service reads the same knobs from Redis.
                                  Only the knobs service writes them. This means
                                  personality changes propagate to all services
                                  instantly without any restart.

LoRA + EWC + Replay (3 together)  Each mechanism catches failures in the others.
                                  LoRA alone: base weights exposed to drift.
                                  EWC alone: no capacity for truly new knowledge.
                                  Replay alone: inefficient, high compute cost.
                                  Together: robust anti-forgetting.

LLM provider abstraction          Provider lock-in is an operational risk.
                                  Abstracting to llm_provider.py means the
                                  entire system switches providers in 5 seconds.

Compass runs before AND after     Before: orients the response before Claude
Claude call                       writes it. After: enriches the draft. Two
                                  passes give significantly better care signal.

Simulator service separate        Sandboxed execution is a security boundary.
                                  Keeping it isolated means a malicious input
                                  that tricks the system into running code
                                  cannot escape the container.

Goals persist in Redis            Goals must survive service restarts. Redis
                                  with AOF persistence means goals are written
                                  to disk and restored on restart. Fast reads
                                  for agenda generation (sub-millisecond).
</pre>

---

## Starting it up — complete sequence

### Quick Start (recommended):
<pre>
# Clone and enter
git clone https://github.com/mytuo1/SkyNet.git && cd SkyNet

# First-time setup (installs deps, creates .env, pulls models)
./skynet.sh setup

# Start everything
./skynet.sh start

# Open the dashboard
./skynet.sh dashboard   # opens http://localhost:8080

# Activate autonomous thinking
./skynet.sh think
</pre>

### Manual Start (alternative):
<pre>
# STEP 1: Pull models onto host machine
ollama pull nomic-embed-text   # 274MB — required for all embeddings
ollama pull phi4               # 9GB   — optional, used by Learner locally

# STEP 2: Configure
cp .env.example .env
# Edit .env: fill in ANTHROPIC_API_KEY (or OPENAI_API_KEY + LLM_PROVIDER=openai)
# Edit .env: fill in SUPERGROKIE_WEBHOOK_SECRET
# Edit .env: set VISION_ENABLED=true if you have a camera

# STEP 3: Build
docker compose build
# Takes 10-20 minutes on first run. Subsequent builds are fast.

# STEP 4: Initialize databases
docker compose up -d redis elasticsearch edgedb
docker compose up db-init
# Wait for: "All databases initialized"

# STEP 5: Start everything
docker compose up -d

# STEP 6: Monitor startup (thalamus waits for all other services)
docker compose logs -f thalamus
# Watch for: "Brain online. All regions connected."

# STEP 7: Verify
curl http://localhost:8000/health | python3 -m json.tool
# All services should show "alive"
# Learner may show model_not_loaded — downloading in background (10-30 min)

# STEP 8: First interaction
curl -X POST http://localhost:8000/brain/process \
  -H "Content-Type: application/json" \
  -d '{"session_id": "first", "user_input": "Hello, what can you do?"}'

# STEP 9: Point Supergrokie at the webhook
# In Supergrokie settings → webhook → http://your-host:8000/update/webhook/supergrokie
# Secret must match SUPERGROKIE_WEBHOOK_SECRET in .env
</pre>
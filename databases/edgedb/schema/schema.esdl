module default {

    # ── Episodic memory — what happened, when, how it felt ─────────────────────
    type Episode {
        required property content     -> str;
        required property timestamp   -> datetime;
        required property importance  -> float64;
        required property emotion_tag -> str;

        property summary              -> str;
        property embedding            -> array<float32>;
        property consolidated         -> bool { default := false; }
        property consolidation_time   -> datetime;

        multi link concepts           -> Concept;
        multi link related_episodes   -> Episode;
    }

    # ── Semantic concepts — pure meaning, no timestamp ─────────────────────────
    type Concept {
        required property name   -> str { constraint exclusive; }
        property description     -> str;
        property embedding       -> array<float32>;
        property access_count    -> int64 { default := 0; }
        property last_accessed   -> datetime;
        multi link related       -> Concept;
    }

    # ── Skills — procedural memory, how to DO things ───────────────────────────
    type Skill {
        required property name            -> str { constraint exclusive; }
        required property trigger_pattern -> str;
        required property procedure       -> str;   # JSON array of steps
        property success_count            -> int64 { default := 0; }
        property failure_count            -> int64 { default := 0; }
        property last_used                -> datetime;
        property avg_execution_ms         -> float64;
    }

    # ── Habits — frequently triggered skills (basal ganglia) ───────────────────
    type Habit {
        required link  skill             -> Skill;
        required property trigger_embedding -> array<float32>;
        required property frequency      -> int64;
        property is_active               -> bool { default := true; }
    }

    # ── Session state ─────────────────────────────────────────────────────────
    type Session {
        required property session_id -> str { constraint exclusive; }
        required property started_at -> datetime;
        property ended_at            -> datetime;
        property episode_count       -> int64 { default := 0; }
        multi link episodes          -> Episode;
    }
}

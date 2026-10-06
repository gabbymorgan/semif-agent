# Input → Output flow

How one input travels through the agent. Every diamond marked **SemIf** is a
real decision-engine call that is logged as a training row; the LLM only ever
*generates* (skill drafts, bodies, questions). There is no up-front
handle/ignore gate — every input is dispatched.

The diagram is split in two: the **main pipeline** (intake → navigation → run →
output) and the **authoring pipeline** (what happens when navigation decides a
category or skill does not exist yet). Authoring is asynchronous: the main gate
is freed immediately and the original request is re-queued once the body lands.

Rendered SVG: [flow-main.svg](flow-main.svg) · [flow-authoring.svg](flow-authoring.svg)

---

## Main pipeline

```mermaid
flowchart TD
    IN["Input<br/>REPL text · gateway DM · dashboard submit"] --> SUB["Scheduler.submit_request()<br/>create Request, trace 'submit'"]

    SUB --> BUSY{"current process?"}

    %% ---------- idle path ----------
    BUSY -- "idle" --> DISPATCH["Scheduler._dispatch()"]

    %% ---------- busy path ----------
    BUSY -- "busy" --> PUSH["RequestQueue.push<br/>FIFO, bounded"]
    PUSH -- "full" --> REJECT["Output: rejected<br/>'queue is full'"]
    PUSH -- "ok" --> QUEUED["Output: queued"]
    QUEUED -. "run_queue when idle" .-> DISPATCH

    %% ---------- navigation ----------
    DISPATCH --> NAV["navigate()<br/>descend tree, one SemIf choice per level"]

    NAV --> CAT["SemIf choice<br/>phase 'navigate:category'<br/>options = categories + descriptions<br/>+ create_category"]

    CAT --> CATWIN{"winner =<br/>create_category<br/>or tree empty?"}
    CATWIN -- "yes" --> CREATECAT["CreateCategory"]
    CATWIN -- "no" --> CATFIT["SemIf choice<br/>phase 'navigate:category_scope'<br/>P covers >= category_tau?"]
    CATFIT -- "rejected" --> CREATECAT
    CATFIT -- "covers" --> CANNED{"category in<br/>CANNED_CATEGORIES?"}

    CANNED -- "yes (response)" --> CANNEDLEAF["_navigate_canned<br/>phase 'navigate:response'<br/>closed canned reply tree"]
    CANNEDLEAF --> RUN["_run_skill()"]

    CANNED -- "no" --> HASSKILLS{"skills in<br/>category?"}
    HASSKILLS -- "none" --> CREATESKILL["CreateSkill"]
    HASSKILLS -- "one" --> SOLE["sole skill<br/>softmax skipped"]
    HASSKILLS -- "many" --> LEAF["SemIf choice<br/>phase 'navigate:leaf'<br/>options = existing skills only"]
    LEAF --> PICK["picked skill"]
    SOLE --> PICK

    PICK --> INTENT["SemIf choice<br/>phase 'navigate:intent'<br/>P same >= intent_tau?<br/>sole reuse-vs-create guard"]
    INTENT -- "different" --> CREATESKILL
    INTENT -- "same" --> RUN

    CREATECAT --> AUTHORING(["authoring pipeline<br/>see below"])
    CREATESKILL --> AUTHORING

    %% ---------- run ----------
    RUN --> WRITING{"leaf still<br/>writing?"}
    WRITING -- "yes" --> PENDING["Output: error<br/>'body still being written'"]
    WRITING -- "no (noop stub)" --> NOOP["Output: error<br/>'no body yet; restart <cat> <skill>'"]
    WRITING -- "runnable" --> RESOLVE["SkillRunner.run()<br/>resolve tiered config<br/>global -> category -> skill"]

    RESOLVE --> UNRES{"contract variable<br/>unresolved?"}
    UNRES -- "yes" --> PAUSE1["Output: needs_input<br/>pre_act pause<br/>ask one variable"]
    PAUSE1 -. "answer() bypasses queue" .-> RECORD["SemIf choice<br/>phase 'config:record'<br/>record as config vs ask each fire"]
    RECORD --> RESOLVE

    UNRES -- "no" --> ACT["skill.act(ctx, request)<br/>REAL action via stdlib transport<br/>values from ctx.config"]
    ACT --> ACTNI{"act asks<br/>for input?"}
    ACTNI -- "yes" --> PAUSE2["Output: needs_input<br/>act pause"]
    PAUSE2 -. "answer()" .-> ACT

    ACTNI -- "no" --> ASSESS

    %% ---------- assessment ----------
    ASSESS["SemIf choice<br/>phase 'assess:outcome'<br/>P success >= tau?"] --> OK{"success?"}
    OK -- "yes" --> SUMMARY
    OK -- "no" --> REQ["SemIf choice<br/>phase 'assess:requeue'<br/>complete vs retry"]
    REQ -- "retry" --> REQUEUE["requeue original request<br/>bounded by max_reentries"]
    REQUEUE -.-> PUSH
    REQ -- "complete" --> REPAIR

    SUMMARY["deterministic summary<br/>category.name: ok/failed - action_log"] --> OUT1["Output: ran<br/>REPL print · gateway reply · dashboard trace"]

    REPAIR["SemIf choice<br/>phase 'repair:choice'<br/>retry / repair_skill / ask_user / no_repair"] --> REPAIROUT["Output: repair offer<br/>surfaced in REPL/dashboard/gateway"]
    REPAIROUT -. "user confirms" .-> AUTHORING

    %% ---------- canned shortcut ----------
    RUN -. "canned category skips assess" .-> SUMMARY

    classDef semfill fill:#e8f0fe,stroke:#4285f4,stroke-width:2px;
    classDef output fill:#fce8e6,stroke:#d93025;
    classDef author fill:#e6f4ea,stroke:#34a853;
    class SCORE,SCORE2,INTERRUPT,CAT,CATFIT,LEAF,INTENT,ASSESS,REQ,REPAIR,RECORD semfill;
    class REJECT,QUEUED,PENDING,NOOP,PAUSE1,PAUSE2,OUT1,REPAIROUT output;
    class CREATECAT,CREATESKILL,AUTHORING author;
```

---

## Authoring pipeline (create / repair)

Entered from `CreateCategory`, `CreateSkill`, or a confirmed repair. Runs in
background single-slot workers so the main gate stays free.

```mermaid
flowchart TD
    START(["CreateCategory / CreateSkill / repair_skill"]) --> DRAFT["single-slot llm worker<br/>DraftAuthor queue"]

    DRAFT --> GEN{"category<br/>or skill?"}
    GEN -- "category" --> GC["generate_category()<br/>title + description"]
    GEN -- "skill" --> GS["generate_skill()<br/>title + description"]
    GC --> APPROVE
    GS --> APPROVE

    APPROVE{"creation_approval<br/>on?"}
    APPROVE -- "no / already approved" --> REGISTER
    APPROVE -- "yes" --> PROMPT["Output: approval prompt<br/>REPL / dashboard / gateway<br/>deny or timeout aborts"]
    PROMPT -- "approved" --> REGISTER
    PROMPT -- "denied / timeout" --> ABORT["Output: creation aborted<br/>nothing registered, not re-dispatched"]

    REGISTER["register stub in registry + tree<br/>trace 'category_created' / 'skill_created'"]
    REGISTER -- "category" --> CHAIN["chain into skill job<br/>approved=true"]
    CHAIN --> DRAFT
    REGISTER -- "skill" --> WRITE["single-slot codegen worker<br/>SkillWrite queue<br/>leaf.writing = true"]

    WRITE --> ELICIT["generate_elicitation()<br/>implementation questions +<br/>integration hint"]
    ELICIT --> QS["Output: questions<br/>posted one at a time<br/>answer_timeout per question"]
    QS -- "answers" --> BODY
    QS -- "timeout" --> BODY

    BODY["generate_skill_body()<br/>or regenerate_skill_body() for repair<br/>CODEGEN.md + bridge catalog<br/>declares INTEGRATION + flat CONTRACT"] --> FID["SemIf choice<br/>phase 'authoring:fidelity'<br/>accept / reconsider<br/>+ static integration_findings"]
    FID -- "reconsider" --> BODY
    FID -- "accept" --> CONTRACT["parse_contract()<br/>read body's CONTRACT constant"]

    CONTRACT --> CFG["SemIf choice per variable<br/>phase 'config:search'<br/>global -> category -> skill"]
    CFG --> TEST["generate_skill_tests()<br/>hermetic mechanics test"]
    TEST --> RUNTEST["run_skill_test()<br/>subprocess in skill folder"]
    RUNTEST -- "fail" --> REGEN["SemIf choice<br/>phase 'codegen_regen'<br/>regenerate code / test<br/>bounded by test_max_attempts"]
    REGEN --> BODY
    RUNTEST -- "pass" --> MAT["materialize_skill()<br/>write body + contract to data/skills/<br/>hot-merge into tree"]

    MAT --> REQ["requeue original request<br/>meta awaiting_skill_body<br/>trace 'skill_requeued'"] --> BACK(["back to main pipeline<br/>re-runs navigation onto new leaf"])

    FAIL["Output: skill_write_failed<br/>leaf stays a restartable stub<br/>restart <cat> <skill>"] 
    BODY -. "CodegenError / ValueError" .-> FAIL
    RUNTEST -. "budget exhausted" .-> FAIL

    classDef semfill fill:#e8f0fe,stroke:#4285f4,stroke-width:2px;
    classDef output fill:#fce8e6,stroke:#d93025;
    classDef author fill:#e6f4ea,stroke:#34a853;
    class FID,CFG,REGEN semfill;
    class PROMPT,ABORT,QS,FAIL output;
    class START,BACK,MAT,REQ author;
```

---

## Reading the path

| Stage | Decision (SemIf) | Phase logged | Output |
| --- | --- | --- | --- |
| Intake, idle | — (no decision) | — | dispatch immediately |
| Intake, busy | — (no decision) | — | FIFO queue (arrival order) |
| Category | which category | `navigate:category` | category / create |
| Category scope | does it cover? | `navigate:category_scope` | descend / create |
| Canned | which reply | `navigate:response` | canned line |
| Leaf | which skill | `navigate:leaf` | skill |
| Intent guard | same action? | `navigate:intent` | run / create |
| Config record | record or ask? | `config:record` | persisted / per-fire |
| Outcome | did it succeed? | `assess:outcome` | ok / failed |
| Requeue | complete or retry? | `assess:requeue` | done / requeue |
| Repair | how to recover? | `repair:choice` | offer |
| Fidelity | real action? | `authoring:fidelity` | accept / rewrite |
| Config search | where is the value? | `config:search` | auto-populate |
| Test regen | code or test? | `codegen_regen` | regenerate |

**The three terminal outputs** are: a run summary (`category.skill: ok/failed — …`),
a question the run is waiting on (`needs_input`), or an offer to fix/learn
(`repair_offer`) / an authoring acknowledgement for a brand-new capability.

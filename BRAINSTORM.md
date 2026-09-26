
## 1.Implementation Details For Skill Creation
- create folder for skill leaf instead of just a file. at the end of skill creation process, folder will contain four deliverables - skill.py, skill.test.py, contract.json, mock_data.json, and config.json

### skill.py
- prompt LLM generate skill. do not generate mock data at this point in the process. do not include the desire for mock data in the SKILL.md for skill creation. mocking and testing will have their own SKILL.md file for this purpose. In the codegen phase, the model should act like a dev who is talking to the product owner about requirements REPL is for questions about refining the product goal and requirements only.
    - defer to runner as authoritative data provider. for example, in an email sending skill, the sender address changes seldom, so it makes more sense as a config var. the receiver address changes often, so it would work better as an input variable. still, have the codegen treat them the same, and simply request this data from the runner.
    - lean toward asking more implementation questions (should sender address change?) during skill creation to ensure user satisfaction and reduce re-gen cycles. user is going to be more frustrated by the excessive failure/regen loop than excessive questions.
    
### test.py and contract.json
- *after cogeden is complete*, clear context then insert testgen SKILL.md and finished skill code. ask it to generate a data contract consisting of a single JSON file (contract.json)
- run test automatically with mock data
    - on failure, decide regen code or test
        - whichever you re-run, feed error and existing code and test files into context
    - on success, proceed to config.json section below

### config.json
- SemIf `choice` search for config vars (global config.json, category config.json) to auto-populate skill config.json
- on first fire of new skill, ask for each variable in the contract, with defaults set to the auto-populated values from semif search
- SemIf `choice` each answer: "record as config, or ask again each time skill fires?"
- on successive firings, ask only the questions not already answered by config vars; skill runner should load both into code at runtime

## 2.Personally identifying information
- Encrypted at rest (still JSON?)
- Decrypted at runtime by skill runner

## 3. MANUAL MODE
- user can toggle automatic or manual mode for everything handled by semif
- create abstraction layer that skills consume instead of base semif model
- abstraction layer routes to human input or semif or semif, and passes 
- manual mode returns semif-like results, with probabilities and confidences at 100%
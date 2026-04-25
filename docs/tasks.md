# DeepCFR Engine: Sub-Team Engineering Specs

Everyone *must* be able to use the provided setup environment. If you are unable, contact Corey or Nathaniel to get assistance in setting up the uv venv and applying it in the Leduc demo.

**Before starting any task:** Create your own branch, then create a subdirectory under `src/` with your name (e.g., `src/alice/`, `src/bob/`) and do all your work there. This keeps the codebase organized and makes merging straightforward.

## Data Setup

**Tasks 1 and 3 require pre-trained models and clustering data.** Download and extract the data archive:

1. Download `clusters_and_checkpoints.tar.gz` from [this Google Drive folder](https://drive.google.com/drive/folders/1GDiEH1ZB5uUtn5TMeLns02_2xH2U7evv?usp=sharing)
2. Move the file to the repo root and extract:
   ```bash
   tar -xzf clusters_and_checkpoints.tar.gz
   ```
3. Verify the directories exist: `src/clusters/` and `src/checkpoints/`

If you encounter issues downloading or extracting, contact Nathaniel or Corey.

---

## Task 1: GTO Frontend & Web GUI
**Allocation:** 2 Engineers

**Objective:** Build a web-based GUI that allows human users to play against our current GTO prototype policy network, wrapping the functionality of `poker_live.py`.

**Specifications:**
* **Foundation:** Study `src/pysrc/evaluation/poker_live.py` and understand how it works. Test it from the command line:
  ```
  python pysrc/evaluation/poker_live.py --net <path to .pt>
  ```
  Play a few hands to get a feel for the interface and game flow.

* **Web Frontend:** Build a web-based GUI (tech stack of your choice: Flask/React, FastAPI/Vue, etc.) that wraps the poker game logic. The interface should allow users to:
  - Play hands against the bot with a clear display of cards, pot, and decision options
  - View current game state and available actions
  
* **Optional Enhancements:** Beyond the core gameplay loop, consider adding features like:
  - Hand replay / history viewing
  - Game statistics (win rate, hand breakdowns, etc.)
  - Save/load game sessions
  - Custom bot parameters (bet sizing, strategy variations)

* **Deliverables:** All code lives in `src/yourname/`. Submit:
  - Web application code (document your tech stack and why you chose it)
  - Documentation on how to run the web server and play a game
  - A brief note on any enhancements you added and how they improve the user experience

---

## Task 2: Sequence-Based Opponent Classification (LSTM)
**Allocation:** 1 Engineer

**Objective:** Design and implement a recurrent neural network architecture that reads a historical sequence of betting actions and outputs a probability distribution over opponent behavioral archetypes.

**Specifications:**
* **Feature Engineering (Core Challenge):** The LSTM must ingest a sequential time series, but the representation needs careful design. Each time step currently consists of the 338-feature state representation concatenated with the actual action taken at that node. Study the InfoSet datatype (cppsrc/CFRTypes.h) and the visual structure below. Your first task is to determine how to re-work these individual decision node representations into a more continuous, learnable form that an LSTM can effectively process.

Here is a visualized example of a single InfoSet datatype in a human-readable format. 

  │  Street: TURN     [0 0 1 0]  Pos: BB
  │  Cards:  [24 32] [15 26 35] [47] [?]
  │  EHS:    81.3%
  │  Buckets: Fl:1960  Tu:7155  Rv:—
  │  Stack:  Me:0.7670  Opp:0.7470  (norm)
  │  Pot:0.4860  ToCall:0.0200  SPR:0.0789  (norm)
  │  BetHist:
  │    PF    [0.500] [0.000]  0.000   0.000   0.000   0.000 
  │    Flop  [0.333] [0.200] [0.750] [0.000]  0.000   0.000 
  │    Turn  [0.000] [0.043]  0.000   0.000   0.000   0.000 
  │    Riv    0.000   0.000   0.000   0.000   0.000   0.000 

* **Architecture:** Implement a complete PyTorch `nn.Module` that:
  - Defines input sequence tensor shapes after your feature engineering step
  - Specifies LSTM hidden dimension sizes (e.g., `HIDDEN_DIM = 256`)
  - Outputs a probability distribution over archetypes (e.g., `[0.5, 0.2, 0.1, 0.2]` where the number of classes is your design decision)
  - Includes cross-entropy loss function setup for training

* **Deliverables:** All code and documentation live in `src/yourname/`. Submit:
  - The `nn.Module` implementation
  - Documentation of your feature engineering choices and why you made them
  - A brief explanation of the archetype count you chose and how the architecture produces the final distribution

---

## Task 3: Zenodo PHH Binary Parser
**Allocation:** 2 Engineers

**Objective:** Convert raw, human-readable Poker Hand Histories (PHH) into two optimized binary datasets: one containing game states in the `InfoSet` format, and one containing the corresponding actions/outcomes in a compact format.

**Specifications:**
* **The Parsing Logic:** Write a Python and/or C++ script that reads raw text logs and reconstructs the betting history and card states from the perspective of each player.

* **State Encoding:** At every action in the log, extract exact information (player positions, community cards, pot size, etc.) and pack it into the custom 168-byte `InfoSet` C++ struct. Select only games that are actually HUNL (Heads-Up No-Limit) and where all information is available. (Some datasources may not disclose hole cards unless a showdown occurred.)

* **Off-Tree Bet Handling:** Some bets may not align with decision nodes. Map these as a combination of probabilities from the nearest valid nodes. For example, a 0.66 bet size maps as a weighted blend of the 0.5 and 0.75 nodes. If you need clarification on this or any aspect of the task, ask.

* **Deliverables:** All code and data stay self-contained in `src/yourname/`. Submit two binary files:
  - **inputs.bin:** Sequential `InfoSet` structs (168 bytes each) representing all game states
  - **outputs.bin:** Corresponding actions/outcomes in a compact binary format of your choice (document your encoding scheme)
  
  These files should be efficient enough to memory-map for training. Include documentation explaining your output encoding and any design decisions. 
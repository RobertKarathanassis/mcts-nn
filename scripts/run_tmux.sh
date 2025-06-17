#!/usr/bin/env bash
# path: scripts/run_tmux.sh

REPO="/home/rob/github/mcts-nn"
SESSION="alphazero"

tmux new-session -d -s $SESSION

# pane 0 ─ trainer
tmux send-keys -t $SESSION \
  "cd $REPO && python -m scripts.trainer --gpu 0" C-m

# pane 1 ─ self-play workers
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION \
  "cd $REPO && python -m scripts.selfplay_worker --games 999999 --gpu 0 --workers 4" C-m

# pane 2 ─ arena loop
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION \
  "cd $REPO && python -m scripts.arena --games 10 --sims 30 --gpus 0" C-m

tmux select-layout -t $SESSION tiled
tmux attach -t $SESSION

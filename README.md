# Q and DeepQ Reinforcement Learning for Nim 

Four [Nim](https://en.wikipedia.org/wiki/Nim) (last player to take loses) players are available:

* [Human Player](elements/players/human.py) (CLI inputs define the moves)
* [Random Player](elements/players/random.py) (choosing random valid moves)
* [Q Player](elements/players/ai_q.py) (learning with q-table)
* [DeepQ Player](elements/players/deep_q.py) (learning q-table through NN)

plus some improved variants of the latter. The game can be started through the [main.py](main.py)-file.

import os

os.system('pwd')
# os.system('conda activate tempEnv')
# os.system('python -m pip install gpytorch')
os.system("python replay_trainer.py --state_dict_path output_weights/2_epochs_2_vertices.pt --train_epochs 50 --n_vertices 2")
# os.system("/tmp/loss-surface-refactored/replay_trainer --state_dict_path output_weights/10_epochs_5_vertices.pt --train_epochs 50 --n_vertices 9")
# os.system("/tmp/loss-surface-refactored/ensemble_trainer_2.py --model output_weights/10_epochs_5_vertices.pt --simplex_epochs 10 --n_verts 5 --task_number 1")
# os.system("/tmp/loss-surface-refactored/replay_trainer --state_dict_path output_weights/15_epochs_5_vertices.pt --train_epochs 75 --n_vertices 9")
# os.system("/tmp/loss-surface-refactored/ensemble_trainer_2.py --model output_weights/15_epochs_5_vertices.pt --simplex_epochs 15 --n_verts 5 --task_number 1")

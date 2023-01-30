from procgen_tools.graph_mazes import save_movie

# num is the number of environments to create in the vec env
# num_levels is the number of levels to sample from (0 for infinite, 1 for deterministic)
venv = ProcgenGym3Env(
    num=3, env_name='maze', num_levels=1, start_level=0,
    distribution_mode='hard', num_threads=1, render_mode="rgb_array",
)
venv = maze.wrap_venv(venv)

try:
    with open('./saved-mazes.pkl', 'rb') as f:
        state_bytes_list = pickle.load(f)
        if len(state_bytes_list) != venv.num_envs:
            print(f'WARN: saved {len(state_bytes_list)} envs but num_envs is {venv.num_envs}')
        else:
            venv.env.callmethod('set_state', state_bytes_list)

except FileNotFoundError:
    print('No ./saved-mazes.pkl file exists, using default envs')
        
# download from https://drive.google.com/file/d/1db1qJn_ZM49CUcA1k7uSMdsdiAQ4eR5a/view?usp=share_link
policy = load_policy('../trained_models/maze_I/model_rand_region_5.pth', action_size=venv.action_space.n, device=t.device('cpu'))

done = np.zeros(venv.num_envs)
obs = venv.reset()

save_movie(venv, policy, condition_names=['far', 'near', 'vanished'])
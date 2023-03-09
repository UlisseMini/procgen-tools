import torch as t
from IPython.display import Video
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array, vfx

import circrl.rollouts as cro
import procgen_tools.maze as maze

# Predict func for rollouts
def get_predict(plcy):
    def predict(obs, deterministic):
        #obs = t.flip(t.FloatTensor(obs), dims=(-1,))
        obs = t.FloatTensor(obs)
        last_obs = obs
        dist, value = plcy(obs)
        if deterministic:
            act = dist.mode.numpy()
        else:
            act = dist.sample().numpy()
        return act, None, dist.logits.detach().numpy()
    return predict

# Run rollout and return a video clip
def rollout_video_clip(predict, level, remove_cheese=False, 
        mouse_inner_pos=None,
        mouse_outer_pos=None):
    venv = maze.create_venv(1, start_level=level, num_levels=1)
    # Remove cheese
    if remove_cheese:
        maze.remove_cheese(venv)
    # Place mouse if specified (no error checking)
    env_state = maze.EnvState(venv.env.callmethod('get_state')[0])
    if mouse_inner_pos is not None:
        padding = (env_state.world_dim - env_state.inner_grid().shape[0]) // 2
        mouse_outer_pos = (mouse_inner_pos[0] + padding,
            mouse_inner_pos[1] + padding)
    if mouse_outer_pos is not None:
        env_state.set_mouse_pos(mouse_outer_pos[1], mouse_outer_pos[0])
        venv.env.callmethod('set_state', [env_state.state_bytes])
    # Rollout
    seq, _, _ = cro.run_rollout(predict, venv, max_episodes=1, max_steps=256)
    vid_fn, fps = cro.make_video_from_renders(seq.renders)
    rollout_clip = VideoFileClip(vid_fn).margin(10)
    # try:
    #     txt_clip = TextClip("GeeksforGeeks", fontsize = 75, color = 'black') 
    #     txt_clip = txt_clip.set_pos('center').set_duration(10) 
    #     final_clip = CompositeVideoClip([rollout_clip, txt_clip]) 
    # except OSError as e:
    #     print('Cannot add text overlays, maybe ImageMagick is missing?  Try sudo apt install imagemagick')
    #     final_clip = rollout_clip
    final_clip = rollout_clip
    return seq, final_clip

# Run rollouts with multiple predict functions, stack the videos side-by-side and return
def side_by_side_rollout(predicts_dict, levels, remove_cheese=False, num_cols=2,
        mouse_inner_pos=None,
        mouse_outer_pos=None):
    policy_descs = list(predicts_dict.keys())
    policy_descs_grid = [policy_descs[x:x+num_cols] for x in 
        range(0, len(policy_descs), num_cols)]
    print(f'Levels:{levels}, cheese:{not remove_cheese}, policies:{policy_descs_grid}')
    clips = []
    seqs = []
    try:
        _ = (level for level in levels)
    except TypeError:
        levels = [levels]
    for level in levels:
        for desc, predict in predicts_dict.items():
            seq, clip = rollout_video_clip(predict, level, remove_cheese, mouse_inner_pos,
                mouse_outer_pos)
            clips.append(clip)
            seqs.append(seq)
    clips_grid = [clips[x:x+num_cols] for x in range(0, len(clips), num_cols)]
    final_clip = clips_array(clips_grid)
    stacked_fn = 'stacked.mp4'
    final_clip.resize(width=600).write_videofile(stacked_fn, logger=None)
    return Video(stacked_fn, embed=True), seqs
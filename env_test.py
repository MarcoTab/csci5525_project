import retro

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')

obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
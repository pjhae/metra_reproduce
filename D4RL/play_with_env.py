import gym
import d4rl # Import required to register environments, you may need to also import the submodule

# Create the environment
# env = gym.make('kitchen_relax-v1')
env = gym.make('hopper-medium-v0')

# d4rl abides by the OpenAI gym interface
env.reset()

for i in range(1000):
	
	env.step(env.action_space.sample())
	# print(env.render('human'))

import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm

# make our generative model with two components, a fair die and a
# loaded die
print("Building generative model...")
gen_model = hmm.CategoricalHMM(n_components=2, random_state=99)

# the first state is the fair die so let's start there so no one
# catches on right away
print("Creating initial data with fair die...")
gen_model.startprob_ = np.array([1.0, 0.0])

# now let's say that we sneak the loaded die in:
# here, we have a 95% chance to continue using the fair die and a 5%
# chance to switch to the loaded die
# when we enter the loaded die state, we have a 90% chance of staying
# in that state and a 10% chance of leaving
print("Creating data with loaded die...")
gen_model.transmat_ = np.array([[0.95, 0.05],
                                [0.1, 0.9]])

# now let's set the emission means:
# the first state is a fair die with equal probabilities and the
# second is loaded by being biased toward rolling a six
print("Setting emission means...")
gen_model.emissionprob_ = \
    np.array([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
              [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2]])

# simulate the loaded dice rolls
n = 30000
print(f"Simulating {n} die rolls...")
rolls, gen_states = gen_model.sample(n)

# plot states over time, let's just look at the first rolls for clarity
print("Plotting data...")
fig, ax = plt.subplots()
ax.plot(gen_states[:500])
ax.set_title('States over time')
ax.set_xlabel('Time (# of rolls)')
ax.set_ylabel('State')

print("States over time:")
plt.show()

# plot rolls for the fair and loaded states
fig, ax = plt.subplots()
ax.hist(rolls[gen_states == 0], label='fair', alpha=0.5,
        bins=np.arange(7) - 0.5, density=True)
ax.hist(rolls[gen_states == 1], label='loaded', alpha=0.5,
        bins=np.arange(7) - 0.5, density=True)
ax.set_title('Roll probabilities by state')
ax.set_xlabel('Count')
ax.set_ylabel('Roll')
ax.legend()

print("Roll probabilities by state:")
plt.show()

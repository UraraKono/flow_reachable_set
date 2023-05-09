import matplotlib.pyplot as plt
import deeptime

n_particles = 1000
dataset = deeptime.data.bickley_jet(n_particles, n_jobs=8)

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16, 10))

for t, ax in zip([0, 1, 2, 200, 300, 400], axes.flatten()):
    ax.scatter(*dataset[t].T, c=dataset[0, :, 0], s=50)
    ax.set_title(f"Particles at t={t}")
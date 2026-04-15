import matplotlib.pyplot as plt
import os

def plot_lifesat_distribution(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    df["lifesat"].hist()

    plt.title("Life Satisfaction Distribution")

    plt.savefig(f"{output_dir}/lifesat_distribution.png")
    plt.close()
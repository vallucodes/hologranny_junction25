import numpy as np
import pandas as pd
import sqlite3

samples = 2000

# Set seed for reproducibility
# np.random.seed(42)

def generate_real_inspired_post_sauna_data(num_sessions=samples):  # Increase to 200+
    data = []
    for _ in range(num_sessions):
        # Pre-sauna metrics
        pre_sleep = np.random.uniform(50, 90)
        pre_stress = np.random.uniform(2, 8)
        pre_readiness = np.random.uniform(60, 95)

        # Sauna parameters
        avg_temp = np.clip(np.random.normal(85, 8), 70, 100)
        target_heat_load = np.random.uniform(100, 125)
        avg_hum = np.clip(target_heat_load - avg_temp, 10, 40)
        duration = np.clip(np.random.normal(45, 20), 15, 120)

        # STRONGER personalization signals
        # High stress → needs MUCH gentler session
        stress_factor = (8 - pre_stress) / 6  # 0 to 1, higher = less stressed

        # Readiness affects duration tolerance
        readiness_factor = (pre_readiness - 60) / 35  # 0 to 1

        # Sleep affects how much heat helps
        sleep_factor = (pre_sleep - 50) / 40  # 0 to 1

        # Personalized optima with STRONGER effects
        optimal_heat_load = 95 + stress_factor * 20  # 95-115 based on stress
        optimal_duration = 35 + readiness_factor * 30  # 35-65 based on readiness

        # Calculate scores
        heat_load = avg_temp + avg_hum
        heat_load_score = np.exp(-((heat_load - optimal_heat_load) ** 2) / 300)
        duration_score = np.exp(-((duration - optimal_duration) ** 2) / 600)

        # Overall benefit - MULTIPLY by factors to make pre-metrics matter more
        overall_benefit = duration_score * heat_load_score * (0.5 + 0.5 * stress_factor)

        # Convert to metric changes
        stress_reduction = overall_benefit * 3.5 + np.random.normal(0, 0.3)
        post_stress = np.clip(pre_stress - stress_reduction, 0, 10)

        sleep_improvement = overall_benefit * 18 + np.random.normal(0, 4)
        post_sleep = np.clip(pre_sleep + sleep_improvement, 0, 100)

        readiness_improvement = overall_benefit * 22 + np.random.normal(0, 4)
        post_readiness = np.clip(pre_readiness + readiness_improvement, 0, 100)

        data.append([pre_sleep, pre_stress, pre_readiness, avg_temp, avg_hum, duration,
                     post_sleep, post_stress, post_readiness])

    columns = ['pre_sleep', 'pre_stress', 'pre_readiness', 'avg_temp', 'avg_hum', 'duration',
               'post_sleep', 'post_stress', 'post_readiness']
    return pd.DataFrame(data, columns=columns)

def save_to_database(df, db_path='database.db'):
	"""
	Save the generated data to SQLite database.
	"""
	conn = sqlite3.connect(db_path)

	# Create table with appropriate schema
	df.to_sql('sauna_sessions', conn, if_exists='replace', index=False)

	conn.commit()
	conn.close()
	print(f"Successfully saved {len(df)} records to {db_path}")

if __name__ == "__main__":
	# Generate data
	print("Generating sauna session data...")
	df = generate_real_inspired_post_sauna_data(num_sessions=samples)

	# Display sample
	print("\nSample of generated data:")
	print(df.head())

	print("\nData statistics:")
	print(df.describe())

	# Save to database
	save_to_database(df)

	print("\nData generation complete!")

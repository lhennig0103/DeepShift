from codecarbon import EmissionsTracker

def dummy_model():
    # Simulating some computation
    for _ in range(1000000):
        _ = 2 + 2

if __name__ == "__main__":
    emissions_target = 0.1  # Arbitrary target emissions in kg

    # Run the model and track emissions
    for i in range(1):
        tracker = EmissionsTracker()
        tracker.start()
        
        dummy_model()

        # Assuming 'tracker' is your EmissionsTracker object
        print(dir(tracker))

        emissions = tracker.stop()
        result = abs(emissions - emissions_target)
        # print(f"Iteration {i}, Emissions: {emissions} kg, Result: {result}")
        timestamp, emissions_in_kg, energy_consumption_in_kwh = get_emissions_data_from_file()




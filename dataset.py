import fastf1
from fastf1 import logger
import pandas as pd
import numpy as np
import time
from typing import Callable, Any
from functools import wraps
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt


# logger.set_log_level('ERROR')
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"  # type: ignore
#TODO: Add offline option to use cached data

def track_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result

    return wrapper


@track_time
def add_weather_info(session, stints_df):
    # TODO: Segmentar clima por stint
    """
    Adds mean weather information from the session to the stints dataframe

    Args:
        session: FastF1 session object containing weather data
        stints_df: DataFrame containing stint data

    Returns:
        DataFrame with added weather columns
    """
    # Get weather data from session
    weather = session.weather_data

    # Calculate mean values
    mean_weather = weather.mean()

    # Add weather columns with broadcast values
    for col in mean_weather.index:
        stints_df[col] = mean_weather[col]

    return stints_df


@track_time
def add_starting_positions(session, stints_df):
    """
    Adds starting grid position for each driver in the stints dataframe

    Args:
        session: FastF1 session object containing grid position info
        stints_df: DataFrame containing stint data

    Returns:
        DataFrame with added StartingPosition column
    """
    # Get starting grid positions
    grid = session.results[["Abbreviation", "GridPosition"]]
    grid = grid.set_index("Abbreviation")
    grid_dict = grid["GridPosition"].to_dict()

    # Add starting position column based on driver
    stints_df["StartingPosition"] = stints_df["Driver"].map(grid_dict)

    return stints_df


@track_time
def add_team_info(session, stints_df):
    """
    Adds team information for each driver in the stints dataframe

    Args:
        session: FastF1 session object containing driver info
        stints_df: DataFrame containing stint data

    Returns:
        DataFrame with added Team column
    """
    # Create driver to team mapping
    driver_teams = {}
    for driver in session.drivers:
        driver_info = session.get_driver(driver)
        driver_teams[driver_info["Abbreviation"]] = driver_info["TeamName"]

    # Add team column based on driver
    stints_df["Team"] = stints_df["Driver"].map(driver_teams)

    return stints_df


@track_time
def get_stints_race(session):
    laps = session.laps
    drivers = session.drivers
    drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]
    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={"LapNumber": "StintLength"})
    return stints


def process_lap(lap):
    """
    Processes telemetry for a single lap to calculate performance metrics.

    Args:
        lap: A FastF1 Lap object.

    Returns:
        A dictionary with calculated metrics for the lap, or None on failure.
    """
    try:
        # get_telemetry() loads data on-demand for this specific lap
        telemetry = lap.get_telemetry().copy()

        required_cols = ["Time", "Brake", "Speed", "nGear"]
        if telemetry.empty or not all(
            col in telemetry.columns for col in required_cols
        ):
            return None

        # Calculate time delta between telemetry points for accurate duration calculation
        telemetry["TimeDelta"] = telemetry["Time"].diff().dt.total_seconds()

        # Calculate total time braking during the lap
        total_braking_time = telemetry.loc[
            telemetry["Brake"] == True, "TimeDelta"
        ].sum()

        # Calculate speed and gear-related metrics
        avg_speed = telemetry["Speed"].mean()
        speed_delta = telemetry["Speed"].max() - telemetry["Speed"].min()
        gear_changes = telemetry["nGear"].diff().ne(0).sum()

        return {
            "brake_duration": total_braking_time if total_braking_time > 0 else np.nan,
            "avg_speed": avg_speed,
            "speed_delta": speed_delta,
            "gear_changes": gear_changes,
        }
    except Exception as e:
        # Log errors for individual laps without stopping the entire process
        print(
            f"Could not process Lap {lap.LapNumber} for Driver {lap.Driver}. Error: {e}"
        )
        return None


def process_stint(session, driver, stint_number):
    """
    Worker function to process telemetry for a single stint sequentially.

    It iterates through all laps for a given stint, processes them one by one,
    and aggregates the results.

    Returns:
        A dictionary containing the driver, stint number, and aggregated metrics.
        Returns None if processing fails.
    """
    try:
        # A more efficient way to select laps for a specific driver and stint
        stint_laps = session.laps.pick_drivers([driver])
        stint_laps = stint_laps[stint_laps["Stint"] == stint_number]

        if stint_laps.empty:
            return None

        laps_to_process = [lap for _, lap in stint_laps.iterlaps()]

        if not laps_to_process:
            return None

        # --- Sequential Lap Processing ---
        valid_results = []
        for lap in laps_to_process:
            lap_result = process_lap(lap)
            if lap_result:
                valid_results.append(lap_result)

        # --- Aggregation ---
        if not valid_results:
            return None

        # Unpack the list of dictionaries into separate lists for aggregation
        lap_brake_durations = [
            r["brake_duration"]
            for r in valid_results
            if not np.isnan(r["brake_duration"])
        ]
        lap_avg_speeds = [r["avg_speed"] for r in valid_results]
        lap_speed_deltas = [r["speed_delta"] for r in valid_results]
        lap_gear_changes = [r["gear_changes"] for r in valid_results]

        # Aggregate the per-lap metrics into stint-level statistics
        results = {
            "Driver": driver,
            "Stint": stint_number,
            "mean_brake_time": np.mean(lap_brake_durations) if lap_brake_durations else np.nan,
            "std_brake_time": np.std(lap_brake_durations) if lap_brake_durations else np.nan,
            "AvgSpeed": np.mean(lap_avg_speeds) if lap_avg_speeds else np.nan,
            "StdSpeed": np.std(lap_avg_speeds) if lap_avg_speeds else np.nan,
            "AvgSpeedDelta": np.mean(lap_speed_deltas) if lap_speed_deltas else np.nan,
            "StdSpeedDelta": np.std(lap_speed_deltas) if lap_speed_deltas else np.nan,
            "AvgGearChanges": np.mean(lap_gear_changes) if lap_gear_changes else np.nan,
        }
        return results

    except Exception as e:
        print(f"Could not process Stint {stint_number} for {driver}. Error: {e}")
        return None


@track_time
def add_stint_telemetry(session, stints_df):
    """
    Adds telemetry-derived metrics to a stints DataFrame by processing each
    stint and its laps sequentially.

    Args:
        session: A loaded FastF1 session object.
        stints_df: A pandas DataFrame with 'Driver' and 'Stint' columns.

    Returns:
        pandas.DataFrame: The original stints_df with new columns for telemetry.
    """
    if not hasattr(session.laps, "iloc"):
        session.load(laps=True, telemetry=True, weather=False, messages=False)

    results = []
    # This loop iterates through all stints sequentially and displays a single progress bar.
    for _, stint in tqdm(list(stints_df.iterrows()), desc="Processing all stints"):
        driver = stint["Driver"]
        stint_number = stint["Stint"]

        result = process_stint(session, driver, stint_number)

        if result:
            results.append(result)

    if not results:
        print("Warning: No stints were processed successfully.")
        return stints_df

    results_df = pd.DataFrame(results)

    # Merge the new telemetry metrics back into the original stints DataFrame
    stints_df = stints_df.merge(results_df, on=["Driver", "Stint"], how="left")

    return stints_df


def main(year_start, year_end):
    """
    Main function to get F1 data for a range of years from the FastF1 API.
    """
    for year in range(year_start, year_end + 1):
        calendar = fastf1.get_event_schedule(year, include_testing=False)

        all_stints = []

        for idx, event in calendar.iterrows():
            try:
                session = fastf1.get_session(year, event["EventName"], "R")
                session.load()

                race_stints = get_stints_race(session)
                race_stints = add_team_info(session, race_stints)
                race_stints = add_starting_positions(session, race_stints)
                race_stints = add_weather_info(session, race_stints)
                race_stints = add_stint_telemetry(session, race_stints)

                race_stints["Year"] = year
                race_stints["Circuit"] = event["EventName"]

                all_stints.append(race_stints)

                print(f"Processed {year} {event['EventName']}")
            except Exception as e:
                print(f"Error processing {year} {event['EventName']}: {e}")

        if all_stints:
            year_stints = pd.concat(all_stints, ignore_index=True)

            # Save the data for this year (optional)
            year_stints.to_csv(f".\\data\\stints_data_{year}.csv", index=False)


if __name__ == "__main__":
    YEAR_START = 2019
    YEAR_END = 2024

    main(YEAR_START, YEAR_END)

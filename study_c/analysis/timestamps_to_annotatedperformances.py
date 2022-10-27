# Alternative to _annotatedscenes.py
# Does not split into shots, but keeps performances as one sequence
import cv2
import pandas as pd

from olympics.oly_modules import scrape_resulttables


def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


# csv [team, ts_perf_start, ts_perf_end]
timestamps = pd.read_csv(r'olympics/manual_timestamps/'
                         r'timestamps_torino2006.csv')

# full replay
video_path = 'olympics/full_replays/torino2006_fullreplay.mp4'
vid = cv2.VideoCapture(video_path)
fps = int(vid.get(cv2.CAP_PROP_FPS))

# query link for competition results
results_url = 'https://skatingscores.com/q/event/' \
              '?show_ranks=on&underline=&season_codes=2006&' \
              'division_codes=sr&event_codes=oly&discipline_codes=pairs&' \
              'unit_country_codes=all&unit_name=%25&sort=score&limit=50&' \
              'submit=Submit'

# scrape results from competition
competition_results = scrape_resulttables.get_table(results_url)
competition_results.columns = competition_results.columns.str.replace(' ', '')

# add scores to performances
scenes_annotated = timestamps.rename(columns={"team": "Team",
                                              "ts_perf_start": "ts_start",
                                              "ts_perf_end": "ts_end"})\
    .merge(competition_results, on='Team', how="inner")
if len(scenes_annotated) != len(timestamps):
    print("Matching of scenes with scores resulted in loss of rows - "
          "Some rows couldn't be matched")

# drop na
scenes_annotated = scenes_annotated.drop(10)
scenes_annotated = scenes_annotated.drop(11).reset_index()
scenes_annotated = scenes_annotated.iloc[:, 1:]

scenes_annotated['frame_start'] = scenes_annotated\
    .apply(lambda row: get_sec(row.ts_start)*fps, axis=1)
scenes_annotated['frame_end'] = scenes_annotated\
    .apply(lambda row: get_sec(row.ts_end)*fps, axis=1)
scenes_annotated.to_csv('olympics/competition_performances_with_results/'
                        'performances-results_torino2006.csv')
print("results saved as csv")

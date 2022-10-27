import pandas as pd

from olympics.oly_modules import scrape_resulttables, shot_transition_detection

# csv [team, ts_perf_start, ts_perf_end]
timestamps = pd.read_csv(r'../manual_timestamps/timestamps_torino2006.csv')

# import full Olympics replay
video_path = '../full_replays/torino2006_fullreplay.mp4'

# query link for competition results
results_url = 'https://skatingscores.com/q/event/?show_ranks=on&underline=&' \
              'season_codes=2006&division_codes=sr&event_codes=oly&' \
              'discipline_codes=pairs&unit_country_codes=all&unit_name=%25&' \
              'sort=score&limit=50&submit=Submit'
print("start scene detection")
# find individual video shots
scenes = shot_transition_detection.find_scenes(video_path, threshold=30.0)
print("finished scene detection")

# scrape results from competition
competition_results = scrape_resulttables.get_table(results_url)
competition_results.columns = competition_results.columns.str.replace(' ', '')

# only keep those video shots that show an actual skating performance
# add team names to individual scenes
# scenes_teams indexes every scene, holds its start and end times;
# and matches it with the team name
scenes_within_timestamps, scenes_teams = shot_transition_detection\
    .get_scenes_within_timestamps(scenes, timestamps)

# add scores to individual scenes
scenes_annotated = scenes_teams.rename(columns={"team": "Team"})\
    .merge(competition_results, on='Team', how="inner")
if len(scenes_annotated) != len(scenes_teams):
    print("Matching of scenes with scores resulted in loss of rows - "
          "Some rows couldn't be matched")

scenes_annotated.to_csv('competition_scenes_with_results/'
                        'scenes-results_torino2006.csv')
print("results saved as csv")

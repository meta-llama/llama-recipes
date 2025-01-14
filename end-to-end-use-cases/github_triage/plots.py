import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from utils import fetch_github_endpoint
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

def plot_views_clones(repo_name, out_folder):
    def json_to_df(json_data, key):
        df = pd.DataFrame(json_data[key])
        df['timestamp'] = df['timestamp'].apply(lambda x: x[5:10])
        if key in ['clones', 'views']:
            df.rename(columns={'uniques': key}, inplace=True)
            df.drop(columns=['count'], inplace=True)
        return df

    unique_clones_2w = fetch_github_endpoint(f"https://api.github.com/repos/{repo_name}/traffic/clones").json()
    unique_views_2w = fetch_github_endpoint(f"https://api.github.com/repos/{repo_name}/traffic/views").json()

    df1 = json_to_df(unique_clones_2w, 'clones')
    df2 = json_to_df(unique_views_2w, 'views')

    df = df1.merge(df2, on='timestamp', how='inner')

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['timestamp'], df['views'], color='blue')
    ax1.set_xlabel('Day', fontsize=18)
    ax1.set_ylabel('Unique Views', color='blue', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.bar(df['timestamp'], df['clones'], color='red')
    ax2.set_ylabel('Unique Clones', color='red', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Views & Clones in the last 2 weeks', fontsize=24)
    plt.savefig(f'{out_folder}/views_clones.png', dpi=120)  
    plt.close()

def plot_high_traffic_resources(repo_name, out_folder):
    popular_paths_2w = fetch_github_endpoint(f"https://api.github.com/repos/{repo_name}/traffic/popular/paths").json()
    df = pd.DataFrame(popular_paths_2w)
    df['path'] = df['path'].apply(lambda x: '/'.join(x.split('/')[-2:]))
    df = df.sort_values(by='uniques', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(df['path'], df['uniques'])
    plt.xlabel('Unique traffic in the last 2 weeks', fontsize=18)
    # plt.ylabel('Resource', fontsize=18, labelpad=15)
    plt.title("Popular Resources on the Repository", fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{out_folder}/resources.png', dpi=120)
    plt.close()
    
def plot_high_traffic_referrers(repo_name, out_folder):
    popular_referrer_2w = fetch_github_endpoint(f"https://api.github.com/repos/{repo_name}/traffic/popular/referrers").json()
    df = pd.DataFrame(popular_referrer_2w)
    df = df.sort_values(by='uniques', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(df['referrer'], df['uniques'])
    plt.xlabel('Unique traffic in the last 2 weeks', fontsize=18)
    plt.ylabel('Referrer', fontsize=18)
    plt.title("Popular Referrers to the Repository", fontsize=24)
    plt.savefig(f'{out_folder}/referrers.png', dpi=120)
    plt.close()

def plot_commit_activity(repo_name, out_folder):
    limit = 10
    today = pd.to_datetime('today')
    weekly_commit_count_52w = fetch_github_endpoint(f"https://api.github.com/repos/{repo_name}/stats/participation").json()['all'][-limit:]
    timestamps = [(today - pd.Timedelta(days=7*(i+1))) for i in range(limit)]
    df = pd.DataFrame({'timestamp': timestamps, 'commit_count': weekly_commit_count_52w})

    plt.figure(figsize=(10, 6))
    plt.bar(df['timestamp'], df['commit_count'])
    plt.xlabel('Week', fontsize=18)
    plt.ylabel('Commit Count', fontsize=18)
    plt.title(f"Commits in the last {limit} weeks", fontsize=24)
    plt.savefig(f'{out_folder}/commits.png', dpi=120)
    plt.close()

def plot_user_expertise(df, out_folder):
    d = df.to_dict('records')[0]
    levels = ['Beginner', 'Intermediate', 'Advanced']
    keys = [f"op_expertise_count_{x.lower()}" for x in levels]
    data = pd.DataFrame({'Expertise': levels, 'Count': [d.get(k, 0) for k in keys]})

    plt.figure(figsize=(10, 6))
    plt.barh(data['Expertise'], data['Count'])
    plt.xlabel('Count', fontsize=18)
    plt.title('User Expertise', fontsize=24)
    plt.savefig(f'{out_folder}/expertise.png', dpi=120)
    plt.close()

def plot_severity(df, out_folder):
    d = df.to_dict('records')[0]
    levels = ['Trivial', 'Minor', "Major", 'Critical']
    keys = [f"severity_count_{x.lower()}" for x in levels]
    data = pd.DataFrame({'Severity': levels, 'Count': [d.get(k, 0) for k in keys]})
    plt.figure(figsize=(10, 6))
    plt.barh(data['Severity'], data['Count'])
    plt.xlabel('Count', fontsize=18)
    plt.title('Severity', fontsize=24)
    plt.savefig(f'{out_folder}/severity.png', dpi=120)
    plt.close()

def plot_sentiment(df, out_folder):
    d = df.to_dict('records')[0]
    levels = ['Positive', 'Neutral', 'Negative']
    keys = [f"sentiment_count_{x.lower()}" for x in levels]
    data = pd.DataFrame({'Sentiment': levels, 'Count': [d.get(k, 0) for k in keys]})
    plt.figure(figsize=(10, 6))
    plt.barh(data['Sentiment'], data['Count'])
    plt.xlabel('Count', fontsize=18)
    plt.title('Sentiment', fontsize=24)
    plt.savefig(f'{out_folder}/sentiment.png', dpi=120)
    plt.close()
        
def plot_themes(df, out_folder):
    d = df.to_dict('records')[0]
    levels = ['Documentation', 'Installation and Environment', 'Model Inference', 'Model Fine Tuning and Training', 'Model Evaluation and Benchmarking', 'Model Conversion', 'Cloud Compute', 'CUDA Compatibility', 'Distributed Training and Multi-GPU', 'Invalid', 'Miscellaneous']
    keys = [f'themes_count_{x.lower().replace(" ", "_").replace("-", "_")}' for x in levels]
    data = pd.DataFrame({'Theme': levels, 'Count': [d.get(k, 0) for k in keys]})
    plt.figure(figsize=(10, 6))
    plt.barh(data['Theme'], data['Count'])
    plt.xlabel('Count', fontsize=18)
    plt.title('Themes', fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{out_folder}/themes.png', dpi=120)
    plt.close()
  
def issue_activity_sankey(df, out_folder):
    
    d = df.to_dict('records')[0]
    label = ["New Issues", "Issues Under Discussion", "Issues Discussed and Closed", "Issues Not Responded to", "Issues Closed Without Discussion"]
    values = [
        d['issues_created'], 
        d['open_discussion'] + d['closed_discussion'],  # 7
        d['closed_discussion'], # 3
        d['open_no_discussion'] + d['closed_no_discussion'],
        d['closed_no_discussion'] 
    ]

    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = [f"{l} ({values[i]})" for i, l in enumerate(label)],
        color = ["#007bff", "#17a2b8", "#6610f2", "#dc3545", "#6c757d"]  # color scheme to highlight different flows
        ),
        link = dict(
        source = [0, 1, 0, 3], # indices correspond to labels, eg A1, A2, etc
        target = [1, 2, 3, 4],
        value = [v if v > 0 else 1e-9 for v in values[1:]]
    ))])

    fig.update_layout(title_text="Issue Flow", font_size=16)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))  # adjust margins to make text more visible
    fig.write_image(f"{out_folder}/engagement_sankey.png")


def draw_all_plots(repo_name, out_folder, overview):
    func1 = [plot_views_clones, plot_high_traffic_resources, plot_high_traffic_referrers, plot_commit_activity]
    func2 = [plot_user_expertise, plot_severity, plot_sentiment, plot_themes, issue_activity_sankey]
    logger.info("Plotting traffic trends...")
    for func in func1:
        try:
            func(repo_name, out_folder)
        except:
            print(f"Github fetch failed for {func}. Make sure you have push-access to {repo_name}!")
    logger.info("Plotting issue trends...")
    for func in func2:
        func(overview, out_folder)
    